# Decode micro-batching across TP groups (Maya ds4, 6x3090, tp2 x 3 groups, --parallel 4)

Analysis only (no GPU run). Tree: /home/michal/project-maya-p2p145opt (p2p-145-opt, v1.1.45 + DS4_AR_P2P + DS4_XFER_P2P).
All refs: `M` = src/ds4/ds4_model.cu, `C` = src/ds4/ds4_main.cpp, `K` = src/ds4/ds4_kernels.cu(h).

## Bottom line
1. The hypothesis "2 of 3 groups idle each step" is TRUE structurally (one window flows g0 -> g1 -> g2 through flag-spinning
   hand-offs), but it is NOT proven to be the cause of N4 ~ N1. Four stronger suspects scale with row count and hit the same wall:
   (a) the single, in-order CPU-lane service thread (9534 experts of this pack live on the CPU lane, log `ds4: 9534 routed experts
   on the CPU lane (149.8 GB)`, VRAM holds only ~35-45% of slots per group); (b) Engram SSD reads x rows; (c) per-row host work
   after each step; (d) the T>1 MoE path (win_experts, distinct experts) instead of the T==1 token path.
2. Micro-batching is a big, risky change (ring of 64 mailbox slots, single hand-off buffers, in-order service thread, single
   `wplan`) and its gain is capped by memory-bound GPU stages whose time barely shrinks with fewer rows. Realistic gain 10-25%
   IF the GPU stages are the bottleneck, ~0 if the CPU lane is. Measure first (section 3).
3. Cheaper first: measure; then batch the post-step host work, and try N=4 drafting via a larger kWin (see 5).

## 1. How a decode step flows (N active sequences)
Scheduler (single engine thread, `C`:2234 `need.size() > 1` branch):
- per active conversation: room/think-budget (C:2246-2257); draft gate (C:2258-2285): drafts only if `m.can_draft_multi`
  (M:16004: `n*T <= kWin(16)` and `n*(T+1) <= kWin+2`; DSpark block T=5 (g.d_block) => n<=3). N=4 -> `nd>1 && !one_pass` -> every
  `tried` reset to 0 (C:2273-2276) -> `!any` -> plain `m.step_multi` (C:2357).
- `step_multi` (M:20654) -> `step_runs` (M:20680): builds rows, `wp.rseq/rpos` (one row per conversation), `seq_park`.
  Batch = ONE window of T=N rows (kWin=16 is a compile-time cap, K:21) in the single global `wplan`.
- Host prelude in step_runs (M:~20745-20762): `eg_issue` for each Engram layer (layers 1 and 14): cache lookup on the engine thread
  under `eg_cache_mu`, misses go to EgReader threads (M:9830-9920); memcpy of embedding rows to `h_emb`; `plan_window` (M:19310):
  per-layer indexer sizes and one mailbox seq per non-host-free layer (`wp.seq[il]=++seq_issued`, M:19337).
  Layers with nm>1 (tp2) are never host_free (M:19348 `l.nm == 1`), so EVERY layer uses the CPU-lane mailbox.
- `forward_par` (M:19568): `win_seq++`; one `Worker` thread per device (6 threads, M:977/4761), each runs `run_dev` (M:19603) and
  `forward_par` blocks on `w->wait()` (M:19577) = host enqueue complete, not GPU done.
- `run_dev` (M:19603-19670): each device thread enqueues ITS group's layers on its single compute stream `d.st`. First layer of a
  group (non-zero) starts with `xfer_get` (M:19624) = spin kernel on the previous group's host-mapped flag (`wait_at_least`,
  K:3070-3085, monotone `>= seq`). Last layer of a group, member 0: `xfer_put` of h|pre|topk|cand into `d.hand` (M:19657; under
  DS4_XFER_P2P the buffer is in the sender's VRAM and the receivers peer-read, M:2499-2590; flag stays host-mapped).
  Cross-group KV sources: `multi_get` / `xfer_get` on `sr.x_flag` (M:19636-19652, 20136-20160).
- `layer()` async path (M:20336-20500): shared expert + gate GEMV, `route_publish` kernel writes routing into `reqs[slot]`; the
  single service thread `serve_lane` (M:14150, started M:4713, strictly `next++` in seq order, `reqs[next % kRing]`) computes the
  CPU-lane experts (`cpu_experts`, M:14355-14373, st.cpu_ms) and answers into `resps`; the device waits on-GPU (`wait_seq`/
  `wa.wait_ptr`) only before the expert sum. T>1: `moe_win_group` + `win_experts` (M:20431-20490) read each distinct expert once.
  tp2: `allreduce` (P2P) after shared+routed (M:20493).
- Engram: `eg_finish(*eg_win[l.eg])` at M:20201 is a HOST wait inside the device's enqueue thread (layer 1 on g0, layer 14 on g1).
  It blocks that thread's enqueue of all later layers of its group until the SSD rows arrived. Layer 1 has only ~1 layer of cover.
- Head: last group runs `head_rows_dev` (M:19666); engine thread then `cudaStreamSynchronize(head dev)` (M:20779, st.head_ms).
- Post-step, all serial on the engine thread, GPUs idle: `cache_tick`, then per row `seq_point`+`seed_drafter` (M:20794-20797,
  several launches per row per DSpark block device, DRAFT_DEV=1,0,0), then in C:2358-2366 per row `penalties()` + `sample_multi`
  (M:~20960: sample kernel + `tok_back` = D2H + `cudaStreamSynchronize`, M:16260) => N host round-trips on the head device.
  Then loop again: Eg lookups, plan, worker wakeups. Next window cannot start until all four tokens are sampled.

## 2. Existing overlap / knobs
- Overlap inside a window only: all 6 device threads enqueue concurrently and groups pipeline via flags; Engram reads are issued
  before the window (M:19470-19480) and the `eg_decode_ahead` option exists (opt-in, "gave nothing" per notes). TP all-reduce
  P2P (DS4_AR_P2P). No overlap BETWEEN steps or between micro-batches: `forward()`/`step_runs` own one `wplan`, one `win_seq`,
  one hand-off buffer per device, and return only after the head sync.
- grep for PIPE/MICRO/UBATCH/OVERLAP in src/ds4: nothing relevant (only GLM's DFLASH_INJ_OVERLAP, docs/SETTINGS.md:168).
  Relevant knobs: DS4_SYNC (disable async routing, slower), DS4_ASYNC_T (M:1604, 16), DS4_HOST_FREE, DS4_AHEAD (route lookahead),
  DS4_ROUTE_FUSE(_WIN), DS4_HC_FUSE(_WIN), DS4_DECODE_TRIM, DS4_DRAFT_COST, MAYA_DS4_MULTI_DRAFT_POLICY (port branch only),
  DS4_COPY_IDLE (expert-cache copies in idle device time, M:4729), DS4_SERVICE_IDLE_MS, DS4_LANE_SPIN_US.
- PR #636 policy (MULTIDRAFT.md) confirms the gate; our notes (memory 01.11) say the policy port gave nothing: draft rounds cost
  ~2x on tp2 groups, and drafts through `draft_groups` (<=3 conversations a pass) did not pay.

## 3. Measuring per-step / per-group time cheaply (before any code)
Already there:
- `DS4_PROF=1` (M:2077-2150, C:241-250 `prof_line`): per-stage GPU time via events — BUT only `prof_window = prof_on && T == 1`
  (M:19487; step_runs sets `prof_window=false`, M:20760). So it covers N=1 plain steps only, not N=4. Cheap patch: set
  `prof_window = prof_on` in step_runs for non-spec windows; `ProfMark` is per-device so the per-group sum gives group busy time
  vs wall step time (idle fraction = 1 - busy_g / step). ~3 lines.
- Server STAT line (C:1259-1345, serve mode): `tok_s ms_tok vram_hit cpu= (CPU-lane experts a token) engram_ms= draft_ms verify_ms`.
  `vram_hit` and `cpu=` at --parallel 4 vs 1 immediately says how much CPU-lane work a token does; `engram_ms` is Engram wait.
- `Stats` (M hpp:117-118): `st.cpu_ms` (service thread busy time, M:14373), `gpu_wait_ms`, `layer_ms` (host enqueue time, includes
  Engram waits inside layers), `head_ms` (host wait for GPU = the real GPU time not hidden), `engram_ms`. The CLI prints them per
  token (C:3863 "decode per token: layers / routing waits / CPU lane / head wait / Engram reads"); step_runs accumulates the same
  fields (M:20766-20783), so one-line logging of their per-step deltas in step_runs gives the breakdown in the server.
- `maya-ds4 --pack ... --parallel 4 --ids FILE --multi-check` (C:2547-2990, needs GPUs): decodes N conversations alone then
  together with plain `step_multi`, prints `alone X tok/s, together Y tok/s (xK)`. This is exactly the plain 4-row step without
  server/drafter/sampling noise. If K is ~1.1-1.3, scaling is lost inside `step_runs` (CPU lane/Engram/MoE), not in the scheduler.
- nsys: `DS4_PROFILE` + `--capture-range=cudaProfilerApi` (C:3796) gives per-GPU timelines; look for the xfer_get spin kernels.
Cheap additions: wall-clock stamps around (a) eg_issue, (b) forward_par enqueue, (c) head sync, (d) seed_drafter loop, (e)
sample loop; plus `serve_lane` busy fraction = st.cpu_ms / wall.
Interpretation:
- Idle fraction of group g = 1 - (sum of its DS4_PROF stage times)/(step wall). Micro-batching can recover at most the
  upstream-wait part, NOT the CPU-lane wait (stage 3->4 `wait_seq`, prof marks 3/4) nor Engram.
- Known data: conc-35-par4-b.txt: N=1 74.5 agg (83/stream, with drafts), N=2 57.1 (!), N=4 73.4 (20.2/stream => ~49 ms per plain
  4-row round). N=2 below N=1 means per-round cost grows steeply with rows even when drafting; an N=1 round (window ~6 rows + draft)
  is ~30 ms, so a plain 4-row step (no draft!) at 49 ms is anomalously slow => per-row costs, not idle groups, dominate.

## 4. Micro-batching design (if measurements justify it)
Per-sequence state (already separate, good): KV window rings `q.win[k]`, compressors `q.c_kv/c_score`, CSA sources `q.ckv/ik`,
`SeqState.pos/toks/ctoks`, drafter rings, Engram ctoks. With nrow>0 attention picks them per row via `wp.rseq` (`attn_bound`/`AttnSeq`,
`multi_attn`, `multi_get` M:20136), so two micro-batches of DISJOINT conversations never touch the same KV. Rollback/backups
`l.ring_bak`/`c_snap`/`c_rows` are indexed by row r0 / run index (verify only), would need disjoint ranges per micro-batch.

Per-batch state that is SINGLE today and must be duplicated or serialised (each is a hazard):
1. `wplan` (M:1048, `plan_window`, `wp.win`, `wp.seq`, `cur_topk_n`): make `WinPlan wplans[2]`, `plan_window` per micro-batch
   (note `cur_topk_n/cur_cand_n` are carried across calls, M:1503; must be per-MB).
2. Mailbox ring `reqs/resps/layer_of`, kRing = 64 (K:344): one window already uses ~40 trunk layers (+ drafter blocks) of seqs;
   2 windows in flight = ~80 > 64: grow kRing to 128 (pinned arrays; cheap). And the service thread consumes `next` strictly in order
   (M:14150-14165), so seq numbers must be pre-assigned in the order requests are really published or it deadlocks / serialises:
   B's g0 layer 0 cannot be answered before A's g2 last layer if A's seqs are all lower. Needs an interleaved assignment following
   the pipeline schedule (A:g0 | B:g0, A:g1 | B:g1, A:g2 | B:g2, layer by layer within a time slot). With fixed per-group layer
   times it is feasible but fragile: one slow CPU answer stalls both. The one thread also makes the CPU lane a global serial resource.
3. Hand-off buffer per device `d.hand`/`d.hand_flag` (M:4145, XFER put writes the same VRAM/host buffer) and CSA `sr.x_stage/x_flag`:
   put of B overwrites A before g1's get has read it. Needs 2 slots (the chunk path already has `c.hand[2]`/`hand_ack`, M:732,5210:
   copy that protocol) or an ack wait on the sender before refilling. Flags are monotone (`wait_at_least >=`) so distinct seqs per MB work.
4. Per-device activation buffers (`d.h,h2,x,glog,ridx,rw,plan_ptr,grp_*,moe_out,sg,su,ed,xg,logits,...`): a device processes one MB at a
   time on one stream, so stream order protects them EXCEPT the buffers read by a later stage while the stream moves on (hand
   buffers, `h_emb`, `h_eg` Engram host rows indexed `[li*eg_cap + t]`, `h_taps`, head `d.logits` rows, `h_idx/h_w/h_x`).
   `h_emb`/`h_eg`/`h_taps` need a per-MB slot (offset rows by MB, or two copies). `d.logits`: head device holds rows for the next
   window while sampling the previous - sample reads `d.logits + row*V`; give each MB its own row range (rows are already per row).
5. Engram: `eg_win[li]` is single; issue both MBs' rows before the step (one EgBatch with all rows is fine) or two batch objects;
   `eg_finish` inside the layer blocks the device thread (M:20201) — with two MBs the second MB's thread time is lost anyway.
6. All-reduce state (`d.ar_seq`, flags, `d.xcnt`) is per group and used in stream order: safe. P2P AR buffers sized for kWin rows:
   fine (MB <= 8 rows). `hc_cnt` counters, `d.hc_done` events: per stream, safe.
7. Host threading: `forward_par` already hands each device its own thread; extend to `submit` two jobs per device (Worker holds one
   job slot, M:977-1010: make it a queue or run both MBs' enqueue inside one job: `run_dev(A); run_dev(B)`). Enqueue is async so B's
   g0 layers queue right after A's g0 layers on the same stream; A's g1 starts as soon as A's put lands: that IS the pipeline.
8. Step/scheduler: `step_runs` is blocking and owns sampling; split into `issue(MB)` / `finish(MB)`; head sync per MB; sampling of
   MB A (needs its own head stream sync) while B is still in g2 -> also overlaps host sampling with GPU. `seed_drafter` per MB.
9. CUDA graphs: none used, so no re-capture issue. PDL launches (pdl_launch) are used by xfer/AR kernels: keep launch order per
   stream = MB order to keep PDL dependencies right.
10. Risks: deadlock from spin kernels (xfer_get/wait_seq occupy SMs while waiting; with 2 MBs in the SAME group the second MB's spin
   kernels queue behind in stream order, fine; but any cross-stream wait cycle hangs the box = X crash class, see memory); bit-exactness
   (rows' bits already independent of window size, M:1604 comment, so splitting should stay bit-equal for greedy: verify with
   --multi-check); spec verify windows' backups; seeding order of drafter rings; expert cache swaps (`cache_tick`/idle_ev per window).

Estimated size: ~400-700 changed lines in M (plan/seq assignment ~100, hand/stage double-buffering ~120, host buffers ~60,
worker/forward_par ~80, step_runs split + sampling ~150, kRing/arrays ~30), + C scheduler ~60. Two weeks of careful work; GPU
hang risk during development.

Expected gain (honest): let f = time of one 4-row stage on a group. Pipelining 2 MBs of 2 rows over 3 groups costs 4m vs 3f serial
(m = 2-row stage time). Decode stages are weight-read bound: m ~ 0.65-0.8 f for GPU-resident work (distinct experts for 2 rows
~ 60-75% of 4 rows), so 4m/3f ~ 0.87-1.07: gain 0-15%. Where the CPU lane dominates, m scales ~0.5 f but the single in-order
service thread does all MBs' CPU work serially, so total CPU time per round is unchanged (no gain unless CPU work overlaps GPU work
of the other MB, which it can: +10-25% best case). 3 MBs: stage m ~0.5-0.6 f, 5m vs 3f = 0.83-1.0. Only a very GPU-bound,
row-proportional profile gives the theoretical x1.5-2. Do not expect 3x.
With drafting off at N=4 (no tokens/step multiplier), the achievable ceiling is maybe 83 -> 95-105 tok/s aggregate.

## 5. Cheaper alternatives (order by value/effort)
A. Measure first (section 3): `--multi-check` K factor, STAT `cpu=`/`vram_hit`, DS4_PROF for T=N. Decide between micro-batching and CPU lane/Engram work.
B. Batch the post-step host work (S, ~80 lines): one batched `sample_tokens` for N rows with a single D2H/sync instead of N
   `tok_back` syncs (M:16260, C:2358-2366); `seed_drafter` for all rows in one call (M:20794 loops rows; taps are contiguous
   rows `row0..`); penalties window on the host overlapped with the next window's Engram lookup. Gain: removes N-1 host round-trips
   + launch bursts with all 6 GPUs idle; probably 1-4 ms of ~49 (2-8%).
C. Overlap sampling/Engram of step t+1 with GPU: for greedy/sampled rows the next token is known only after sampling, so only
   Engram row ISSUE for token t+1 is blocked; `eg_issue` itself (cache lookup/hash under mutex) is host work ~ms: issue the lookups
   for the candidate row from a worker thread while the sample kernel runs, or pre-read the Engram n-gram rows for the top-1 token.
   Notes say Engram read-ahead "gave nothing" at N=1, but at N=4 rows x4 the miss traffic is 4x: re-test (`MAYA_DS4_EG_DECODE_AHEAD=1` only covers draft().
D. N=4 drafting (best lever, exists as the gate): (i) raise kWin 16 -> 24 (K:21; check kernels' per-row arrays/smem, `kWin*kPicks`
   ring/arrays, `MaRun`/`kMaRows`, `d_ints`, GEMV job limit "activations per job", AttnRows) so `can_draft_multi` passes for n=4
   (4*5=20 <= 24, 4*6=24 <= 26); effort moderate, global constant; (ii) or smaller per-conversation draft budget: a draft pass of
   2+2 conversations (PR #636's draft_groups) — already tested, drafter cost doubles, no gain. Net if the acceptance is ~2.2 tok/round
   (N=1 shows 83 tok/s/stream on ~30 ms rounds): potential +40-80% aggregate, provided one 4x(1+3) = 16-row verify window costs
   no more than ~1.3x a 4-row step (verify_ms in STAT tells). This is where N=1's 75 tok/s comes from and the most direct fix for N4~N1.
E. Re-balance DS4_SPLIT: stage times 10/15/15 layers but groups differ in VRAM-hit and CPU work (CUDA0/1 also hold DSpark blocks, drafter
   runs on them): DS4_PROF per group tells whether the middle/last group is the long pole; moving a layer is a 1-line env change
   (does not need code) and shortens every step regardless of batching.
F. Remove the Engram host stall in-layer (M:20201): make the Engram gather an on-device wait on a host-mapped flag (like wait_seq) so the
   device thread keeps enqueuing; only the stream waits. ~60 lines; matters if engram_ms per step is > 2 ms at N=4.
G. Overlap draft (g0) with verify (g1/g2) of OTHER conversations — a special case of micro-batching at conversation granularity;
   would need the same state duplication as section 4 (skip until A..D are measured).

## Recommendation
Do A now (one `--multi-check` at N=4 + STAT `cpu=`/`vram_hit`/`engram_ms`, + DS4_PROF extended to T=N). If K (together/alone) is ~1.2 and `cpu`
per token is >> N=1's, micro-batching will not help; go for D (draft at N=4 via kWin 24) and B. If DS4_PROF shows each group busy
<45% of the step wall with CPU-wait (marks 3->4) small, then micro-batching (section 4, 2 MBs of 2) is justified; expect +10-25%.
