# Forks and PRs of mw00/project-maya: what could speed up our ds4 setup (review of 2026-11-01)

Research only (gh read-only). Upstream is already at **v1.1.46** (v1.1.45 + #728 and tool-call fixes). Our target: 6x3090, --tp 2 x3 groups, --parallel 4, DSpark on pair 0, EXL3 3.5bpw (and GGUF Maya-S).

## Method and key finding
- 100 open PRs read by title/files; ~15 ds4 PRs read in full (bodies, measured numbers).
- ~120 closed-unmerged PRs since 10.10 checked: **almost all were in fact squash-merged by the maintainer** ("closed rather than merged", shipped in v1.1.28..v1.1.42: #511-#519 series, #464 DSPARK_GROUP, #527 helper EXL3, #622 lazy drafter, #647 draft timing, #669 COPY_FAIR, #587 event scheduler, #691 EXL3 lend, #524 Engram read-ahead, #596 EXL3 AVX512-VNNI tier, #561/#562 lane...). Really rejected/superseded: #558 (-> #636/#641), #600 (whole-chunk skip, no gain claim), #627/#598/#597/#629 (HIP, 0.2-0.9%).
- Forks: 323 listed, 138 with a push after creation in the last 2 weeks, compared with `mw00:main...owner:branch`. Only ~25 forks are ahead of main at all; those touching src/ds4 are PR branches of the authors listed below. No fork carries unseen ds4 speed work for NVIDIA TP/EXL3/AVX2 other than the items below. (Non-default branches of 20 active authors scanned too.)

## Ranking for our setup

| # | Candidate | Link | Expected gain | Effort | Risk |
|---|---|---|---|---|---|
| 1 | **#636 measured-cost draft policy** (MAYA_DS4_MULTI_DRAFT_POLICY, default on in PR) | https://github.com/mw00/project-maya/pull/636 | Directly targets our N=4 problem: main's gate "draft only where every drafter fits one pass"; with 4 requests and DSpark's 5-row block it **never drafts** -> N=4 aggregate ~ N=1. PR: copy-style text N=4 108.8 -> 118.6 (+9%), prose N=4 -4%/noise, layer split 2 GPUs. TP2 not measured by the author. | Medium: +1621/-167, depends on #587 (merged), must be rebased on v1.1.46 and onto our P2P patches (touches verify_multi/draft_groups) | Medium: default-on policy, greedy exact per PR; our drafter sits on pair 0 over 3 groups, so pricing may differ |
| 2 | **DS4_ENGRAM_THREADS=32..64** (issue, no code) | https://github.com/mw00/project-maya/issues/716 | Linux default is min(16,max(4,hw/2)) (=16 for us); reporter: cold prompts +21% with 32 threads, decode unchanged (SSD queue depth). Fits our SSD Engram (203 GB, page cache cold). Might also cut the 2.4 ms layer-1 wait if the decode reader shares the pool (unknown). | Env var only | Very low |
| 3 | **DS4_ATTN_SPLIT (+_TC) split-KV verify attention** (fork branch, no PR) | https://github.com/SeriousRising/project-maya/tree/ds4-split-kv-attn | For windows of 1-16 rows (our DSpark windows, 4 convs x 4-5 rows): verify window -4% (split) / -8% (tensor-core phase 1) on 2x CMP170HX; sm_80+ ok on 3090. Helps the batched N=4 round. Bounded not bit-exact. | Low-medium: +600 lines in ds4_kernels.cu, 190 behind main, rebase | Low-medium (answers change slightly, FP4 KV, heads%16 for TC) |
| 4 | **#712 (rebase of #624) Q5_K int8 tensor-core GEMVs, DS4_Q5K_IMMA=5** | https://github.com/mw00/project-maya/pull/712 | GGUF Maya-S only (not EXL3): the Q5_K dense layers per verify-window row; 4090D 32K+2K: 60.2 -> 70.4 tok/s (+16.9%), bit-exact vs dp4a, prefill unchanged. Our N=4 windows have >=5 rows so threshold 5 applies. | Low: +316 lines, based on v1.1.42, opt-in | Low (bit-exact, tested) |
| 5 | **#641 two verify windows in flight (MAYA_DS4_MD_PIPE)** | https://github.com/mw00/project-maya/pull/641 | Best-measured batched-decode result: 2-GPU layer split N=4 essay 145 -> 174 tok/s (+20%), copy +40%. **Not supported for TP groups** (pipe_ok refuses), needs #636. Value for us is the idea: overlap two windows across stages = our "groups wait on each other" problem. | High: +3871 lines, a port to our 3-pipeline-group TP2 layout | High |
| 6 | **#682 accept STRATA_ aliases of READ_ORDER/PIECE/ROUNDS/SHARE** (check!) | https://github.com/mw00/project-maya/pull/682 | Not a speed-up but a trap: through `serve` MAYA_DS4_READ_* get converted to STRATA_* and the engine ignored them (reporter: prefill 621 vs 1053 t/s with 2x32K). If we launch via serve, our READ_SHARE=90 may not be applied. 10-line patch. | Trivial | None |
| 7 | **#569 DS4_PF_SAMPLES=4** (skip expert read-ahead hints when pages are resident) | https://github.com/mw00/project-maya/pull/569 | +14-18% time on agent turns (128-512 added tokens after 8K cached prefix) on GGUF file-backed experts. Doc says it does not touch Engram and RAM copies bypass it, so probably ~0 for our pinned experts. | Low, opt-in | Low |
| 8 | **#732 save every parallel conversation at QUIT** | https://github.com/mw00/project-maya/pull/732 | Restart no longer throws away the 3 other agent conversations (9K tokens: 8.06 s -> 0.84 s; for 72K+ contexts minutes). Quality-of-life for our test cycles. +8 lines. | Trivial | Very low |

Also noted:
- #684 report drafts by source (DSpark vs prompt lookup) in log and /metrics: useful measuring tool for any drafter tuning (we need to know who pays at N=4). +355. Low risk.
- #747 FLOW_DIRECT_NH: +0.9-3.6% prefill, RAM-tier MXFP4 GGUF experts only. Marginal.
- #736 FLOW_AR_DMA: needs groups of >=3 members; 2-member groups keep the old path -> not applicable (we have our own P2P AR).
- #701 HOST_FREE_TP: +2-5% decode, only when every expert of a TP group's layer is in VRAM. We hold 27% -> not applicable (unless some layers are fully VRAM-resident).
- #741 LANE_PARITY: bit-identical lane/VRAM, costs speed -> no.
- #695 IQ1_S CPU lane AVX2 vector decode (Zen 3 measured, bit-exact) and #623/#584: IQ1_S/IQ1_M GGUF rows only; our EXL3 lane is unaffected. Maya-S MXFP4 not affected.
- #557 (still listed open) is already in main in effect via #587 (the v1.1.39 event scheduler, READ_SHARE). #144 is GLM only. #745 router BF16 is AMD only (NVIDIA off by default; +3.5% on Strix Halo; could be tried with MAYA_DS4_ROUTER_BF16=1 on NVIDIA but no data).
- #641/#636 authors (somnergy) hardware: 2x72 GB Blackwell, so TP2 numbers are missing; whoever tests them first on TP2 gets the data.

## Engram (SSD layer-1 wait, row cache hit 37%)
- Nothing in forks/PRs prefetches decode Engram rows from DSpark's draft tokens or pins only hot layer-1 rows. Existing upstream pieces: DS4_ENGRAM_RAM=1 (locks both whole tables, impossible at 203 GB + 150 GB experts in 252 GB), DS4_ENGRAM_PREFETCH (#524, **one GPU only**, prompt only), DS4_ENGRAM_CACHE_MB (row cache size), MAYA_DS4_EG_RANDOM=1 (merged from the 0c33 gfx906 fork: readahead off for Engram files; Linux; can stop Engram reads evicting CPU-lane experts), DS4_ENGRAM_THREADS (#716). Cheap knobs worth checking in our env: EG_RANDOM, ENGRAM_CACHE_MB larger, ENGRAM_THREADS 32.
- Idea with no upstream implementation: layer-1 rows depend only on token ids; with DSpark the next tokens are drafted before the verify, so issue the layer-1 reads early (read-ahead on the draft tokens' n-grams). Own work.

## CPU lane on AVX2
- No fork improves the EXL3 CPU rows on AVX2 (#596 added an AVX-512 VNNI tier only, #452 IQ1_M AVX2 merged earlier, #561 small-call lane speed merged in v1.1.36). Zen 3 has no AVX-VNNI, so nothing to port directly.

## Pipeline balancing between groups
- Only #641 (two windows in flight; 2-GPU layer split) and DS4_TP_SHARES (#591 merged, uneven shares inside a TP group) exist. needmorevram branch `ds4-prompt-speed` (https://github.com/needmorevram/project-maya/tree/ds4-prompt-speed, 1 commit unreleased: lending layer keeps one expert in RAM so it has slots to lend; 118K prompt 2307 tok/s on 2 TP4 groups with split 3,10,27) is only useful for layouts where a group fits all experts of a layer; not ours at 27%.

## Fork/branch scan, ds4 commits ahead of main (recent)
- somnergy: pr/ds4-md-pipe, pr/ds4-multi-draft (= #641, #636).
- SeriousRising: ds4-split-kv-attn (no PR; item 3), ds4-zen2-tuning (IQ1_S pdep, merged).
- handmade0octopus: v1.1.46-prtest (7 ahead, 0 behind; #682, #684, #712 variants), pr/ds4-q5k-imma-v1142.
- LostVector: HIP/AMD P2P experiments (peer logits gather, helper-input vector reads) - AMD/HIP, <1% gains, skip. experiment/ds4-global-draft-gpus (DSpark block placement across groups, 2 commits, merged variant is DSPARK_GROUP/DRAFT_DEV, which we already use).
- Non-ds4 or irrelevant: jefflaichunyin (ppc64le), xcpatr9231975 (DGX Spark/aarch64), 0c33 (MI50), JadDid911 branch `ctc` (a "cheap path / lossy skip" CTC experiment, 12 commits, 5 ds4 files: unknown research; lossy, not examined in depth), neron82 (two-GPU expert helper for 1.1.4x, superseded by merged #514/#527), tntcannon5000 (benchmark docs).
