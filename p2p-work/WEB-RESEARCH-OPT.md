# Web research: optimization ideas for Maya ds4, 6x3090 TP2 x3, N=4 decode

Date 2026-11-01. Sources: mw00/project-maya via `gh` (read-only), plus web search. The search tool is shallow: it returned abstracts and summaries, and I did not read the papers or the exllamav3 repo in depth. Gains marked "est." are my guesses, not measurements.
Evidence level: [M] = measured in a Maya PR/issue by someone else, [P] = paper claim, [G] = general guidance, no data.

## Ranked list (gain per effort)

### 1. Prefill chunk size (`--prefill-chunk`), pure tuning
- What: a 4x RTX 4090 user measured 8192 -> 5120 chunk on DS4.1: prefill +52% (8K prompt), +37% (16K), +19% (32K), +10% (64K). Decode is unchanged within noise. The cause is smaller prompt buffers and different resident-expert counts. Output bits change.
- Evidence [M]: https://github.com/mw00/project-maya/pull/650 (docs PR with all 93 records). Related: #747 (`MAYA_DS4_FLOW_DIRECT_NH=1`, +1..3.6% prefill, opt-in, RAM-tier prompt flow). #731 (slow-link ring rule makes prompt reads ~8% slower on 3090s on PCIe3 x8).
- Fit: not the decode bottleneck, but it is free for our 72K prefill (2280 tok/s). It gains less at long context.
- Est. gain: +5..15% prefill at 72K. Effort: minutes (sweep 4096/5120/6144/8192, 3+ repeats). Code exists: yes, a flag.
- Check: the Gen3 card may fall under #731's rule.

### 2. Placement and hit-rate knobs already in open PRs
- `DS4_RECENT_FIT=2` (#761): the per-byte queue gives some layers no "recent" cache slots. On 1x3090 it gave +4.6% decode [M]. Open PR, off by default. We have a 35-45% VRAM slot share, so the hit rate of 86-90% could go up a little.
- #501 (maintainer-confirmed, not fixed): the DSpark drafter's routes are never counted in the profile when all drafter experts sit in VRAM, so their placement is blind. We use DSpark, so check CUDA1 idle VRAM in the startup log. `MAYA_DS4_DRAFT_DEV` is the workaround, but it costs ~45% prompt speed (their measurement). `MAYA_DS4_DRAFT_VRAM` caps (#519) are untested on CUDA.
- #758 `DS4_CPU_SPLIT_COLS=1`: CPU-lane steps are cut by columns, bit-identical. +3.2% on a CPU-bound 1-GPU box [M]. At N=1 we are CPU-thread sensitive (16 threads -> -33%), so it may help N=1, and N=4 probably not (lane wait ~0).
- Est. gain: +2..5% each. Effort: low (env vars). Code exists: yes. Link https://github.com/mw00/project-maya/pull/761, /pull/758.

### 3. 1-row wavefront pipelining of conversations across the 3 groups (new twist on the micro-batch dead end)
- What: GLM in the same engine has `STRATA_GLM_SEQS`: "a token of each conversation goes through the split's GPUs one behind the other". 4x4090 went from 110 tok/s alone to 225 for four conversations [M] (quoted in issue #317, a request for the same on DS4: https://github.com/mw00/project-maya/issues/317). It is the PP-with-batch>depth pattern (vLLM PP serving, [AMD playbook](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)).
- Why it differs from what we tried: we measured 2-row micro-batches costing ~4 rows. If a 1-row group pass costs about N=1/3 (~4.4 ms/group at 75 tok/s), then 4 single-row tokens staggered across 3 groups could overlap, whereas 2x2 rows pays the weight read twice for nothing. The experts of different streams are different weights, so the weight-read cost scales with the union of experts. Expect much less than 3x, because the groups compete for the CPU lane and the host thread.
- Open question (measure first, no GPU run needed to ask): the per-group time of a 1-row pass vs a 4-row pass at N=4. Our "~1 ms per group per layer" at N=4 says what the 4-row pass costs. If the 1-row pass is ~0.8 ms, the idea is dead.
- Est. gain: 0 to +60%, high variance. Effort: high (stream/graph scheduling in ds4, hop copies, per-stream state). Code to port: the GLM SEQS path in glm_fast_path.cu (not ds4).

### 4. CUDA Graphs for the ds4 decode layer loop
- What: capture per-layer or per-group decode into graphs with device-side routing (fixed grids, device-resident counts). Decode with dynamic routing is graph-safe if index tables are built on the GPU into max-sized buffers: see the Megatron-Core vLLM-fused-MoE docs (https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.inference.moe.vllm_fused_moe.html) and MonoMoE (https://arxiv.org/pdf/2609.04244, fused mega-kernel for quantized MoE decode, run under vLLM's CUDA-Graph decode path).
- Hybrid CPU/GPU precedent: KTransformers captures the hybrid CPU/GPU execution as graphs and says launch overhead falls from >20% to ~0 ([LMSYS blog](https://lmsys.org/blog/2025-10-22-KTransformers), [SOSP25 paper](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/)) [P].
- Maya precedent [M]: #59 (merged) opt-in KDA-layer graphs for GLM single GPU, +2.35% on a 4090D at ~31 tok/s; `CapturedGraph` class with key invalidation on CPU-plan or buffer change. #234 had `glm_layer_graph_test` (22 captures / 74 replays / 20 invalidations).
- Do not expect PDL: #495 (Basalt's "programmatic dependent launch") is gated to cc >= 9.0 in ds4, so it does nothing on sm_86. Graphs are the Ampere-side alternative.
- Why it fits: ~1 ms per layer per group is only a few dozen kernels. A launch gap of 3-5 us each, plus the all-reduce/flag-wait kernels, could be 10-20% of the 40 ms window. Caveat: the maintainers' measured gain from graphs on GLM was only 2.35%, so launch overhead may be smaller than we think. Measure the gap first with nsys (kernel timeline gaps, no new code).
- Est. gain: +3..10% decode N=4, maybe more at N=1. Effort: high (wait flags and mailbox kernels must be capturable, hop copies, TP all-reduce P2P). Code exists: `CapturedGraph` in the GLM path.

### 5. KTransformers-style expert deferral for ds4
- What: defer CPU-lane expert results to after the next layer's mixer, so CPU expert work overlaps GPU attention. KTransformers reports up to 1.45x on top of its other optimizations, CPU utilization to ~100%, accuracy change < 0.5% [P]. Source: SOSP25 paper above, [SGLang issue](https://github.com/sgl-project/sglang/issues/11425).
- Maya has it for GLM only: `STRATA_GLM_DEFER_CPU` (#252); #722 (open, `MAYA_GLM_DEFER_CPU=2`) +7.7% decode on 2x4060Ti, +13% with a re-biased router [M]. It changes the answers. https://github.com/mw00/project-maya/pull/722
- Fit: our CPU-lane wait is ~0 at N=4, so little to gain there. At N=1 (75 tok/s, thread-count sensitive, 10-14% misses) it is the right technique.
- Est. gain: N=1 +5..10%, N=4 ~0. Effort: medium. Code exists: GLM path, to be ported to ds4 layers. Relative value is low for the 4-agent target.

## Lower-ranked / informational

### Small-batch MoE kernels with shared weight decode
- Maya #164 (merged): EXL3 verify windows decode each weight once for all window tokens. It was modelled on exllamav3's multi-row kernels (CPU expert kernel up to 4 rows; small-m GEMV and fused MoE share a tile's decode among an MMA's rows). Check that the ds4 N=4 decode path uses a `block_gemv_n`-style kernel for the 4 rows of the shared-expert and attention projections, and that routed experts hit by several rows are decoded once. Our "2-row costs as 4 rows" result suggests the weight-read dominates, as expected, but also that a 4-row kernel is roughly free up to 4 rows: confirm that 4 rows is not already the cheapest point, and test 8 rows (N=8 `--parallel 8` as the 4090 box runs).
- Megakernels: Mirage MPK (https://arxiv.org/html/2512.22219v1, OSDI26): 1.0-1.7x vs SGLang/vLLM, mostly on small dense models and H100. [AutoMegaKernel](https://huggingface.co/papers/2606.09682.md) reports 1.33x at batch 1 on an L4. No 3090 numbers, and our model is a 748B EXL3 trellis MoE with a host lane. Not portable at reasonable effort.
- Basalt (https://github.com/jesdga95/basalt, another Strata fork) ideas were merged in #495: CPU lane in one batch per NUMA node, self-answered CPU requests. Already in v1.1.45/46 if you run them. `DS4_CPU_FLOW=0` switches it off.

### Hybrid CPU/GPU MoE literature
- HybriMoE (https://huggingface.co/papers/2504.05897): dynamic intra-layer CPU/GPU split, impact-driven inter-layer prefetch, score-based cache. 1.33x prefill, 1.70x decode vs a kTransformers baseline [P]. Fiddler (https://arxiv.org/html/2402.07033v3): the CPU computes for missing experts when the weight copy is dearer. Maya's CPU lane already does this. The new part is the score-based cache and prefetch. Our profile + #761 `RECENT_FIT` is the equivalent.
- #644: with the default expert cache, CPU-lane experts make output non-repeatable. `--no-expert-cache` gives bit-identical output. This matters for A/B benchmarking: compare configurations with a frozen profile.

### TP+PP vs EP on PCIe 3090s
- General guidance [G] ([premai](https://blog.premai.io/multi-gpu-llm-inference-tp-vs-pp-vs-ep-parallelism-guide-2026), AMD playbook above): PP suits PCIe, needs concurrency; TP+EP for low concurrency; DP+EP for high concurrency. No 3090 data found. vLLM's own DeepSeek write-up shows MoE dispatch/combine dominating decode, which does not apply here because the experts are host-pinned. The 10x3090 TP2 report (#494, 97 tok/s, Xeon AVX-512, 200 W cap, x4 links) suggests more PP stages do not slow decode much. For 6 cards, the alternative is TP1 x6 (layer split) at N=4, which Maya falls back to for some packs, but that loses P2P all-reduce prefill gains.

### Maya items to know about (all open unless stated)
- #317 request: layer-split idle gaps (GPU util ~50% each): exactly our serial pipeline. No maintainer answer.
- #701 `MAYA_DS4_HOST_FREE_TP=1`: skips the mailbox round trip for TP layers with all experts in VRAM (+2..5% on 8x3090 with everything resident). Not applicable to our 86-90% hit rate.
- #699: sampled decode (T>0) 3.5x slower in v1.1.37+ on a Windows PRO 6000 (not reproduced on Linux 4x4090, #699 comments). If agents sample with temperature, benchmark at T>0 too.
- #630: GLM route kernel (3 barriers instead of 9), GLM only. DeepSeek route is a possible port target, expected gain tiny (-0.35% decode).
- #556: 4x3090 + DSpark leaves 4-6 GB VRAM empty on one GPU: same as #501.
- #650/#647-style docs show benchmarks conducted with `tools/agent_bench.py` (#406): use it for N=4 A/B.

## System-level knobs (no data specific to this setup found; treat as hypotheses, A/B each with 3+ repeats)
- NVIDIA: `nvidia-smi -pm 1` (persistence), power limit at 350 W (some boxes in the wild run 200 W, #494; make sure ours does not throttle: Maya's gpulog already records clocks). Locked clocks (`-lgc`) reduce clock-ramp variance during bursty decode. Est. 0..5%, effort minutes. Source: general; no benchmark found.
- CPU: governor `performance`, disable deep C-states (the AMD EPYC 7003 tuning guide: https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/tuning-guides/high-performance-computing-tuning-guide-amd-epyc7003-series-processors.pdf). The CPU lane spins for 20 ms, so C-states matter mostly between rounds. Est. 0..3%.
- THP `always` (the page-pinned expert RAM is probably already hugepage-backed via cudaHostRegister; unverified) [G].
- NPS1 is right for a single process with no NUMA pinning (AMD guide); L3-as-NUMA already measured worse.
- DDR4: Milan 3200 MT/s with FCLK 1:1 is the stock maximum; no reliable data on gain beyond it. IRQ affinity, PCIe MaxReadReq, relaxed ordering: nothing found. P2P/ReBAR work is done.

## What I could not find
- No exllamav3 issue/commit data on trellis GEMV at batch>1 beyond what #164's description quotes. No Marlin-MoE-on-3090 small-batch numbers. No 3090 PP-vs-TP MoE benchmark. A deeper pass on the turboderp/exllamav3 repo (mgemm, fused MoE) would be the next step.
