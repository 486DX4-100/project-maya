# MAYA_DS4_MICROBATCH=2 (branch p2p-145-mb, tree /home/michal/project-maya-p2p145mb)
Stage 1 = full 3-group pipeline of 2 micro-batches (rows split (N+1)/2 | rest), plain non-spec steps, n>=4 only, default off.
Design: step_runs plans the whole window (plan_noseq), forward_mb slices it into 2 sub-plans (own win seq, r0 row offset into
h_emb/h_taps/h_eg/logits, hand-off slot 0/1: Dev::hand1/hand_flag1, Src::x_stage_mb/x_flag_mb, DS4_XFER_P2P placement kept).
Each device thread enqueues A's layers then B's on its stream => g(k+1) runs A while g(k) runs B. Mailbox seqs are numbered in
pipeline publish order (A:g0 | B:g0+A:g1 interleaved | B:g1+A:g2 | B:g2), a topological order of the dependency graph => no
deadlock; kRing 64->128 (~86 requests in flight; +~70 MB pinned). No new spin kernels; head per micro-batch with logits row offset.
Results (DS4_XFER_P2P=1, reserve 1300, 1 run each, noise 5-8%):
 off (mb0): 72K 1560 (cold outlier), N1 72.2, N4 85.5 | on (mb2): 72K 2210, N1 78.6, N4 85.4 | on (mb1, short): N1 75.3, N4 82.4
DS4_PROF N=4 window: off busy g0/g1/g2 ~10-12/12.5/15 ms, wall 38-40 ms; on busy 17-19/21-22/22 ms (sum ~62 vs wall 40-43 => groups overlap
~1.5x) but wall unchanged: a 2-row micro-batch costs ~1.6-1.7x less than... i.e. nearly as much as the 4-row window (weights/launch bound),
so 4m ~ 3f as the analysis predicted. No gain => stopped after stage 1.
Correctness: needle 5/6 both (same miss); cc4 (4 prompts, T=0, 128 tok) on vs off: 5 of 8 outputs identical-or-diverging late (chars 152-455),
one diverges at char 22 (mb1); model not bit-reproducible run to run.
Risks: kRing/RAM, N<4 unsupported, sampling not overlapped with B, eg_finish still host-blocks the device thread.
