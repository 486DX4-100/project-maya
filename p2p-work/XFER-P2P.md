# Group hand-offs (v1.1.45 + a0fcd0d) - how activations move between pipeline groups

Prompt chunks (run_chunks, ds4_model.cu ~17550/17604): the group's member 0 (mi == 0) runs xfer_put(chunk_xfer(c.hand[slot]...)),
a kernel on its compute stream (d.st) that stores h | pre | topk | cand (4096 rows: h = rows*hc*dim*4 ~ 335 MB, ~0.35 GB with the rest)
into a pinned, host-mapped buffer (cudaHostAlloc Mapped, c.hand[2 slots], chunk_hand_bytes()), then a system fence and the host-mapped
flag c.hand_flag[slot] = seq. EVERY member of the next group runs xfer_get on its own stream: a kernel that spins on the flag, then reads
the host buffer with uncached loads (ld_fresh) into its own c.h/pre/topk/cand. Not cudaMemcpy, no events: kernels + flags (PDL launches).
Reuse protocol: receivers set c.hand_ack flags (host), the sender waits (wait_flags) before refilling a slot (two deep).
Decode/windows (run_dev, hand_xfer ~19442/19479/19512): the same xfer_put/xfer_get with d.hand (bounce_bytes()) and d.hand_flag; one
host-mapped region per device, only mi == 0 sends. (DS4_DECODE_TRIM 8 / dhand is the drafter's own variant of the same.)
So both go through the same host-staged path, and both put+get kernels sit ON the compute streams (serialised with the layers).

What "between layers" in DS4_PROF is: prof_mark(d, 0) at the start of layer_chunk, prof_mark(d, 3) at the end of its experts; the stage-0 time
is the GPU-stream gap (event-to-event, per GPU, summed over 6 GPUs) between a layer's end and the next layer's start. It contains
hc_post/copies at layer end, xfer_put / xfer_get (incl. the get kernel SPINNING on the flag = waiting for the upstream group), set_flag,
wait_flags(hand_ack), Engram/source-row gets, and kicks. 28.3 s / 6 GPUs = 4.7 s a GPU out of ~50 s of prompt: most of it is expected to be
pipeline waiting (the middle group has 16 layers vs 11 and 13, so the outer groups idle at the get), not copy bandwidth; the copy's own cost
(335 MB written + read through host mapping, 64-block kernels, both members of the next group reading at once) is on the order of 0.1 s a
chunk a hand-off, 18 chunks x 2 hand-offs, a few seconds on the streams at most. Not measured here (no GPU runs).
Reducing it: DS4_XFER_P2P=1 (this change: local put, peer-read get, no host round trip); rebalance the layer split so the middle group has fewer
layers; the wait part is not copy time and only shrinks with a better balance or deeper pipelining.
