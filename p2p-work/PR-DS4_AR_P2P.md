# DeepSeek on 2+ GPUs --tp: up to 29% faster prompt reading (prefill) via peer-to-peer all-reduce over ReBAR (opt-in, DS4_AR_P2P=1)

I run DeepSeek V4.1 on six RTX 3090 with `--tp 2`. I noticed that on every layer the two cards of a group exchange their partial sums through host RAM. My cards have ReBAR, so they can read each other's memory directly, and I made the all-reduce do exactly that.

With `DS4_AR_P2P=1` the all-reduce buffers go to the first GPU of the group and the second card reads and writes them peer-to-peer over PCIe. Without the variable nothing changes.

## My results

6x RTX 3090 (ReBAR), EPYC 7R13, `--tp 2 --parallel 4`, chunk 4096, real text. Same binary, flag on and off, runs alternated:

| | without | with |
|---|---|---|
| Maya-S, 12K prompt | 881 tok/s | 1138 (+29%) |
| Maya-S, 72K prompt | 1204 tok/s | 1482 (+23%) |
| EXL3 3.5bpw, 72K prompt | 1208 tok/s | 1906 (+58%) |
| all-reduce alone (`ds4_gguf_test --ar`) | 4.0 GB/s | 9.3 GB/s |

Decode is more or less the same (N=1 81 -> 85 tok/s, N=4 in the noise). I also tried `--tp 3` and there was no gain.

## Good to know

- It works only on Linux, with ReBAR (BAR1 >= 1 GiB on every card) and groups of 2-4 GPUs. In any other case the old path is used.
- Peer access has to be turned on at the very end of `load()`. When I did it earlier, `route_kernel` crashed.
- The buffers need about 1 GB on the first GPU of the group. With EXL3 3.5bpw the experts filled the whole VRAM and the log said `left host-mapped (no room)`, so the gain was small. With `--reserve-mib 1300` everything fit.
- The sums are bit-identical (`ds4_gguf_test --ar`). The full model output is not bit-reproducible between runs even without the flag; with the flag the differences are in the same range, and my needle-in-30K test gave the same answers.
- I tested only on NVIDIA (sm_86). `tools/check_hip_compat.py` passes. I could not try AMD, V100 or Windows.
