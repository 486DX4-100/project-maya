# Intel Arc: the Maya-XS DeepSeek GGUFs on llama.cpp's SYCL backend

**This is outside Maya.** Maya's engine has no Intel backend ([#37](https://github.com/mw00/project-maya/issues/37)),
and we have not tested anything on this page: it is what @Jumbomuffin777 reported in
[#401](https://github.com/mw00/project-maya/issues/401), written down so others with an Intel Arc card can find it.
The setup, the dashboard, DSpark drafting and the other Maya features do not run on this path.

## What was reported

The Maya-XS DeepSeek-V4.1 GGUFs run on Intel Arc through llama.cpp's SYCL backend and the open `deepseek41` pull request
[ggml-org/llama.cpp#30817](https://github.com/ggml-org/llama.cpp/pull/30817), after a small patch to that pull
request's model loader:

- Maya's GGUFs store the Engram metadata as `deepseek41.engram.*` with u32 offsets and primes, one Engram table per
  layer (`blk.1/14.engram_embd`, with local offsets) and no `hash_layer_count`.
- That pull request's loader expects `ple.*` keys and one global `per_layer_token_embd`.
- The patch accepts both layouts; files from the pull request's own converter load as before.

The reporter's branch with the patch and a longer write-up (build steps, flags, known limits) is here:
[Jumbomuffin777:docs/intel-arc-llamacpp-sycl](https://github.com/mw00/project-maya/compare/main...Jumbomuffin777:project-maya:docs/intel-arc-llamacpp-sycl).
We have not reviewed or built it; read the patch before you run it.

## What was measured

Their PC: Intel Arc Pro B70 32 GB + B65 32 GB + B60 24 GB, Ryzen 9 9950X, 64 GB DDR5, Linux. Model: Maya-XS IQ1_S,
64K context.

| Run | Prefill | Decode |
| --- | --- | --- |
| 1,934-token prompt | 45.6 tok/s | 10.9 tok/s |
| 2,470 tokens through an OpenAI-compatible gateway | 41.0 tok/s | 10.8 tok/s |

Tool calls, streaming and an 11K-token recall check passed.

## Limits

- Text only.
- No DSpark drafting and no Engram streaming: the model is placed by llama.cpp's own rules.
- It depends on a llama.cpp pull request that was still open when it was reported, plus the loader patch above.
- The numbers are one person's, on three Arc cards, with the one quant above.

The patch, the measurements and the write-up were prepared by the reporter with an AI coding agent on their own hardware;
they reviewed them. Questions about this setup belong in [#401](https://github.com/mw00/project-maya/issues/401).
