"""tools/vision/ggml_hip_precision.py - float32 accumulation for the vision encoder's two kinds of sums on AMD GPUs,
in the pinned llama.cpp's ggml-cuda (built as ggml's HIP backend only by maya-vision).

The encoder (24 ViT blocks, then the merger) amplifies small rounding errors on a few high-norm patches; on an AMD
GPU two of ggml's choices round far more than CPU or NVIDIA (V100), and those patches' rows came out wrong
(cosine 0.59 against a float32 reference on Strix Halo, gfx1151, where the CPU is at 0.99):
  - the F16 weight matmuls go to hipBLAS with an F16 compute type and an F16 result.  ggml takes the F32 result for
    Volta, RDNA4 and CDNA already; every AMD GPU gets it here (F16 inputs, F32 sums and result);
  - flash attention with this shape (no mask, as many K/V heads as Q heads) runs ggml's tile kernel on AMD, which on
    GPUs with fast FP16 sums softmax(KQ)*V in half precision - the graph's GGML_PREC_F32 is not looked at.  On HIP
    the kernel keeps those sums (VKQ) in float here; its K/V tiles and KQ values stay FP16 as before.
Both are compiled only in a HIP build (GGML_USE_HIP / an AMD device): a CUDA build is the same code as before.
Every edit is anchored on the pinned source and idempotent, as glm5next_patch.py's (which runs this one).
    python ggml_hip_precision.py <llama.cpp dir>
"""
from __future__ import annotations

import pathlib
import sys

MARK = "GGML_FATTN_TILE_VKQ_F32"


def edit(path: pathlib.Path, pairs: list[tuple[str, str, int]]):
    s = path.read_text(encoding="utf-8")
    for old, new, count in pairs:
        if new in s:
            continue                                   # already applied
        n = s.count(old)
        if n != count:
            raise SystemExit(f"ggml HIP precision patch: {path}: anchor found {n} times (expected {count}):\n{old}")
        s = s.replace(old, new)
    path.write_text(s, encoding="utf-8", newline="\n")


FP16 = "#ifdef FAST_FP16_AVAILABLE\n"
FP16_HALF_VKQ = "#if defined(FAST_FP16_AVAILABLE) && !defined(GGML_FATTN_TILE_VKQ_F32)\n"


def main(root: pathlib.Path):
    cu = root / "ggml" / "src" / "ggml-cuda"
    edit(cu / "ggml-cuda.cu", [(
        "        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);\n",
        "        // (Maya, tools/vision/ggml_hip_precision.py: every AMD GPU - hipBLAS's F16 compute type rounds the\n"
        "        // vision encoder's matmuls too coarsely)\n"
        "        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_AMD(cc);\n", 1)])
    edit(cu / "fattn-tile.cuh", [
        ('#include "common.cuh"\n#include "fattn-common.cuh"\n',
         '#include "common.cuh"\n#include "fattn-common.cuh"\n\n'
         "// (Maya, tools/vision/ggml_hip_precision.py) on AMD GPUs the softmax(KQ)*V sums (VKQ) are kept in float: in\n"
         "// half precision they lose the vision encoder's attention (GGML_PREC_F32 is not looked at by this kernel)\n"
         "#if defined(GGML_USE_HIP) && defined(FAST_FP16_AVAILABLE)\n"
         "#define GGML_FATTN_TILE_VKQ_F32\n"
         "#endif\n", 1),
        # the VKQ rescales (softmax max update, attention sinks): the float path's
        (FP16 + "            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);\n",
         FP16_HALF_VKQ + "            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);\n", 2),
        # VKQ += V*KQ: FP16 V and KQ, float sums
        ("                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size] += V_k[i0/warp_size]*KQ_k[jc_VKQ_0];\n",
         "#ifdef GGML_FATTN_TILE_VKQ_F32\n"
         "                    const float2 v_f = __half22float2(V_k[i0/warp_size]);\n"
         "                    const float  p_f = __low2float(KQ_k[jc_VKQ_0]);\n"
         "                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].x += v_f.x*p_f;\n"
         "                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].y += v_f.y*p_f;\n"
         "#else\n"
         "                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size] += V_k[i0/warp_size]*KQ_k[jc_VKQ_0];\n"
         "#endif // GGML_FATTN_TILE_VKQ_F32\n", 1),
        # the accumulators themselves
        ("    __align__(16) half2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};\n",
         "#ifdef GGML_FATTN_TILE_VKQ_F32\n"
         "    __align__(16) float2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};\n"
         "#else\n"
         "    __align__(16) half2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};\n"
         "#endif // GGML_FATTN_TILE_VKQ_F32\n", 1),
        # combining the parallel warps' VKQ (in KV_tmp, as floats) and writing the result: the float path's
        ("        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, \"KV_tmp too small\");\n",
         "        static_assert(nbatch_fa*nbatch_K >= nwarps*DVp, \"KV_tmp too small\");\n"
         "#ifdef GGML_FATTN_TILE_VKQ_F32\n"
         "        static_assert(nbatch_fa*(nbatch_K/2 + cpy_ne) + DVp-DV >= nwarps*DVp, \"KV_tmp too small for float VKQ\");\n"
         "#endif // GGML_FATTN_TILE_VKQ_F32\n", 1),
        (FP16 + "        half2 * VKQ_combine    = (half2 *) KV_tmp;\n",
         FP16_HALF_VKQ + "        half2 * VKQ_combine    = (half2 *) KV_tmp;\n", 1),
        (FP16 + "            constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;\n",
         FP16_HALF_VKQ + "            constexpr int cpy_ne_D = cpy_ne < (DVp/2)/warp_size ? cpy_ne : (DVp/2)/warp_size;\n",
         2),
        (FP16 + "        constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;\n",
         FP16_HALF_VKQ + "        constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;\n",
         1),
    ])
    print(f"llama.cpp at {root}: ggml HIP float32 sums for the vision encoder")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
