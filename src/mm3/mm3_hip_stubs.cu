// src/mm3/mm3_hip_stubs.cu - maya-mm3's AMD (HIP) build: what a GGUF pack runs on (mm3_gguf.cu over maya-ds4's ds4_iq
// kernels) and nothing it does not.  The NVIDIA-only parts are left out - the EXL3 kernels (never ported to HIP: the
// setup refuses EXL3 on AMD), the tensor-parallel path, the DSpark draft and the vision tower (both ride on the EXL3
// repo's files) - and stand here as their interfaces that hand GGUF matrices on and refuse the rest with a message.
// CMakeLists.txt builds this file in their place when STRATA_ENABLE_HIP is on.
#include "mm3_dspark.hpp"
#include "mm3_exl3.hpp"
#include "mm3_gguf.cuh"
#include "mm3_tp.hpp"
#include "mm3_vision.hpp"

#include <cstdio>

namespace strata::mm3::exl3 {

namespace gg = ::mm3::gg;

namespace {
void refuse(const char* what) { std::fprintf(stderr, "maya-mm3: %s: EXL3 matrices run on NVIDIA GPUs only\n", what); }
}  // namespace

bool k2_supported(int, int) { return false; }
bool supported(const Mat&) { return false; }
void set_skip(const int*) {}
int mv_splits(const Mat& m) { return m.fmt == gg::kFmtGG ? 1 : 0; }
int mv_cols(const Mat& m) { return m.fmt == gg::kFmtGG ? kMaxMvCols : 0; }
unsigned wait_timeouts() { return gg::wait_timeouts(); }
int launch_errors() { return gg::launch_errors(); }
bool prepare(cudaStream_t) { return true; }
bool several_ok() { return true; }
bool mixed_supported(const ExpertLayout&) { return false; }
bool mma_supported() { return false; }
bool blas_enable(cudaStream_t, size_t) { return false; }
size_t blas_bytes(cudaStream_t) { return 0; }
bool blas_wanted() { return false; }

bool mv(const MvJob* jobs, int n, cudaStream_t s) {
    for (int i = 0; i < n; ++i)
        if (jobs[i].m == nullptr || jobs[i].m->fmt != gg::kFmtGG) {
            refuse("mv");
            return false;
        }
    return gg::mv(jobs, n, s);
}
void gemm_rows(const Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta, cudaStream_t s) {
    if (m.fmt == gg::kFmtGG) gg::gemm_rows(m, x16, ldx, rows, y, ldy, beta, s);
    else refuse("gemm_rows");
}
void moe_gate_up(const MoeArgs& a, const float* x, float* h, cudaStream_t s) {
    if (a.lay.fmt == gg::kFmtGG) gg::moe_gate_up(a, x, h, s);
    else refuse("moe_gate_up");
}
void moe_down(const MoeArgs& a, const float* h, const float* sh_out, float* out, cudaStream_t s) {
    if (a.lay.fmt == gg::kFmtGG) gg::moe_down(a, h, sh_out, out, s);
    else refuse("moe_down");
}
bool win_experts(const WinArgs& a, cudaStream_t s) {
    if (a.lay.fmt == gg::kFmtGG) return gg::win_experts(a, s);
    refuse("win_experts");
    return false;
}
void set_gate_up(const SetArgs& sa, const float* x, int ldx, const int* row_idx, uint16_t* h16, int ldh,
                 cudaStream_t s) {
    if (sa.lay.fmt == gg::kFmtGG) gg::set_gate_up(sa, x, ldx, row_idx, h16, ldh, s);
    else refuse("set_gate_up");
}
void set_down(const SetArgs& sa, const uint16_t* h16, int ldh, float* y, int ldy, cudaStream_t s) {
    if (sa.lay.fmt == gg::kFmtGG) gg::set_down(sa, h16, ldh, y, ldy, s);
    else refuse("set_down");
}

}  // namespace strata::mm3::exl3

namespace mm3 {

namespace {
const char* kNoTp = "tensor parallel (--tp) runs on NVIDIA GPUs only (an AMD GPU: the layer split)";
const char* kNoDraft = "the DSpark draft runs on NVIDIA GPUs only";
const char* kNoVision = "maya-mm3's vision tower runs on NVIDIA GPUs only";
}  // namespace

// ---- the tensor-parallel path: never loads (Model::load takes the layer split then)
struct TpEngine::Impl {
    Stats st;
    std::vector<double> nll;
};
TpEngine::TpEngine() : p(new Impl) {}
TpEngine::~TpEngine() = default;
bool TpEngine::load(const Options&, std::string& err) {
    err = kNoTp;
    return false;
}
bool TpEngine::forward(const int*, int, int, int, float*, std::string& err) {
    err = kNoTp;
    return false;
}
int TpEngine::vocab() const { return 0; }
int TpEngine::max_ctx() const { return 0; }
int TpEngine::bos() const { return 0; }
int TpEngine::eos() const { return 0; }
const Stats& TpEngine::stats() const { return p->st; }
std::vector<std::string> TpEngine::info_lines() const { return {}; }
std::vector<std::string> TpEngine::vram_lines() const { return {}; }
bool TpEngine::save_profile(const std::string&, std::string& err) const {
    err = kNoTp;
    return false;
}
int TpEngine::draft_block() const { return 0; }
bool TpEngine::draft(int, int, int, std::vector<int>&, std::vector<float>&, std::string& err) {
    err = kNoTp;
    return false;
}
bool TpEngine::forward_cands(const int*, int, int, int, Cands&, std::string& err) {
    err = kNoTp;
    return false;
}
void TpEngine::set_progress(std::function<bool(int)>) {}
void TpEngine::set_ppl(bool) {}
const std::vector<double>& TpEngine::ppl_nll() const { return p->nll; }
int TpEngine::done() const { return 0; }
bool TpEngine::draft_sampled(int, int, int, const DraftSampling&, const std::vector<float>&, std::vector<int>&,
                             std::vector<float>&, std::vector<std::vector<std::pair<int, float>>>&, std::string& err) {
    err = kNoTp;
    return false;
}
const VisionGeo* TpEngine::vision() const { return nullptr; }
bool TpEngine::encode_image(const uint8_t*, int, int, std::vector<float>&, std::string& err) {
    err = kNoTp;
    return false;
}
void TpEngine::set_images(std::vector<Model::Image>) {}

// ---- the DSpark draft: never loads (a GGUF pack has none; the engine turns --dspark off for it)
struct Draft::Impl {
    DraftGeo g;
};
Draft::Draft() : p(new Impl) {}
Draft::~Draft() { delete p; }
bool Draft::load(const Pack&, int, cudaStream_t, cublasHandle_t, const strata::mm3::exl3::Mat*, const uint16_t*, int,
                 std::string& err) {
    err = kNoDraft;
    return false;
}
void Draft::set_emb_rows(std::function<void(int, uint16_t*)>) {}
const DraftGeo& Draft::geo() const { return p->g; }
bool Draft::context(const float*, int, int, std::string& err, size_t) {
    err = kNoDraft;
    return false;
}
bool Draft::context_tap(int, const float*, int, std::string& err) {
    err = kNoDraft;
    return false;
}
bool Draft::context_finish(int, int, std::string& err) {
    err = kNoDraft;
    return false;
}
bool Draft::propose(int, int, int, std::vector<int>&, std::vector<float>&, float*, std::string& err) {
    err = kNoDraft;
    return false;
}
bool Draft::propose_sampled(int, int, int, const k::SampleArgs&, const std::vector<float>&, std::vector<int>&,
                            std::vector<float>&, std::vector<std::vector<std::pair<int, float>>>&, std::string& err) {
    err = kNoDraft;
    return false;
}
size_t Draft::bytes() const { return 0; }

// ---- the vision tower: never in a pack here (a GGUF set has none; an EXL3 pack is refused on AMD)
struct Vision::Impl {
    VisionGeo g;
};
Vision::Vision() : p(new Impl) {}
Vision::~Vision() = default;
bool Vision::in_pack(const Pack&) { return false; }
bool Vision::load(const Pack&, int, std::string& err) {
    err = kNoVision;
    return false;
}
void Vision::set_lender(std::function<void*(size_t)>, std::function<void()>) {}
const VisionGeo& Vision::geo() const { return p->g; }
int Vision::tokens(int, int) const { return 0; }
bool Vision::encode(const uint8_t*, int, int, std::vector<float>&, std::string& err) {
    err = kNoVision;
    return false;
}
size_t Vision::bytes() const { return 0; }
size_t Vision::host_bytes() const { return 0; }
size_t Vision::room_bytes() const { return 0; }
size_t Vision::last_bytes() const { return 0; }
int Vision::device() const { return 0; }

}  // namespace mm3
