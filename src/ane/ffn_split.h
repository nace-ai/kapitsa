// GPU + Neural Engine split of a dense SwiGLU FFN during prefill (NI_ANE). Per layer and prefill chunk
// of T rows (T >= min_rows):
//   GPU: rotate + pack the post-norm input to int8 (IOSurfaces), signal `ready`
//   ANE: (queued) wait `ready`, run the W8A8 program on channels [gpu, I) -> partial, signal `done`
//   GPU: meanwhile its own channels [0, gpu) (the model's GEMMs, caller-encoded, adding into x), then
//        stage layer l+1's rotated int8 weights from the model's weights (any format) into the other weight set
//   GPU: wait `done`, join x += partial * scales (fp32)
// The ANE share is `units` x 512 intermediate channels. Weights are not stored twice: each chunk
// re-quantizes them one layer ahead into two double-buffered weight sets (~2 x 200 MB); only the
// per-row int8 scales (a few MB) are computed once and cached on disk.
//
// Design: channel split with a
// partial down projection joined in fp32, weights as IOSurface inputs of one multi-function MIL
// program, Hadamard rotations, the scale layout, one-layer-ahead staging.
#pragma once
#include "ane/ane.h"
#include "metal/gpu.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ni::ane {

struct Proj {  // MLX affine `bits`-bit, group `gs` (w uint32 [N][K * bits / 32], s/b bf16 [N][K / gs]), or bf16 (bits 16)
  Buffer w, s, b;
  uint32_t N = 0, K = 0;
  int bits = 4, gs = 64;
};
struct FfnLayer {
  Proj gate, up, down;
};

struct FfnConfig {
  uint32_t units = 25;              // ANE share in 512-channel units
  uint32_t min_rows = 512;          // smallest chunk that splits
  std::vector<uint32_t> fn_rows;    // ANE functions (rows); empty: every 128 rows in [min_rows, 2048]
  std::string cache_dir;            // row-scale cache directory ("" = no cache)
  std::string model_key;            // identifies the weights in the cache
  int fault_layer = -1;             // test: never start the ANE job of this layer (first chunk)
  int lanes = 1;                    // activation surface sets (2: the two-half wavefront schedule)
  bool compile_only = false;        // only compile the program into the ANE daemon's cache, allocate nothing
};

class FfnSplit {
 public:
  static constexpr uint32_t kUnit = 512, kSeg = 2560, kMaxRows = 2048;
  // ane_rotate's simdgroups for a hidden size (its 128-blocks split evenly, <= 8 each); 0 = not supported
  static uint32_t rotate_simdgroups(uint32_t hidden);

  // Allocates the surfaces, computes (or loads) the row scales, compiles/loads the program.
  // Throws std::runtime_error with the reason when the split cannot be used.
  FfnSplit(const FfnConfig& c, std::vector<FfnLayer> layers);
  ~FfnSplit();

  uint32_t units() const { return units_; }
  uint32_t gpu_channels() const { return gpu_; }
  uint32_t ane_channels() const { return ane_; }
  uint32_t min_rows() const { return min_rows_; }
  size_t metal_bytes() const { return bytes_; }
  const std::vector<uint32_t>& fn_rows() const { return fn_rows_; }
  double startup_ms() const { return startup_ms_; }
  std::string startup_note() const { return note_; }

  bool splits(int T) const;
  int lanes() const { return int(cfg_.lanes); }
  // Starts a chunk of T rows; every layer must then be encoded in order.
  void begin_chunk(int T);
  // Encodes layer li of the chunk (the simple schedule). xh: half [T][hidden] post-norm input;
  // x: fp32 residual [T][hidden], to which the FFN is added. gpu_part encodes the GPU's channels
  // (adding into x). ane=false: no ANE job (the join adds whatever the partial surface holds;
  // verify/benchmarks); stage=false: skip the weight staging (timing only).
  void encode_layer(Encoder& e, int li, const Buffer& xh, const Buffer& x, const std::function<void(Encoder&)>& gpu_part,
                    bool ane = true, bool stage = true);

  // ---- primitives (the wavefront schedule interleaves two lanes = row halves) ----
  // rotate + pack T rows of xh into the lane's int8 input surfaces
  void pack(Encoder& e, int lane, const Buffer& xh, int T);
  // queues the ANE job of layer li (weight set li & 1) on the lane's surfaces and signals ready
  // (ends the command buffer)
  void launch(Encoder& e, int lane, int li, int T);
  // waits for the lane's job (if one is pending) and adds its partial to x (T rows)
  void join(Encoder& e, int lane, const Buffer& x, int T);
  // stages layer li's rotated int8 weights into weight set `set`
  void stage_layer(Encoder& e, int li, int set);
  // After the chunk's command completed: false if the split failed (the chunk must be rerun on
  // the GPU alone; the split is then retired for the rest of the process).
  bool finish_chunk(double* ane_ms = nullptr);
  int last_jobs() const { return last_jobs_; }  // ANE evaluations of the last finished chunk
  // system wired memory change (MB) at the program's load, and after the first chunk with jobs
  double wired_at_load() const { return wired_load_; }
  double wired_after_run() const { return wired_run_; }
  void retire(const std::string& why);
  bool retired() const;
  std::string reason() const;

  // ---- diagnostics (verify, calibration) ----
  void fill_partial(uint16_t half_bits);       // CPU fill of the partial output surfaces
  double eval_sync(int set);                   // ANE alone, current chunk's function, weight set `set`
  void fill_status(uint32_t v);

 private:
  void init_resources();  // surfaces, row scales, program, bindings, handoff (the constructor's)
  void compile_program_only();  // FfnConfig::compile_only: layout-only surfaces, compile, unload
  void program_source(std::string& mil, std::vector<uint8_t>& blob);  // needs lane_[0]/set_[0] layouts + signs
  void free_all();        // releases whatever exists (destructor; a constructor that throws)
  struct Set {
    std::vector<Surface> wg, wu, wd;
    Surface sg, su, sd;
  };
  std::string function(uint32_t R) const;
  void compute_row_scales();
  Buffer row_scales(int li, int which) const;  // 0 gate, 1 up, 2 down
  int fn_for(int T) const;                     // smallest function holding T rows

  std::vector<FfnLayer> layers_;
  uint32_t hidden_ = 0, inter_ = 0, units_ = 0, ane_ = 0, gpu_ = 0, min_rows_ = 0;
  uint32_t rot_sg_ = 8;  // ane_rotate's simdgroups: they split the hidden's 128-blocks evenly, <= 8 each
  std::vector<uint32_t> dseg_, fn_rows_;
  struct Lane {
    std::vector<Surface> x;  // int8 [2560][2048] per hidden segment
    Surface tx, y;           // fp16 [1][2048], fp16 [hidden + 1][2048]
    uint64_t done = 0;       // pending job's done value (0: none)
  } lane_[2];
  Set set_[2];
  Buffer rot_, signs_, status_, rs_;
  std::vector<float> signs_host_;
  std::unique_ptr<Program> prog_;
  std::vector<std::shared_ptr<Program::Binding>> bind_[2][2];  // [weight set][lane][function]
  std::unique_ptr<Handoff> h_;
  FfnConfig cfg_;
  int fn_ = -1, T_ = 0, jobs_ = 0, last_jobs_ = 0;
  // Keepalive: the Neural Engine sleeps after about a second idle and its first evaluation then blocks for
  // 70-120 ms. For keep_window_s_ seconds after the last chunk a side thread runs one 512-row evaluation every
  // keep_period_ms_ (about 7 ms of ANE time, ~0.4% duty; nothing after the window), so a request after a pause finds it awake. Zero window: off.
  // The thread's state is shared with it, so a thread wedged in a Neural Engine evaluation can be detached (and
  // what it uses leaked) by the destructor instead of hanging it.
  struct KeepState {
    std::mutex m;  // stop / cv
    std::condition_variable cv;
    bool stop = false;
    std::atomic<bool> done{false};
    std::timed_mutex eval_mu;  // held by the keepalive evaluation (it uses lane 0 / set 0) and briefly by begin_chunk
  };
  void keep_loop(std::shared_ptr<KeepState> st, Program* prog, std::shared_ptr<Program::Binding> bind);
  double keep_window_s_ = 0, keep_period_ms_ = 2000;
  std::thread keep_;
  std::shared_ptr<KeepState> keep_st_ = std::make_shared<KeepState>();
  std::atomic<bool> in_chunk_{false};
  std::atomic<double> last_use_{0};
  double wired0_ = 0, wired_load_ = 0, wired_run_ = 0;
  bool fault_used_ = false;
  size_t bytes_ = 0;
  double startup_ms_ = 0;
  std::string note_;
};

double wired_mb();

// ---- configuration persisted per Mac / macOS / model (calibration) ----
std::string cache_dir();                         // ~/.cache/nace-infer-models/ane-cache
std::string model_key(const std::string& dir);   // weights' identity: path, file sizes and mtimes
std::string os_build();                          // kern.osversion
// "<chip>|<os build>|<model key>", the key of a calibration line
std::string machine_key(const std::string& chip, const std::string& model);
struct Calibration {
  uint32_t units = 0, min_rows = 0;
  std::string detail;  // free text (measurements)
};
bool read_calibration(const std::string& key, Calibration& out);
bool write_calibration(const std::string& key, const Calibration& c);
// Metal memory of a split (surfaces, rotation scratch, row scales) for an FFN of hidden x inter channels with
// `units` x 512 of them on the ANE, `layers` layers and `lanes` activation lanes (FfnSplit::metal_bytes)
size_t split_bytes(uint32_t hidden, uint32_t inter, uint32_t units, int layers, int lanes);
// Default share (units of 512 channels) for a chip without a calibration, for an FFN of `total` units
// on a GPU of `cores` cores; 0 = the GPU alone.
uint32_t default_units(const std::string& chip, int cores, uint32_t total, bool wave, bool gpu_mma = false);

}  // namespace ni::ane
