// A model split across Macs by layers, over TCP (Qwen3.8-Flash-Next: Qwen4Exp::Split). The head is the Mac whose
// `kapitsa serve` takes the requests. It holds the first layers, the embedding and everything after the last layer:
// the final mix, the LM head, the MTP head and its drafts. Each stage (`kapitsa stage`) holds a run of the layers after
// that. A pass goes around the ring once. The head runs its layers and sends the rows' state to the first stage, each
// stage runs its layers and sends the result to the next, and the last one sends it back to the head, which finishes
// the pass. So a pass of R rows (a decode step: 1; a verify of the MTP head's drafts: 3) costs one hand-off per Mac,
// whatever R is. Sequence and checkpoint operations travel the same way, in order with the passes. The rows go as the
// fp32 state the layers read and write, so a split model's outputs are the same bits as the whole model's on one Mac.
//
// The wire: messages of a 64-byte little-endian header (pipe.cpp's Header) and a payload. A pass's payload is its rows
// of the residual streams (hc * hidden fp32), of the last block's output (hidden fp32) and of that block's injection
// weights (hc fp32): 51 KB a row for Flash-Next. A prefill's way back carries the streams alone, the injection already
// applied: 40 KB a row.
#pragma once
#include "model/remote.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ni {
class Qwen4Exp;
}

namespace ni::cluster {

constexpr int kStagePort = 9330;

// A stage as it answers the head's handshake
struct StageInfo {
  std::string addr;   // where the head (or the stage before it) reached it
  std::string name;   // its Mac's name
  int lo = 0, hi = 0; // its layers, [lo, hi)
  int rows = 0;       // the most rows a pass may send it (its prefill chunk)
  double mem = 0;     // its GPU memory budget (bytes)
  double gbs = 0;     // its memory bandwidth (GB/s, 0 unknown)
  bool fixed = false; // started with --layers: its layers are its own, not the head's to place
};

// The head as the placement sees it (src/cluster/place.h)
struct HeadInfo {
  std::string name;
  double mem = 0;     // its GPU memory budget (bytes)
  double gbs = 0;     // its memory bandwidth (GB/s, 0 unknown)
  int chunk = 2048;   // its prefill chunk (rows)
  bool may_stream = true;  // the placement may have the Macs stream the experts their memory does not hold (place.h)
};

// The head's side, the Remote of its Qwen4Exp (Qwen4Exp::set_split before the engine loads the model). A sender thread
// sends what the model posts, in order, a pass's rows once the GPU has reached them; a reader thread takes the ring's
// returns and writes the rows back, and the pass's caller, waiting, goes on. When a stage stops answering, every
// message still waiting is let go (a pass's outputs are garbage) and lost() says why.
class Link final : public Remote {
 public:
  // Connects to the stages, in the order of the ring (host:port each), and checks that they run this model (its
  // files' digest). Stages started without --layers take the layers the placement gives them (src/cluster/place.h:
  // every Mac's share in its memory, the fastest pass, from each Mac's budget and bandwidth; the head first); those
  // started with --layers keep theirs. Either way they must hold layers [first, layers) between them without a gap.
  // Null with *why when not.
  static std::shared_ptr<Link> connect(const std::vector<std::string>& addrs, const std::string& model_dir,
                                       const HeadInfo& me, std::string* why);
  ~Link() override;
  int first() const { return stages_.empty() ? 0 : stages_.front().lo; }  // the head holds layers [0, first)
  int rows() const;                                                       // the most rows a pass may carry
  const std::vector<StageInfo>& stages() const { return stages_; }
  double rtt_ms() const { return rtt_ms_; }  // an empty message around the ring, measured at connect (median of 5)
  bool streams() const { return streams_; }  // placed with the Macs streaming experts from their SSDs (place.h)

  bool pass(const Pass& p, const Rows& r) override;
  int new_seq(int limit) override;
  int grow(int seq, int need) override;
  void drop_kv(int seq) override;
  void reset(int seq) override;
  int new_checkpoint() override;
  void save(int seq, int ckpt, int pos) override;
  void restore(int seq, int ckpt) override;
  void flush(int seq, int n_pend) override;
  void rope(int seq, int rope_n, int rope_delta, const int32_t* table) override;
  std::string lost() const override;

  struct Impl;

 private:
  Link() = default;
  std::unique_ptr<Impl> d_;
  std::vector<StageInfo> stages_;
  double rtt_ms_ = 0;
  bool streams_ = false;
};

struct StageOptions {
  std::string model_dir;
  std::string host = "0.0.0.0";
  int port = kStagePort;
  std::string name;   // this Mac's name, as the head shows it
  double mem = 0;     // GPU memory budget (bytes)
  double gbs = 0;     // memory bandwidth (GB/s, 0 unknown)
  int chunk = 2112;   // the most rows a pass may bring (its prefill chunk)
  int lo = -1, hi = -1;  // --layers: loaded at start and kept; -1: the layers a head gives it, loaded when it does
  // called once listening, and when a head connects or leaves, or layers load (for a status line)
  std::function<void(const std::string&)> status;
};
// Serves heads, one at a time, until the process is stopped, holding the layers it was started with or, without them,
// the layers each head gives it (loaded when they change).
[[noreturn]] void run_stage(const StageOptions& o);

// The model's identity for the handshake: config.json's bytes and the names and sizes of its trunk's safetensors files
// (not ple.safetensors or mtp.safetensors: the n-gram table and the MTP head stay with the head)
std::string model_digest(const std::string& model_dir);

}  // namespace ni::cluster
