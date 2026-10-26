#pragma once
// A JSON file parsed once while it stays the same file. The planners (kapitsa doctor, the server's model picker,
// default_context / auto_slots / plan_load) ask a model's config.json for its type and shapes hundreds of times per
// model, and Nemotron 3.5's (Bolt, Super-VL) are 8 MB of quantized-layer lists: a parse each time took doctor minutes.
#include <nlohmann/json.hpp>

#include <memory>
#include <string>

namespace ni {

// The file at `path` parsed: null if it is missing or not valid JSON. Kept for the last few files read, keyed by the
// file's device, inode, size and modification time, so a file written again is read again. Thread-safe; the value
// lives as long as the pointer does.
std::shared_ptr<const nlohmann::json> json_file(const std::string& path);

}  // namespace ni
