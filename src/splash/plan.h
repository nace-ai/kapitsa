// The arguments Kapitsa adds to `splash serve-native` when the user gave none for them (docs/splash-backend.md). A pure
// function of the user's arguments and the memory free now, so a test needs no Splash and no Mac state.
#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ni::splash {

// Appended after `user` (an argument the user gave is never replaced or doubled):
//  - --idle-release off: Splash's default frees the weights after 10 idle minutes and the next request waits ~7 s for them;
//  - --expert-memory <N>G when free+reclaimable memory is below what Splash's own auto plan needs (about 36 GiB, plus its
//    2 GiB protected for macOS): the free memory less 14 GiB, whole GiB, at least 8; the 16 GiB cache that started the
//    engine beside other processes at 30.4 GiB free is the one point this was seen to hold at. Above that memory, or with
//    less than 22 GiB free, the child's auto plan is left to decide (and to refuse with its own message);
//  - --persistent-cache on when `persistent_cache` (Splash's SSD cache across restarts; its default is off), with
//    --cache-dir `cache_dir` (one directory per model, so models never read each other's states) when that is not empty.
// `note` receives what was added and why, for the log.
std::vector<std::string> default_engine_args(const std::vector<std::string>& user, uint64_t available_bytes,
                                             bool persistent_cache, const std::string& cache_dir,
                                             std::string* note);

// The 128-bit cache_salt of a conversation (--splash-conversation-salt): a hash of its messages up to and including the
// first user message, which every later turn repeats as it is, so a conversation keeps its salt and another one gets its own.
// A history with no user message hashes all of it. Two 64-bit FNV-1a hashes with different bases make the 128 bits.
struct Salt { uint64_t lo = 0, hi = 0; };
// A client's own cache_salt, hashed as Splash's frontend does (BLAKE2b, 16-byte digest, person "splash-salt"); the engine keys
// its cache by it, so a request shares cached prompts only with requests of the same salt.
Salt client_salt(const std::string& salt);
Salt conversation_salt(const nlohmann::ordered_json& messages);

}  // namespace ni::splash
