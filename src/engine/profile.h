// Kapitsa's two profiles (docs/profiles.md). `precise` runs the exact defaults. `fast` adds speedups under the
// documented quality gates. Each member names its model class and measurement setups.
#pragma once

namespace ni {

enum class Profile { kPrecise, kFast };

// KAPITSA_PROFILE=fast or precise (`kapitsa serve --profile` sets it). Cache only this explicit override.
// Unset or empty uses the caller's fallback, from the device profile table for a model-dependent default.
// The no-argument form keeps precise; an invalid override warns and selects precise regardless of the fallback.
Profile profile(Profile fallback = Profile::kPrecise);

// One switch's value under the profile. The switch's own environment variable wins when set ("0" off, other values
// on). When unset, the switch is on under the fast profile where `fast_here` is true: the member's class (AGENTS.md).
// Under the precise profile the switch stays off. `what` names the member in the one log line the fast profile prints
// for it.
bool profile_switch(const char* env, bool fast_here, const char* what, Profile fallback = Profile::kPrecise);

}  // namespace ni
