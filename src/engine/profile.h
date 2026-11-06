// Kapitsa's two profiles (docs/profiles.md). `precise` runs the exact defaults. `fast` also turns on every speedup that
// moves the outputs by no more than the fastest other engine does for that model on that Mac. Each member names the
// model and the Mac it was measured on.
#pragma once

namespace ni {

enum class Profile { kPrecise, kFast };

// KAPITSA_PROFILE=fast or precise (`kapitsa serve --profile` sets it). Unset or empty: precise. Read once.
Profile profile();

// One switch's value under the profile. The switch's own environment variable wins when set ("0" off, other values
// on). When unset, the switch is on under the fast profile where `fast_here` is true: the member's class (AGENTS.md).
// Under the precise profile the switch stays off. `what` names the member in the one log line the fast profile prints
// for it.
bool profile_switch(const char* env, bool fast_here, const char* what);

}  // namespace ni
