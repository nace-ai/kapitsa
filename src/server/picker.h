// Which model to run, asked in the terminal: `kapitsa serve` and `kapitsa claude|codex|opencode|pi` without --model.
// The model first (arrow keys or j/k, Enter; Esc or Ctrl+C cancels), then its precision (Esc goes back; a model with
// one precision skips it). Each choice shows its download size, whether it is downloaded and whether it fits this Mac
// (the engine's own plan, offline, as the home screen's ★). The last choice comes first, so Enter twice repeats the
// last run: it is kept in state.json in Kapitsa's home (~/.cache/kapitsa, KAPITSA_HOME moves it), and a --model
// naming a catalog model updates it too; with no history, the best model for this Mac.
//
// Never asked where it could block a program: stdin, stdout or stderr not a terminal (the app, scripts, a brew
// service), TERM=dumb, CI, KAPITSA_NO_PROMPT, --yes, a background job. Those keep the plain behavior: the best model
// for this Mac.
#pragma once
#include "engine/engine.h"
#include "hub/hub.h"

#include <optional>
#include <string>

namespace ni::ui {

// Whether the picker may ask (see above); `yes`: --yes was given
bool can_prompt(bool yes);

// Asks, with the terminal in raw mode until the answer (restored on Esc, Ctrl+C and signals). The choice as --model
// takes it ("qwen3.6-35b-a3b:6bit"), after a line that confirms it (and the download it needs), or nullopt when
// cancelled. `opt` and `budget`: what the fit marks are planned for; `command`: the command that runs it
// ("kapitsa serve"), for the hint that names the choice with --model.
std::optional<std::string> pick_model(const EngineOptions& opt, size_t budget, const std::string& command);

// Keeps a catalog model named with --model (or picked) as the next picker's first choice, when a person runs the
// command in a terminal (the app's and scripts' runs leave it alone); other models are not kept
void remember(const hub::ModelRef& r);

// "qwen3.6-35b-a3b · 4-bit": a model id as a server reports it, for a line about the running server
std::string model_label(const std::string& id);

}  // namespace ni::ui
