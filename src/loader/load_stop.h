// A stop asked while a model's weights are read: kapitsa serve's POST /stop during the first load, which used to wait
// for the whole load (minutes from an external drive). A signal does not reach it: the stop handlers are installed after
// the first load. The server says what a stop does; the readers reach a stop point before each 32 MB piece and each
// tensor read (src/loader/safetensors.cpp) and before each streamed-expert read (src/model/expert_store.cpp's fill),
// and the first one there once a stop was asked does it. Until an action is set nothing stops: ni_run, bench and the SDK read as
// before.
#pragma once

namespace ni {

using LoadStopAction = void (*)();  // ends the process: it never returns

void set_load_stop_action(LoadStopAction action);
void ask_load_stop();  // from any thread; the readers act on it at their next stop point
// A reader's stop point: returns at once unless a stop was asked and an action is set. Then the first reader here runs
// the action; another one waits for the process to end instead of reading on.
void load_stop_point();

}  // namespace ni
