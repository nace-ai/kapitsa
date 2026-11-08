// A C test for the Kapitsa C API (include/kapitsa.h).
// Loads a model, generates tokens with a streaming callback, and cancels mid-stream.
//
// Usage: kapitsa_c_test <model_dir> [max_tokens]
//   model_dir   path to a Kapitsa model directory (e.g. the 4B)
//   max_tokens  tokens to generate before the callback stops (default 16)
#include "../include/kapitsa.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int count;
    int limit;
    int cancel_at;   // cancel via kapitsa_cancel() at this token (0 = don't)
    kapitsa_ctx* ctx; // for cancel
} State;

static bool on_token(const char* token, size_t len, void* userdata) {
    State* s = (State*)userdata;
    s->count++;
    fwrite(token, 1, len, stdout);
    fflush(stdout);
    if (s->cancel_at > 0 && s->count >= s->cancel_at && s->ctx) {
        fprintf(stderr, "\n[cancel requested at token %d]\n", s->count);
        kapitsa_cancel(s->ctx);
        return true;  // let the engine see the cancel flag
    }
    if (s->count >= s->limit) {
        fprintf(stderr, "\n[stopped by callback at token %d]\n", s->count);
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model_dir> [max_tokens] [cancel_at]\n", argv[0]);
        return 2;
    }
    const char* model_dir = argv[1];
    int max_tokens = argc > 2 ? atoi(argv[2]) : 16;
    int cancel_at = argc > 3 ? atoi(argv[3]) : 0;

    printf("kapitsa C API %s\n", kapitsa_version());

    // init
    kapitsa_status st = kapitsa_init();
    if (st != KAPITSA_OK) {
        fprintf(stderr, "init failed: %d %s\n", st, kapitsa_last_error(NULL));
        return 1;
    }

    // load
    kapitsa_load_opts lo;
    memset(&lo, 0, sizeof(lo));
    lo.model_path = model_dir;
    lo.context_length = 4096;

    kapitsa_ctx* ctx = NULL;
    st = kapitsa_load(&lo, &ctx);
    if (st != KAPITSA_OK) {
        fprintf(stderr, "load failed: %d %s\n", st, kapitsa_last_error(NULL));
        return 1;
    }
    printf("model loaded\n");

    // generate
    kapitsa_generate_opts go;
    memset(&go, 0, sizeof(go));
    go.prompt = "What is 2+2? Answer in one word.";
    go.temperature = 0.0f;
    go.max_tokens = max_tokens + 32;  // room beyond the callback's limit

    State state;
    memset(&state, 0, sizeof(state));
    state.limit = max_tokens;
    state.cancel_at = cancel_at;
    state.ctx = ctx;

    printf("generating (limit %d, cancel_at %d):\n", max_tokens, cancel_at);
    st = kapitsa_generate(ctx, &go, on_token, &state);
    printf("\ngenerate returned: %d (%d tokens)\n", st, state.count);
    if (st == KAPITSA_ERROR_CANCELLED) printf("cancelled as expected\n");
    if (st != KAPITSA_OK && st != KAPITSA_ERROR_CANCELLED)
        fprintf(stderr, "error: %s\n", kapitsa_last_error(ctx));

    // cleanup
    kapitsa_free(ctx);
    printf("done\n");
    return (st == KAPITSA_OK || st == KAPITSA_ERROR_CANCELLED) ? 0 : 1;
}
