// Oracle for the vision encoder: run llama.cpp's mmproj (libmtmd) on an image and dump the
// projected embeddings the language model would see.
//
// Build (against the user's llama.cpp build, no tree changes):
//   L=$HOME/gitprojects/llama.cpp
//   g++ -O2 -std=c++17 -I$L/include -I$L/ggml/include -I$L/tools/mtmd tools/oracle/mtmd_embd.cpp \
//       -L$L/build/bin -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin -o $JOB/mtmd_embd
// Run:
//   mtmd_embd <text-model.gguf (vocab only)> <mmproj.gguf> <image> <out.bin> [max_tokens]
// out.bin: u32 n_tokens, u32 nx, u32 ny, u32 n_embd, then n_tokens*n_embd f32 (row-major).
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s model.gguf mmproj.gguf image out.bin [max_tokens]\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const char * mmproj_path = argv[2];
    const char * image_path = argv[3];
    const char * out_path = argv[4];
    int max_tokens = argc > 5 ? atoi(argv[5]) : 0;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "failed to load vocab from %s\n", model_path);
        return 1;
    }
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = getenv("ORACLE_GPU") != nullptr;
    cp.n_threads = 32;
    cp.print_timings = true;
    cp.warmup = false;
    if (max_tokens > 0) {
        cp.image_max_tokens = max_tokens;
    }
    mtmd_context * ctx = mtmd_init_from_file(mmproj_path, model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to load mmproj %s\n", mmproj_path);
        return 1;
    }
    mtmd_helper_init_opt opt = mtmd_helper_init_opt_default();
    mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, image_path, false, opt);
    if (!bw.bitmap) {
        fprintf(stderr, "failed to load image %s\n", image_path);
        return 1;
    }
    std::string prompt = std::string(mtmd_default_marker());
    mtmd_input_text text;
    text.text = prompt.c_str();
    text.text_len = prompt.size();
    text.add_special = false;
    text.parse_special = true;
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bitmaps[1] = { bw.bitmap };
    int32_t rc = mtmd_tokenize(ctx, chunks, &text, bitmaps, 1);
    if (rc != 0) {
        fprintf(stderr, "mtmd_tokenize failed: %d\n", rc);
        return 1;
    }
    size_t n_chunks = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            continue;
        }
        const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(chunk);
        size_t n_tokens = mtmd_image_tokens_get_n_tokens(it);
        // grid from the decoder positions of the last token: (x, y) of token n-1 relative to token 0
        mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(it, 0, n_tokens - 1);
        uint32_t nx = last.x + 1, ny = last.y + 1;
        if (mtmd_encode_chunk(ctx, chunk) != 0) {
            fprintf(stderr, "encode failed\n");
            return 1;
        }
        float * embd = mtmd_get_output_embd(ctx);
        int n_embd = llama_model_n_embd(model);
        // vocab-only models report n_embd 0: take it from the mmproj metadata instead
        if (n_embd <= 0) {
            n_embd = getenv("ORACLE_N_EMBD") ? atoi(getenv("ORACLE_N_EMBD")) : 2560;
        }
        FILE * f = fopen(out_path, "wb");
        uint32_t hdr[4] = { (uint32_t) n_tokens, nx, ny, (uint32_t) n_embd };
        fwrite(hdr, 4, 4, f);
        fwrite(embd, 4, n_tokens * n_embd, f);
        fclose(f);
        printf("tokens %zu grid %ux%u n_embd %d -> %s\n", n_tokens, nx, ny, n_embd, out_path);
        double s = 0, ss = 0;
        for (size_t k = 0; k < n_tokens * (size_t) n_embd; k++) { s += embd[k]; ss += embd[k] * embd[k]; }
        printf("mean %.6f rms %.6f first %.5f %.5f %.5f %.5f\n", s / (n_tokens * n_embd), sqrt(ss / (n_tokens * n_embd)), embd[0], embd[1], embd[2], embd[3]);
    }
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bw.bitmap);
    mtmd_free(ctx);
    llama_model_free(model);
    return 0;
}
