#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

template<typename T = void>
struct Lazy;

template<typename T = void>
struct Lazy;

struct EmbeddingResult {
  const float *embeddings;
  uintptr_t len;
  const char *error;
};



extern "C" {

bool init_model(const char *config_path_raw,
                const char *tokenizer_path_raw,
                const char *weights_path_raw,
                bool approximate_gelu);

EmbeddingResult generate_embeddings(const char *text);

void free_embeddings(EmbeddingResult result);

} // extern "C"
