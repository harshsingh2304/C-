#pragma once
#include "common.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void ggml_backend_amx_mul_mat(ggml_backend_amx_context * ctx, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
