#include "ggml-amx.h"
#include "ggml-amx/common.h"
#include "ggml-amx/mmq.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#if defined(__gnu_linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <cstdlib>
#include <cstring>
#include <memory>

#if defined(__AMX_INT8__)

static const size_t TENSOR_ALIGNMENT = 64;

// AMX buffer interface

// TODO: change ggml_nbytes(...)
//      parallel `set_tensor`, `get_tensor` and `clear`

GGML_CALL static const char * ggml_backend_amx_buffer_get_name(ggml_backend_buffer_t buffer) {
  return "AMX";

  GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_amx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
  free(buffer->context);
}

GGML_CALL static void * ggml_backend_amx_buffer_get_base(ggml_backend_buffer_t buffer) {
  return (void *)(buffer->context);
}

GGML_CALL static void ggml_backend_amx_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_amx_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

GGML_CALL static bool ggml_backend_amx_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_amx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static ggml_backend_buffer_i ggml_backend_amx_buffer_interface = {
    /* .get_name        = */ ggml_backend_amx_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_amx_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_amx_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .set_tensor      = */ ggml_backend_amx_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_amx_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_amx_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_amx_buffer_clear,
    /* .reset           = */ NULL,
};

GGML_CALL static const char * ggml_backend_amx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
  return "AMX";

  GGML_UNUSED(buft);
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_amx_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
  printf("\n###### ggml_backend_amx_buffer_type_alloc_buffer: size = %d\n\n", size);
  void * data = aligned_alloc(TENSOR_ALIGNMENT, size);
  if (data == NULL) {
    fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
    return NULL;
  }

  return ggml_backend_buffer_init(buft, ggml_backend_amx_buffer_interface, data, size);
}

GGML_CALL static size_t ggml_backend_amx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
  return TENSOR_ALIGNMENT;

  GGML_UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_amx_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor* tensor) {
    size_t size = ggml_nbytes(tensor);
    printf("\n###### ggml_backend_amx_buffer_type_get_alloc_size, size = %d\n", size);

    return size;

    GGML_UNUSED(buft);
}

GGML_CALL static bool ggml_backend_amx_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
  return true;

  GGML_UNUSED(buft);
}

GGML_CALL ggml_backend_buffer_type_t ggml_backend_amx_buffer_type() {
  static struct ggml_backend_buffer_type ggml_backend_buffer_type_amx = {
    /* .iface = */ {
    /* .get_name         = */ ggml_backend_amx_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_amx_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_amx_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_amx_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_amx_buffer_type_is_host,
    },
    /* .context = */ NULL,
  };

  return &ggml_backend_buffer_type_amx;
}

// backend interface

GGML_CALL static const char * ggml_backend_amx_name(ggml_backend_t backend) {
  return "AMX";

  GGML_UNUSED(backend);
}

GGML_CALL static void ggml_backend_amx_free(ggml_backend_t backend) {
  ggml_backend_amx_context * ctx = (ggml_backend_amx_context *)backend->context;
  delete ctx;
  delete backend;
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_amx_get_default_buffer_type(ggml_backend_t backend) {
    //return ggml_backend_amx_buffer_type();
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(backend);
}

GGML_CALL static enum ggml_status ggml_backend_amx_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
  ggml_backend_amx_context * ctx = (ggml_backend_amx_context *)backend->context;

  for (int i = 0; i < cgraph->n_nodes; i++) {
    struct ggml_tensor * node = cgraph->nodes[i];

    switch (node->op) {
      case GGML_OP_MUL_MAT:
        ggml_backend_amx_mul_mat(ctx, node);
        break;

      case GGML_OP_NONE:
      case GGML_OP_RESHAPE:
      case GGML_OP_VIEW:
      case GGML_OP_PERMUTE:
      case GGML_OP_TRANSPOSE:
        break;

      default:
        fprintf(stderr, "%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        GGML_ASSERT(false);
      }
    }

  return GGML_STATUS_SUCCESS;

  GGML_UNUSED(backend);
}

GGML_CALL static bool ggml_backend_amx_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {

  if (op->op != GGML_OP_MUL_MAT) {
    return false;
  }

  const struct ggml_tensor * src0 = op->src[0];
  const struct ggml_tensor * src1 = op->src[1];

  const enum ggml_type type = src0->type;
  const int64_t ne0 = op->ne[0];

  bool is_training = src0->grad || src1->grad;

  bool has_amx_kernels = (type == GGML_TYPE_Q4_0);

  // handle only 2d gemm for now
  auto is_contiguous_2d = [](const struct ggml_tensor * t) {
    return ggml_is_contiguous(t) && t->ne[3] == 1 && t->ne[2] == 1;
  };

  bool can_use_amx =
      is_contiguous_2d(src0) &&       // src0 must be contiguous
      is_contiguous_2d(src1) &&       // src1 must be contiguous
      !is_training &&                 // inference only
      src1->type == GGML_TYPE_F32 &&  // src1 must be float32
      has_amx_kernels &&              // with amx kernel impls
      ne0 % (TILE_N * 2) == 0;        // out_features is 32x

  return can_use_amx;

  GGML_UNUSED(backend);
}

GGML_CALL static bool ggml_backend_amx_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
  //bool res = buft->iface.get_name == ggml_backend_amx_buffer_type_get_name;
  //printf("\n###### ggml_backend_amx_supports_buft, buft->iface.get_name = %s\n", buft->iface.get_name(buft));
  //printf("\n###### ggml_backend_amx_supports_buft, res: %s", res ? "true" : "false");
  //return res;

  return ggml_backend_buft_is_host(buft);

  GGML_UNUSED(backend);
}

static struct ggml_backend_i ggml_backend_amx_i = {
  /* .get_name                = */ ggml_backend_amx_name,
  /* .free                    = */ ggml_backend_amx_free,
  /* .get_default_buffer_type = */ ggml_backend_amx_get_default_buffer_type,
  /* .set_tensor_async        = */ NULL,
  /* .get_tensor_async        = */ NULL,
  /* .cpy_tensor_async        = */ NULL,
  /* .synchronize             = */ NULL,
  /* .graph_plan_create       = */ NULL,
  /* .graph_plan_free         = */ NULL,
  /* .graph_plan_update       = */ NULL,
  /* .graph_plan_compute      = */ NULL,
  /* .graph_compute           = */ ggml_backend_amx_graph_compute,
  /* .supports_op             = */ ggml_backend_amx_supports_op,
  /* .supports_buft           = */ ggml_backend_amx_supports_buft,
  /* .offload_op              = */ NULL,
  /* .event_new               = */ NULL,
  /* .event_free              = */ NULL,
  /* .event_record            = */ NULL,
  /* .event_wait              = */ NULL,
  /* .event_synchronize       = */ NULL,
};

static ggml_guid_t ggml_backend_amx_guid() {
  static ggml_guid guid = { 0x13, 0xb8, 0xa4, 0xc4, 0xba, 0xfe, 0x51, 0x67, 0x87, 0x44, 0x55, 0x15, 0xb2, 0x35, 0x62, 0x3e };
  return &guid;
}

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

GGML_CALL static bool ggml_amx_init() {
#if defined(__gnu_linux__)
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
    fprintf(stderr, "AMX is not ready to be used!\n");
    return false;
  }
  return true;
#elif defined(_WIN32)
  return true;
#endif
}

// define amx tile config data structure
struct tile_config_t{
  uint8_t palette_id = 0;
  uint8_t start_row = 0;
  uint8_t reserved_0[14] = {0};
  uint16_t colsb[16] = {0};
  uint8_t rows[16] = {0};
};

// Notes: amx tile config
//
// Typically, TMUL calculates A and B of size 16 x 64 containing INT8 values,
// and accumulate the result to a 16 x 16 matrix C containing INT32 values,
//
// As many GGUF quantized types as `block_size` of 32, so a 16-16-32 config is used
// instead of the normally used 16-16-64 config.
//
//   Block A: {16, 32}, dtype = int8_t
//   Block B: {16, 32}, dtype = uint8_t/int8_t
//   Block C: {16, 16}, dtype = int32_t
//
// Block B needs to be prepacked to vnni format before feeding into  TMUL:
//   packed_B: from {n, k} to {k/vnni_blk, n, vnni_blck}, viewed in 2d, we get {8, 64}
//
// Therefore, we get tileconfig:
//             A    B    C
//    rows    16    8   16
//    colsb   32   64   16
//
// For tile distribution, follow a 2-2-4 pattern, e.g. A used TMM2-TMM3, B used TMM0-TMM1,
// C used TMM4-TMM7:
//            B TMM0  B TMM1
//    A TMM2  C TMM4  C TMM6
//    A TMM3  C TMM5  C TMM7
//
// Each `amx` kernel handles 4 blocks at a time: 2MB * 2NB, when m < 2 * BLOCK_M, unpack A
// will be needed.
//
// Here another commonly used pattern 1-3-3 is skipped, as it is mostly used when m <=16;
// and the sinlge batch gemm (m=1) has a special fast path with `avx512-vnni`.
//
// ref: https://www.intel.com/content/www/us/en/developer/articles/code-sample/
//   advanced-matrix-extensions-intrinsics-functions.html
//

#define TC_CONFIG_TILE(i, r, cb) tc.rows[i] = r; tc.colsb[i] = cb
GGML_CALL static void ggml_tile_config_init() {
  static thread_local tile_config_t tc;
  tile_config_t current_tc;
  _tile_storeconfig(&current_tc);

  // load only when config changes
  if (tc.palette_id == 0 || (memcmp(&current_tc.colsb, &tc.colsb, sizeof(uint16_t) * 8) != 0 &&
                             memcmp(&current_tc.rows, &tc.rows, sizeof(uint8_t) * 8) != 0)) {
    tc.palette_id = 1;
    tc.start_row = 0;
    TC_CONFIG_TILE(TMM0, 8, 64);
    TC_CONFIG_TILE(TMM1, 8, 64);
    TC_CONFIG_TILE(TMM2, 16, 32);
    TC_CONFIG_TILE(TMM3, 16, 32);
    TC_CONFIG_TILE(TMM4, 16, 64);
    TC_CONFIG_TILE(TMM5, 16, 64);
    TC_CONFIG_TILE(TMM6, 16, 64);
    TC_CONFIG_TILE(TMM7, 16, 64);
    _tile_loadconfig(&tc);
  }
}

ggml_backend_t ggml_backend_amx_init() {

  // invoke a Linux system call to request access to AMX features
  ggml_amx_init();

  // init tile config
  ggml_tile_config_init();

  // backend context
  ggml_backend_amx_context * ctx = new ggml_backend_amx_context;

  // ggml amx backend
  ggml_backend_t backend = new ggml_backend {
    /* .guid      = */ ggml_backend_amx_guid(),
    /* .interface = */ ggml_backend_amx_i,
    /* .context   = */ ctx,
  };

  return backend;
}

GGML_CALL static ggml_backend_t ggml_backend_reg_amx_init(const char * params, void * user_data) {
    return ggml_backend_amx_init();

    GGML_UNUSED(params);
    GGML_UNUSED(user_data);
}

extern "C" GGML_CALL void ggml_backend_amx_reg_devices();

GGML_CALL void ggml_backend_amx_reg_devices() {
  ggml_backend_register("AMX", ggml_backend_reg_amx_init, ggml_backend_amx_buffer_type(), NULL);
}

GGML_CALL bool ggml_backend_is_amx(ggml_backend_t backend) {
  return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_amx_guid());
}

void ggml_backend_amx_set_n_threads(ggml_backend_t backend_amx, int n_threads) {
  GGML_ASSERT(ggml_backend_is_amx(backend_amx));

  ggml_backend_amx_context * ctx = (ggml_backend_amx_context *)backend_amx->context;
  ctx->n_threads = n_threads;
}

#else // if defined(__AMX_INT8__)

ggml_backend_t ggml_backend_amx_init(void) {
  fprintf(stderr, "GGML is not compiled with AMX support!\n");
  return ggml_backend_t{};
}

extern "C" GGML_CALL void ggml_backend_amx_reg_devices();

GGML_CALL void ggml_backend_amx_reg_devices() {
  fprintf(stderr, "GGML is not compiled with AMX support!\n");
}

void ggml_backend_amx_set_n_threads(ggml_backend_t backend_amx, int n_threads) {
  fprintf(stderr, "GGML is not compiled with AMX support!\n");

  GGML_UNUSED(backend_amx);
  GGML_UNUSED(n_threads);
}

#endif
