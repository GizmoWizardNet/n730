/*
 * Build (Windows, MSVC):
 *   cl /O2 /arch:AVX2 /LD n730core.cpp /Fe:n730core.dll
 *
 * Build (Windows, MinGW/g++):
 *   g++ -O3 -march=native -shared -o n730core.dll n730core.cpp
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>

#ifdef _WIN32
  #define N730_API extern "C" __declspec(dllexport)
#else
  #define N730_API extern "C" __attribute__((visibility("default")))
#endif

static const uint8_t  FILE_MAGIC[8]  = {'N','7','3','0',0,1,0,0};
static const uint8_t  LAYER_MAGIC[4] = {'L','Y','R',0};
static const uint32_t PAGE_SIZE      = 4096;


static const int PREC_INT2 = 2;
static const int PREC_INT4 = 4;
static const int PREC_INT8 = 8;
static const int PREC_FP16 = 16;
static const int N730_OK              =  0;
static const int N730_ERR_BAD_MAGIC   = -1;
static const int N730_ERR_FILE        = -2;
static const int N730_ERR_ALLOC       = -3;
static const int N730_ERR_BAD_LAYER   = -4;
static const int N730_ERR_BAD_PREC    = -5;
static const int N730_ERR_NULL        = -6;

static inline uint32_t bswap32(uint32_t x) {
    return ((x & 0xFF000000u) >> 24) |
           ((x & 0x00FF0000u) >>  8) |
           ((x & 0x0000FF00u) <<  8) |
           ((x & 0x000000FFu) << 24);
}

static inline float bswap_float(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    u = bswap32(u);
    memcpy(&x, &u, 4);
    return x;
}

static inline uint32_t read_be32(FILE* f) {
    uint32_t v = 0;
    fread(&v, 4, 1, f);
    return bswap32(v);
}

static inline float read_be_float(FILE* f) {
    float v = 0.0f;
    fread(&v, 4, 1, f);
    return bswap_float(v);
}

static void dequant_int8(
    const uint8_t* __restrict src,
    float*         __restrict dst,
    int32_t n_elements,
    float scale,
    float zero_point
) {
    for (int32_t i = 0; i < n_elements; ++i) {
        dst[i] = ((float)src[i] - zero_point) * scale;
    }
}

static void dequant_int4(
    const uint8_t* __restrict src,
    float*         __restrict dst,
    int32_t n_elements,
    float scale,
    float zero_point
) {
    int32_t n_bytes = (n_elements + 1) / 2;
    int32_t out = 0;
    for (int32_t i = 0; i < n_bytes && out < n_elements; ++i) {
        uint8_t byte = src[i];
        dst[out++] = ((float)(byte & 0x0F) - zero_point) * scale;
        if (out < n_elements) {
            dst[out++] = ((float)((byte >> 4) & 0x0F) - zero_point) * scale;
        }
    }
}

static void dequant_int2(
    const uint8_t* __restrict src,
    float*         __restrict dst,
    int32_t n_elements,
    float scale,
    float zero_point
) {
    int32_t n_bytes = (n_elements + 3) / 4;
    int32_t out = 0;
    for (int32_t i = 0; i < n_bytes && out < n_elements; ++i) {
        uint8_t byte = src[i];
        if (out < n_elements) dst[out++] = ((float)(byte & 0x03) - zero_point) * scale;
        if (out < n_elements) dst[out++] = ((float)((byte >> 2) & 0x03) - zero_point) * scale;
        if (out < n_elements) dst[out++] = ((float)((byte >> 4) & 0x03) - zero_point) * scale;
        if (out < n_elements) dst[out++] = ((float)((byte >> 6) & 0x03) - zero_point) * scale;
    }
}

static float fp16_to_float(uint16_t h) {
    uint32_t sign     = (h & 0x8000u) << 16;
    uint32_t exponent = (h & 0x7C00u) >> 10;
    uint32_t mantissa = (h & 0x03FFu);

    uint32_t result;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            // Subnormal
            exponent = 1;
            while (!(mantissa & 0x0400)) { mantissa <<= 1; exponent--; }
            mantissa &= 0x03FF;
            result = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        // Inf or NaN
        result = sign | 0x7F800000u | (mantissa << 13);
    } else {
        result = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }

    float f;
    memcpy(&f, &result, 4);
    return f;
}

static void dequant_fp16(
    const uint8_t* __restrict src,
    float*         __restrict dst,
    int32_t n_elements
) {
    const uint16_t* src16 = (const uint16_t*)src;
    for (int32_t i = 0; i < n_elements; ++i) {
        dst[i] = fp16_to_float(src16[i]);
    }
}

#pragma pack(push, 1)
struct LayerBlockHeader {
    uint8_t  magic[4];
    uint32_t layer_idx;   // big-endian
    uint8_t  prec_id;
    uint32_t rows;        // big-endian
    uint32_t cols;        // big-endian
    float    scale;       // big-endian
    float    zero_point;  // big-endian
    uint32_t data_size;   // big-endian
};
#pragma pack(pop)

struct N730File {
    FILE*    fp;
    int64_t  data_start_offset;
};

N730_API int64_t n730_open(const char* path) {
    N730File* state = (N730File*)malloc(sizeof(N730File));
    if (!state) return (int64_t)N730_ERR_ALLOC;

    state->fp = fopen(path, "rb");
    if (!state->fp) {
        free(state);
        return (int64_t)N730_ERR_FILE;
    }

    // Validate magic
    uint8_t magic[8];
    if (fread(magic, 1, 8, state->fp) != 8 || memcmp(magic, FILE_MAGIC, 8) != 0) {
        fclose(state->fp);
        free(state);
        return (int64_t)N730_ERR_BAD_MAGIC;
    }

    state->data_start_offset = 0;
    return (int64_t)(uintptr_t)state;
}

N730_API void n730_close(int64_t handle) {
    N730File* state = (N730File*)(uintptr_t)handle;
    if (state) {
        if (state->fp) fclose(state->fp);
        free(state);
    }
}

N730_API int32_t n730_read_layer(
    int64_t  handle,
    int64_t  file_offset,    // absolute byte offset from seek table
    float*   out_buffer,     // caller-allocated, must be rows*cols floats
    int32_t* out_rows,
    int32_t* out_cols,
    int32_t* out_prec_id
) {
    if (!handle || !out_buffer) return N730_ERR_NULL;
    N730File* state = (N730File*)(uintptr_t)handle;

    if (fseek(state->fp, (long)file_offset, SEEK_SET) != 0)
        return N730_ERR_FILE;

    // Read and validate layer magic
    uint8_t lmagic[4];
    if (fread(lmagic, 1, 4, state->fp) != 4 || memcmp(lmagic, LAYER_MAGIC, 4) != 0)
        return N730_ERR_BAD_LAYER;

    // Read header fields (all big-endian)
    uint32_t layer_idx = read_be32(state->fp);
    uint8_t  prec_id   = 0;
    fread(&prec_id, 1, 1, state->fp);
    uint32_t rows      = read_be32(state->fp);
    uint32_t cols      = read_be32(state->fp);
    float    scale     = read_be_float(state->fp);
    float    zero_pt   = read_be_float(state->fp);
    uint32_t data_size = read_be32(state->fp);

    if (out_rows)    *out_rows    = (int32_t)rows;
    if (out_cols)    *out_cols    = (int32_t)cols;
    if (out_prec_id) *out_prec_id = (int32_t)prec_id;

    // Read raw quantized bytes
    uint8_t* raw = (uint8_t*)malloc(data_size);
    if (!raw) return N730_ERR_ALLOC;

    if (fread(raw, 1, data_size, state->fp) != data_size) {
        free(raw);
        return N730_ERR_FILE;
    }

    // Dequantize into caller buffer
    int32_t n_elements = (int32_t)(rows * cols);
    switch (prec_id) {
        case PREC_INT8: dequant_int8(raw, out_buffer, n_elements, scale, zero_pt); break;
        case PREC_INT4: dequant_int4(raw, out_buffer, n_elements, scale, zero_pt); break;
        case PREC_INT2: dequant_int2(raw, out_buffer, n_elements, scale, zero_pt); break;
        case PREC_FP16: dequant_fp16(raw, out_buffer, n_elements);                 break;
        default:
            free(raw);
            return N730_ERR_BAD_PREC;
    }

    free(raw);
    return N730_OK;
}

N730_API int32_t n730_probe_layer_size(
    int64_t handle,
    int64_t file_offset
) {
    if (!handle) return N730_ERR_NULL;
    N730File* state = (N730File*)(uintptr_t)handle;

    if (fseek(state->fp, (long)file_offset, SEEK_SET) != 0)
        return N730_ERR_FILE;

    uint8_t lmagic[4];
    if (fread(lmagic, 1, 4, state->fp) != 4 || memcmp(lmagic, LAYER_MAGIC, 4) != 0)
        return N730_ERR_BAD_LAYER;

    read_be32(state->fp);      // layer_idx
    fread(NULL, 1, 1, state->fp); // prec_id — fread with NULL is UB, use dummy
    uint8_t dummy; fseek(state->fp, (long)file_offset + 4 + 4 + 1, SEEK_SET);
    uint32_t rows = read_be32(state->fp);
    uint32_t cols = read_be32(state->fp);
    return (int32_t)(rows * cols);
}

N730_API int32_t n730_layer_elements(int64_t handle, int64_t file_offset) {
    if (!handle) return N730_ERR_NULL;
    N730File* state = (N730File*)(uintptr_t)handle;

    // Seek past: magic(4) + layer_idx(4) + prec_id(1) = 9 bytes
    if (fseek(state->fp, (long)(file_offset + 4 + 4 + 1), SEEK_SET) != 0)
        return N730_ERR_FILE;

    uint32_t rows = read_be32(state->fp);
    uint32_t cols = read_be32(state->fp);
    return (int32_t)(rows * cols);
}

N730_API const char* n730_version() {
    return "N730Core version whatever";
}