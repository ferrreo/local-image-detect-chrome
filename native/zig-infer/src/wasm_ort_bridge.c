/**
 * Browser ORT sessions linked with libonnxruntime_webassembly.a (emscripten).
 * Zig owns SIMD preprocess exports; this file owns OrtEnv / sessions.
 */
#include "onnxruntime_c_api.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { TP_MAX_SESSIONS = 2 };

typedef struct {
  int used;
  OrtSession* session;
  OrtMemoryInfo* memory_info;
  char* input_name;
  char* output_name;
} TpSlot;

static const OrtApi* g_api = NULL;
static OrtEnv* g_env = NULL;
static TpSlot g_slots[TP_MAX_SESSIONS];

static int ensure_env(void) {
  if (g_api && g_env) return 1;
  const OrtApiBase* base = OrtGetApiBase();
  if (!base) return 0;
  g_api = base->GetApi(ORT_API_VERSION);
  if (!g_api) return 0;
  OrtStatus* st = g_api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "truepixel-wasm", &g_env);
  if (st) {
    g_api->ReleaseStatus(st);
    g_env = NULL;
    return 0;
  }
  return 1;
}

static void free_slot(int i) {
  if (i < 0 || i >= TP_MAX_SESSIONS) return;
  TpSlot* s = &g_slots[i];
  if (!s->used || !g_api) return;
  if (s->session) g_api->ReleaseSession(s->session);
  if (s->memory_info) g_api->ReleaseMemoryInfo(s->memory_info);
  free(s->input_name);
  free(s->output_name);
  memset(s, 0, sizeof(*s));
}

uint32_t tp_has_ort_session(void) { return 1; }

void* tp_malloc(size_t n) { return malloc(n); }

void tp_free(void* ptr, size_t n) {
  (void)n;
  free(ptr);
}

int32_t tp_session_create(const uint8_t* model, size_t model_len, uint32_t graph_opt_disabled) {
  if (!ensure_env() || !model || model_len == 0) return -1;
  int idx = -1;
  for (int i = 0; i < TP_MAX_SESSIONS; i++) {
    if (!g_slots[i].used) {
      idx = i;
      break;
    }
  }
  if (idx < 0) return -1;

  OrtSessionOptions* opts = NULL;
  OrtStatus* st = g_api->CreateSessionOptions(&opts);
  if (st) {
    g_api->ReleaseStatus(st);
    return -1;
  }

  GraphOptimizationLevel level =
      graph_opt_disabled ? ORT_DISABLE_ALL : ORT_ENABLE_ALL;
  st = g_api->SetSessionGraphOptimizationLevel(opts, level);
  if (st) {
    g_api->ReleaseStatus(st);
    g_api->ReleaseSessionOptions(opts);
    return -1;
  }
  st = g_api->SetIntraOpNumThreads(opts, 1);
  if (st) {
    g_api->ReleaseStatus(st);
    g_api->ReleaseSessionOptions(opts);
    return -1;
  }

  OrtSession* session = NULL;
  st = g_api->CreateSessionFromArray(g_env, model, model_len, opts, &session);
  g_api->ReleaseSessionOptions(opts);
  if (st) {
    g_api->ReleaseStatus(st);
    return -1;
  }

  OrtMemoryInfo* memory_info = NULL;
  st = g_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
  if (st) {
    g_api->ReleaseStatus(st);
    g_api->ReleaseSession(session);
    return -1;
  }

  OrtAllocator* allocator = NULL;
  st = g_api->GetAllocatorWithDefaultOptions(&allocator);
  if (st) {
    g_api->ReleaseStatus(st);
    g_api->ReleaseMemoryInfo(memory_info);
    g_api->ReleaseSession(session);
    return -1;
  }

  char* in_name = NULL;
  char* out_name = NULL;
  st = g_api->SessionGetInputName(session, 0, allocator, &in_name);
  if (st) {
    g_api->ReleaseStatus(st);
    g_api->ReleaseMemoryInfo(memory_info);
    g_api->ReleaseSession(session);
    return -1;
  }
  st = g_api->SessionGetOutputName(session, 0, allocator, &out_name);
  if (st) {
    g_api->ReleaseStatus(st);
    g_api->AllocatorFree(allocator, in_name);
    g_api->ReleaseMemoryInfo(memory_info);
    g_api->ReleaseSession(session);
    return -1;
  }

  char* in_copy = strdup(in_name);
  char* out_copy = strdup(out_name);
  g_api->AllocatorFree(allocator, in_name);
  g_api->AllocatorFree(allocator, out_name);
  if (!in_copy || !out_copy) {
    free(in_copy);
    free(out_copy);
    g_api->ReleaseMemoryInfo(memory_info);
    g_api->ReleaseSession(session);
    return -1;
  }

  g_slots[idx].used = 1;
  g_slots[idx].session = session;
  g_slots[idx].memory_info = memory_info;
  g_slots[idx].input_name = in_copy;
  g_slots[idx].output_name = out_copy;
  return idx;
}

void tp_session_free(int32_t id) { free_slot((int)id); }

float tp_session_run(int32_t id, const float* nchw, size_t size, uint32_t ai_label) {
  if (id < 0 || id >= TP_MAX_SESSIONS || !g_api) return -1.f;
  TpSlot* s = &g_slots[id];
  if (!s->used || !s->session || !nchw || size == 0) return -1.f;

  int64_t shape[4] = {1, 3, (int64_t)size, (int64_t)size};
  size_t n = 3 * size * size;
  OrtValue* input_tensor = NULL;
  OrtStatus* st = g_api->CreateTensorWithDataAsOrtValue(
      s->memory_info, (void*)nchw, n * sizeof(float), shape, 4,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
  if (st) {
    g_api->ReleaseStatus(st);
    return -2.f;
  }

  const char* input_names[] = {s->input_name};
  const char* output_names[] = {s->output_name};
  OrtValue* output_tensor = NULL;
  st = g_api->Run(s->session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1,
                  output_names, 1, &output_tensor);
  g_api->ReleaseValue(input_tensor);
  if (st) {
    g_api->ReleaseStatus(st);
    return -3.f;
  }

  float* logits = NULL;
  st = g_api->GetTensorMutableData(output_tensor, (void**)&logits);
  if (st) {
    g_api->ReleaseStatus(st);
    g_api->ReleaseValue(output_tensor);
    return -4.f;
  }

  float a = logits[0];
  float b = logits[1];
  float m = a > b ? a : b;
  float ea = __builtin_expf(a - m);
  float eb = __builtin_expf(b - m);
  float sum = ea + eb;
  float p0 = ea / sum;
  float p1 = eb / sum;
  g_api->ReleaseValue(output_tensor);
  return ai_label == 0 ? p0 : p1;
}
