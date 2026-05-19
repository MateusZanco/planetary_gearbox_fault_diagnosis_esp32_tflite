// Camada 2 — Fase A: valida a extracao de features em C contra os valores
// calculados pelo Python (test_signals.bin embutido em flash).
//
// A cada periodo:
//   1. Pega o proximo sinal de g_test_signals[]
//   2. Roda features_extract() sobre os 3 eixos
//   3. Compara as 66 features com expected_features[] do Python
//   4. Imprime erro maximo absoluto e a feature mais divergente

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "main_functions.h"
#include "model.h"
#include "test_vectors.h"
#include "test_signals.h"
#include "scaler_params.h"
#include "features.h"

#include "esp_log.h"
#include <cstring>
#include <cstdio>
#include <cmath>

#define INFERENCIA_VIA_FEATURES_C  1  // 1 = usa features extraidas em C, 0 = usa test_vectors.h int8
#define MOSTRAR_FEATURES           0

namespace {
const tflite::Model*       model       = nullptr;
tflite::MicroInterpreter*  interpreter = nullptr;
TfLiteTensor*              input       = nullptr;
TfLiteTensor*              output      = nullptr;

constexpr int kTensorArenaSize = 4 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Buffer para as 66 features calculadas em C (3 eixos × 22)
float features_calculadas[3 * FEATURES_PER_AXIS];

const char* kNomesClasses[] = {
    "Normal", "Desgaste Superficial", "Dente Trincado",
    "Dente Lascado", "Dente Ausente"
};

#define FEATURES_POR_EIXO 22
const char* kNomesFeatPorEixo[FEATURES_POR_EIXO] = {
    "rms         ", "peak_value  ", "kurtosis    ", "crest_fact  ",
    "fm_real(Hz) ",
    "amp_fcsd_e  ", "amp_fcsd_d  ", "amp_fcsl_e  ", "amp_fcsl_d  ",
    "amp_fm_h1   ", "amp_fm_h2   ", "amp_fm_h3   ", "amp_fm_h4   ", "amp_fm_h5   ",
    "fm1_real(Hz)", "amp_fcsd1_d ", "amp_fcsl1_d ",
    "amp_fm1_h1  ", "amp_fm1_h2  ", "amp_fm1_h3  ", "amp_fm1_h4  ", "amp_fm1_h5  "
};

int g_indice_sinal  = 0;
int g_total_rodados = 0;
int g_classe_ok     = 0;
}  // namespace

void setup() {
  // 1) Modelo TFLite
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Schema mismatch: model=%d expected=%d",
                static_cast<int>(model->version()), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<2> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk) { MicroPrintf("AddFullyConnected failed"); return; }
  if (resolver.AddSoftmax()        != kTfLiteOk) { MicroPrintf("AddSoftmax failed");        return; }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed — aumente kTensorArenaSize");
    return;
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  // 2) Pipeline de features (FFT + Hann)
  if (!features_init()) {
    MicroPrintf("features_init() falhou — abortando");
    return;
  }

  MicroPrintf("=== Camada 2 Fase A: valida extrator de features em C ===");
  MicroPrintf("Modelo : input int8[%d]  output int8[%d]",
              (int)input->bytes, (int)output->bytes);
  MicroPrintf("Arena  : %d / %d bytes",
              (int)interpreter->arena_used_bytes(), kTensorArenaSize);
  MicroPrintf("Sinais de teste embutidos: %d (cada %d samples × 3 eixos)",
              NUM_TEST_SIGNALS, N_SAMPLES_PER_SIGNAL);
  MicroPrintf("Features esperadas/sinal : %d", N_FEATURES_EXPECTED);
  MicroPrintf("");
}

// Aplica scaler e quantizacao, escrevendo direto em input->data.int8[]
static void normalizar_e_quantizar(const float* features) {
  for (int i = 0; i < N_FEATURES; ++i) {
    float norm = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    int   q    = (int)roundf(norm / INPUT_SCALE) + INPUT_ZERO_POINT;
    if (q < -128) q = -128;
    if (q >  127) q =  127;
    input->data.int8[i] = (int8_t)q;
  }
}

static void comparar_features(const float* calculadas, const float* esperadas) {
  float max_abs_err = 0.0f;
  float max_rel_err = 0.0f;
  int   idx_max_abs = -1;
  int   nan_count   = 0;

  for (int i = 0; i < N_FEATURES_EXPECTED; ++i) {
    float c = calculadas[i];
    float e = esperadas[i];

    if (isnan(c) || isnan(e)) { nan_count++; continue; }

    float abs_err = fabsf(c - e);
    float ref     = fmaxf(fabsf(e), 1e-9f);
    float rel_err = abs_err / ref;

    if (abs_err > max_abs_err) {
      max_abs_err = abs_err;
      idx_max_abs = i;
    }
    if (rel_err > max_rel_err) max_rel_err = rel_err;
  }

  MicroPrintf("  Comparacao features (C vs Python):");
  MicroPrintf("    NaN     : %d", nan_count);
  MicroPrintf("    max_abs : %.6f  (feature [%d])", max_abs_err, idx_max_abs);
  MicroPrintf("    max_rel : %.4f%%", max_rel_err * 100.0f);

  if (idx_max_abs >= 0) {
    int eixo_idx = idx_max_abs / FEATURES_POR_EIXO;
    int feat_idx = idx_max_abs % FEATURES_POR_EIXO;
    const char* eixos[] = {"X", "Y", "Z"};
    MicroPrintf("    -> %s_%s   C=%.6f  Py=%.6f  delta=%+.6f",
                kNomesFeatPorEixo[feat_idx], eixos[eixo_idx],
                calculadas[idx_max_abs], esperadas[idx_max_abs],
                calculadas[idx_max_abs] - esperadas[idx_max_abs]);
  }
}

void loop() {
  if (interpreter == nullptr || input == nullptr) return;

  const TestSignalData* sinais = test_signals_array();
  const TestSignalData& s      = sinais[g_indice_sinal];
  const int true_class         = g_test_signal_classes[g_indice_sinal];

  // 1) Extrai features em C
  uint32_t t_feat0 = esp_log_timestamp();
  features_extract(s.signal_x, s.signal_y, s.signal_z, features_calculadas);
  uint32_t dt_feat = esp_log_timestamp() - t_feat0;

  // 2) Compara com o que o Python calculou
  MicroPrintf("\n[#%d] true=%d (%s)  extracao=%u ms",
              g_indice_sinal, true_class, kNomesClasses[true_class],
              (unsigned)dt_feat);
  comparar_features(features_calculadas, s.expected_features);

  // 3) Normaliza, quantiza e roda inferencia
#if INFERENCIA_VIA_FEATURES_C
  normalizar_e_quantizar(features_calculadas);
#else
  // Modo debug: usa o input int8 ja quantizado do test_vectors.h (mesma classe)
  std::memcpy(input->data.int8,
              g_test_vectors[g_indice_sinal % NUM_TEST_VECTORS].input,
              TEST_INPUT_SIZE);
#endif

  uint32_t t_inf0 = esp_log_timestamp();
  if (interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("  Invoke FAILED");
    g_indice_sinal = (g_indice_sinal + 1) % NUM_TEST_SIGNALS;
    return;
  }
  uint32_t dt_inf = esp_log_timestamp() - t_inf0;

  int pred = 0;
  for (int k = 1; k < TEST_OUTPUT_SIZE; ++k) {
    if (output->data.int8[k] > output->data.int8[pred]) pred = k;
  }
  float prob_top = (output->data.int8[pred] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;

  bool acertou = (pred == true_class);
  if (acertou) g_classe_ok++;
  g_total_rodados++;

  MicroPrintf("  Inferencia=%u ms  pred=%d (%s)  prob=%.3f  classe=%s  acuracia=%d/%d",
              (unsigned)dt_inf, pred, kNomesClasses[pred], prob_top,
              acertou ? "OK" : "FAIL", g_classe_ok, g_total_rodados);

  g_indice_sinal = (g_indice_sinal + 1) % NUM_TEST_SIGNALS;
  if (g_indice_sinal == 0) {
    MicroPrintf("\n--- Ciclo completo: %d/%d acertos ---\n",
                g_classe_ok, g_total_rodados);
    g_classe_ok = 0; g_total_rodados = 0;
  }
}
