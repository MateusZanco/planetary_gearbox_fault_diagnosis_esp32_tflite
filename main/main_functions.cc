// Programa de inferencia periodica usando os vetores de teste como
// "simulacao" de sinal real. A cada chamada de loop() o ESP32 roda UMA
// inferencia e imprime predicao + features dequantizadas.

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "main_functions.h"
#include "model.h"
#include "test_vectors.h"
#include "scaler_params.h"

#include <cstring>
#include <cstdio>

// 1 = imprime as 57 features dequantizadas a cada inferencia
// 0 = imprime apenas a linha de predicao (compacto)
#define MOSTRAR_FEATURES   1

namespace {
const tflite::Model*       model       = nullptr;
tflite::MicroInterpreter*  interpreter = nullptr;
TfLiteTensor*              input       = nullptr;
TfLiteTensor*              output      = nullptr;

constexpr int kTensorArenaSize = 4 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const char* kNomesClasses[] = {
    "Normal", "Desgaste Superficial", "Dente Trincado",
    "Dente Lascado", "Dente Ausente"
};

// Nomes curtos das 19 features por eixo, na ordem em que foram geradas.
const char* kNomesFeatPorEixo[19] = {
    "rms        ", "peak_value ", "kurtosis   ", "crest_fact ",
    "fm_real(Hz)",
    "amp_fcsd_e ", "amp_fcsd_d ", "amp_fcsl_e ", "amp_fcsl_d ",
    "amp_fm_h1  ", "amp_fm_h2  ", "amp_fm_h3  ", "amp_fm_h4  ",
    "amp_fm_h5  ", "amp_fm_h6  ", "amp_fm_h7  ",
    "fm1_real(Hz)", "amp_fcsd1_d", "amp_fcsl1_d"
};

int g_indice_vetor  = 0;
int g_total_rodados = 0;
int g_classe_ok     = 0;
}  // namespace

// Dequantiza um valor int8 e reverte a normalizacao do StandardScaler:
//   real = ((q - input_zp) * input_scale) * scaler_scale + scaler_mean
static inline float dequantizar_e_desnormalizar(int8_t q, int feature_idx) {
  float normalizado = (q - INPUT_ZERO_POINT) * INPUT_SCALE;
  return normalizado * SCALER_SCALE[feature_idx] + SCALER_MEAN[feature_idx];
}

static void imprimir_features(const int8_t* input_q) {
  MicroPrintf("  Features (unidades reais):");
  MicroPrintf("    %-13s %12s %12s %12s", "feature", "X", "Y", "Z");
  for (int i = 0; i < 19; ++i) {
    float vx = dequantizar_e_desnormalizar(input_q[i +  0], i +  0);
    float vy = dequantizar_e_desnormalizar(input_q[i + 19], i + 19);
    float vz = dequantizar_e_desnormalizar(input_q[i + 38], i + 38);
    MicroPrintf("    %s %12.5f %12.5f %12.5f",
                kNomesFeatPorEixo[i], vx, vy, vz);
  }
}

static void imprimir_probabilidades(const int8_t* out_q) {
  char buf[160];
  int off = 0;
  off += snprintf(buf + off, sizeof(buf) - off, "  Probs:");
  for (int k = 0; k < TEST_OUTPUT_SIZE; ++k) {
    float p = (out_q[k] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
    off += snprintf(buf + off, sizeof(buf) - off, "  %d=%.3f", k, p);
  }
  MicroPrintf("%s", buf);
}

void setup() {
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Schema mismatch: model=%d expected=%d",
                static_cast<int>(model->version()), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<2> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    MicroPrintf("AddFullyConnected failed"); return;
  }
  if (resolver.AddSoftmax() != kTfLiteOk) {
    MicroPrintf("AddSoftmax failed"); return;
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed — aumente kTensorArenaSize");
    return;
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  MicroPrintf("=== Inferencia periodica iniciada ===");
  MicroPrintf("Modelo : input int8[%d]  output int8[%d]",
              (int)input->bytes, (int)output->bytes);
  MicroPrintf("Arena  : %d / %d bytes",
              (int)interpreter->arena_used_bytes(), kTensorArenaSize);
  MicroPrintf("Vetores simulados: %d", NUM_TEST_VECTORS);
  MicroPrintf("Features visiveis: %s\n",
              MOSTRAR_FEATURES ? "SIM" : "NAO");
}

void loop() {
  if (interpreter == nullptr || input == nullptr || output == nullptr) {
    MicroPrintf("Interpreter nao inicializado.");
    return;
  }

  const TestVector& tv = g_test_vectors[g_indice_vetor];

  // 1) Carrega input pre-quantizado (simulando bloco vindo do acelerometro)
  std::memcpy(input->data.int8, tv.input, TEST_INPUT_SIZE);

  // 2) Inferencia
  uint32_t t0 = esp_log_timestamp();
  if (interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("[#%d] Invoke FAILED", g_indice_vetor);
    g_indice_vetor = (g_indice_vetor + 1) % NUM_TEST_VECTORS;
    return;
  }
  uint32_t dt_ms = esp_log_timestamp() - t0;

  // 3) argmax + probabilidade do top1
  int pred = 0;
  for (int k = 1; k < TEST_OUTPUT_SIZE; ++k) {
    if (output->data.int8[k] > output->data.int8[pred]) pred = k;
  }
  float prob_top = (output->data.int8[pred] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;

  bool acertou = (pred == tv.true_class);
  if (acertou) g_classe_ok++;
  g_total_rodados++;

  MicroPrintf("\n[#%d] true=%d (%s)  pred=%d (%s)  prob=%.3f  t=%u ms  acuracia=%d/%d",
              g_indice_vetor,
              tv.true_class, kNomesClasses[tv.true_class],
              pred,          kNomesClasses[pred],
              prob_top, (unsigned)dt_ms,
              g_classe_ok, g_total_rodados);

  imprimir_probabilidades(output->data.int8);

#if MOSTRAR_FEATURES
  imprimir_features(input->data.int8);
#endif

  g_indice_vetor = (g_indice_vetor + 1) % NUM_TEST_VECTORS;
}
