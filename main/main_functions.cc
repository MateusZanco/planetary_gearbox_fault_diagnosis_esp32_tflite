// Programa de validação bit-exata do modelo TFLite Micro no ESP32.
// Executa todos os vetores de g_test_vectors[] e compara com a saída do
// interpretador Python (gerada no notebook de treinamento).

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "main_functions.h"
#include "model.h"
#include "test_vectors.h"

#include <cstring>

namespace {
const tflite::Model*       model       = nullptr;
tflite::MicroInterpreter*  interpreter = nullptr;
TfLiteTensor*              input       = nullptr;
TfLiteTensor*              output      = nullptr;

// 8 KB — folgado para uma MLP 57→64→32→5 int8.
// Se AllocateTensors() falhar, aumente; se sobrar muito, reduza (veja log).
constexpr int kTensorArenaSize = 8 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

void setup() {
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Schema mismatch: model=%d expected=%d",
                static_cast<int>(model->version()), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<1> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    MicroPrintf("AddFullyConnected failed");
    return;
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

  MicroPrintf("=== Modelo carregado ===");
  MicroPrintf("Input  : dtype=%d  bytes=%d", input->type,  (int)input->bytes);
  MicroPrintf("Output : dtype=%d  bytes=%d", output->type, (int)output->bytes);
  MicroPrintf("Arena usada: %d / %d bytes",
              (int)interpreter->arena_used_bytes(), kTensorArenaSize);
  MicroPrintf("Input  quant : scale=%.6f  zp=%d",
              input->params.scale,  input->params.zero_point);
  MicroPrintf("Output quant : scale=%.6f  zp=%d",
              output->params.scale, output->params.zero_point);

  if ((int)input->bytes != TEST_INPUT_SIZE) {
    MicroPrintf("ERRO: input bytes (%d) != TEST_INPUT_SIZE (%d)",
                (int)input->bytes, TEST_INPUT_SIZE);
  }
  if ((int)output->bytes != TEST_OUTPUT_SIZE) {
    MicroPrintf("ERRO: output bytes (%d) != TEST_OUTPUT_SIZE (%d)",
                (int)output->bytes, TEST_OUTPUT_SIZE);
  }
}

void loop() {
  if (interpreter == nullptr) {
    MicroPrintf("Interpreter nao inicializado.");
    return;
  }

  int bit_exact_ok = 0;
  int classe_ok   = 0;

  MicroPrintf("\n=== Validacao com %d vetores de teste ===", NUM_TEST_VECTORS);

  for (int i = 0; i < NUM_TEST_VECTORS; ++i) {
    const TestVector& tv = g_test_vectors[i];

    // 1) Copia input pre-quantizado
    std::memcpy(input->data.int8, tv.input, TEST_INPUT_SIZE);

    // 2) Inferencia
    if (interpreter->Invoke() != kTfLiteOk) {
      MicroPrintf("[%2d] Invoke FAILED", i);
      continue;
    }

    // 3) Compara saida byte-a-byte com a do Python
    bool bit_exact = (std::memcmp(output->data.int8, tv.expected_output,
                                  TEST_OUTPUT_SIZE) == 0);

    // 4) argmax
    int pred = 0;
    for (int k = 1; k < TEST_OUTPUT_SIZE; ++k) {
      if (output->data.int8[k] > output->data.int8[pred]) pred = k;
    }

    bool ok_class = (pred == tv.expected_class);
    if (bit_exact) bit_exact_ok++;
    if (ok_class)  classe_ok++;

    MicroPrintf("[%2d] true=%d pred=%d expected=%d  classe=%s  bit-exact=%s",
                i, tv.true_class, pred, tv.expected_class,
                ok_class  ? "OK" : "FAIL",
                bit_exact ? "OK" : "diff");
  }

  MicroPrintf("\n=== Resumo ===");
  MicroPrintf("Bit-exato      : %d / %d", bit_exact_ok, NUM_TEST_VECTORS);
  MicroPrintf("Classe correta : %d / %d", classe_ok,    NUM_TEST_VECTORS);

  const char* veredito;
  if (bit_exact_ok == NUM_TEST_VECTORS)      veredito = "PASS (bit-exato)";
  else if (classe_ok == NUM_TEST_VECTORS)    veredito = "OK  (mesma classe; diferencas de arredondamento)";
  else                                       veredito = "FAIL — verifique kTensorArenaSize, schema e op resolver";

  MicroPrintf("Veredito       : %s", veredito);
}
