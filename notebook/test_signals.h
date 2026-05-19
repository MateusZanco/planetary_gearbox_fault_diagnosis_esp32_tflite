// Auto-gerado pelo notebook. test_signals.bin contem os sinais brutos
// (x/y/z, 8192 amostras float32) + vetor de features esperado
// (66 float32) para cada um dos 5 sinais.
#ifndef TEST_SIGNALS_H_
#define TEST_SIGNALS_H_

#include <stdint.h>

#define NUM_TEST_SIGNALS         5
#define N_SAMPLES_PER_SIGNAL     8192
#define N_FEATURES_EXPECTED      66

typedef struct {
  float signal_x[N_SAMPLES_PER_SIGNAL];
  float signal_y[N_SAMPLES_PER_SIGNAL];
  float signal_z[N_SAMPLES_PER_SIGNAL];
  float expected_features[N_FEATURES_EXPECTED];
} TestSignalData;

// Embutido em flash via EMBED_FILES no CMakeLists.txt do main/.
extern const uint8_t _binary_test_signals_bin_start[];
extern const uint8_t _binary_test_signals_bin_end[];

// Helper: cast para array de structs.
static inline const TestSignalData* test_signals_array(void) {
  return (const TestSignalData*) _binary_test_signals_bin_start;
}

// Classe verdadeira de cada sinal (na ordem do .bin)
static const int g_test_signal_classes[NUM_TEST_SIGNALS] = { 0, 1, 2, 3, 4 };

#endif  // TEST_SIGNALS_H_
