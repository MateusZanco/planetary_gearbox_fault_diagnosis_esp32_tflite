// Extracao de features para diagnostico de falhas em caixa planetaria.
// Pipeline: sinal bruto (X/Y/Z, 8192 samples cada) -> Hann window -> FFT (esp-dsp)
//           -> 22 features por eixo (4 tempo + 18 frequencia).
//
// Para validar contra Python: as 66 features geradas aqui devem bater (dentro
// de ~1e-3) com expected_features[] do test_signals.bin.

#ifndef FEATURES_H_
#define FEATURES_H_

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FEATURES_PER_AXIS  22

// Inicializa FFT, janela de Hann e buffers internos. Chamar uma vez no setup().
// Retorna false se a inicializacao falhar.
bool features_init(void);

// Extrai 66 features (3 eixos × 22) a partir dos sinais brutos.
//   signal_{x,y,z} : N_SAMPLES_PER_SIGNAL float32 cada
//   out_features   : array de saida com 3*FEATURES_PER_AXIS = 66 floats
void features_extract(const float* signal_x,
                      const float* signal_y,
                      const float* signal_z,
                      float* out_features);

#ifdef __cplusplus
}
#endif

#endif  // FEATURES_H_
