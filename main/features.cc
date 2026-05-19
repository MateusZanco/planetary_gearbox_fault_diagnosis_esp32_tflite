// Implementacao do extrator de features.
// FFT implementada em C puro (Cooley-Tukey radix-2 iterativa) para evitar
// conflito de inicializacao com esp-dsp (que pode ser inicializado por
// outros componentes como esp-nn/TFLM). Sem estado global compartilhado.

#include "features.h"
#include "scaler_params.h"

#include "esp_log.h"

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

static const char* TAG = "features";

#define N_SAMPLES   8192   // = AMOSTRAS_POR_SEG
#define N_BINS      (N_SAMPLES / 2 + 1)
#define N_HARM_FM    5
#define N_HARM_FM1   5

// Buffers globais (em DRAM).
//   fft_buf  : complex interleaved (Re, Im) — 8192 complex = 65536 bytes
//   amp_buf  : magnitudes single-sided     —  4097 floats = 16388 bytes
//   hann     : janela de Hann pre-calculada — 8192 floats = 32768 bytes
__attribute__((aligned(16))) static float fft_buf[N_SAMPLES * 2];
__attribute__((aligned(16))) static float amp_buf[N_BINS];
__attribute__((aligned(16))) static float hann_window[N_SAMPLES];

static float hann_sum = 0.0f;
static float df_hz    = 0.0f;   // resolucao espectral = fs / N
static bool  init_ok  = false;

// ---------- FFT pura em C (Cooley-Tukey radix-2 iterativa, in-place) ----------
//
// data: array de 2*N floats, interleaved (Re, Im).
// N: potencia de 2 (8192 no nosso caso).
//
// Saida: FFT no mesmo buffer, ordenada por bin (0..N-1).
static void fft_radix2(float* data, int N) {
  // 1) Bit-reverse permutation (in-place)
  int j = 0;
  for (int i = 1; i < N; ++i) {
    int bit = N >> 1;
    while (j & bit) { j ^= bit; bit >>= 1; }
    j ^= bit;
    if (i < j) {
      float tmp;
      tmp = data[2*i  ]; data[2*i  ] = data[2*j  ]; data[2*j  ] = tmp;
      tmp = data[2*i+1]; data[2*i+1] = data[2*j+1]; data[2*j+1] = tmp;
    }
  }

  // 2) Butterfly stages
  for (int len = 2; len <= N; len <<= 1) {
    float angle   = -2.0f * M_PI / (float)len;
    float wlen_re = cosf(angle);
    float wlen_im = sinf(angle);
    int   half    = len >> 1;
    for (int i = 0; i < N; i += len) {
      float w_re = 1.0f, w_im = 0.0f;
      for (int k = 0; k < half; ++k) {
        int ia = (i + k) * 2;
        int ib = (i + k + half) * 2;
        float u_re = data[ia];
        float u_im = data[ia + 1];
        float v_re = data[ib] * w_re - data[ib + 1] * w_im;
        float v_im = data[ib] * w_im + data[ib + 1] * w_re;
        data[ia]     = u_re + v_re;
        data[ia + 1] = u_im + v_im;
        data[ib]     = u_re - v_re;
        data[ib + 1] = u_im - v_im;
        // w *= wlen
        float nw_re = w_re * wlen_re - w_im * wlen_im;
        float nw_im = w_re * wlen_im + w_im * wlen_re;
        w_re = nw_re;
        w_im = nw_im;
      }
    }
  }
}

// ---------- Inicializacao (Hann window) ----------

bool features_init(void) {
  if (init_ok) return true;

  // Janela de Hann: 0.5 * (1 - cos(2*pi*i / (N-1)))
  hann_sum = 0.0f;
  for (int i = 0; i < N_SAMPLES; ++i) {
    hann_window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * (float)i / (float)(N_SAMPLES - 1)));
    hann_sum += hann_window[i];
  }

  df_hz   = FS_HZ / (float)N_SAMPLES;   // = 10000/8192 ~ 1.2207 Hz
  init_ok = true;

  ESP_LOGI(TAG, "features_init OK  N=%d  df=%.4f Hz  hann_sum=%.4f",
           N_SAMPLES, df_hz, hann_sum);
  return true;
}

// ---------- Estatisticas no dominio do tempo ----------

static float compute_rms(const float* x, int n) {
  float sum_sq = 0.0f;
  for (int i = 0; i < n; ++i) sum_sq += x[i] * x[i];
  return sqrtf(sum_sq / (float)n);
}

static float compute_peak(const float* x, int n) {
  float pk = 0.0f;
  for (int i = 0; i < n; ++i) {
    float a = fabsf(x[i]);
    if (a > pk) pk = a;
  }
  return pk;
}

static float compute_mean(const float* x, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; ++i) s += x[i];
  return s / (float)n;
}

static float compute_kurtosis(const float* x, int n) {
  float mean = compute_mean(x, n);
  float m2 = 0.0f, m4 = 0.0f;
  for (int i = 0; i < n; ++i) {
    float d  = x[i] - mean;
    float d2 = d * d;
    m2 += d2;
    m4 += d2 * d2;
  }
  m2 /= (float)n;
  m4 /= (float)n;
  if (m2 <= 0.0f) return NAN;
  return m4 / (m2 * m2);
}

// ---------- Espectro de amplitude ----------

static void compute_amplitude_spectrum(const float* signal) {
  float mean = compute_mean(signal, N_SAMPLES);

  // Centraliza, janela e empacota como complexo (Re, 0)
  for (int i = 0; i < N_SAMPLES; ++i) {
    fft_buf[2*i]     = (signal[i] - mean) * hann_window[i];
    fft_buf[2*i + 1] = 0.0f;
  }

  fft_radix2(fft_buf, N_SAMPLES);

  // Magnitude single-sided: |X(k)| * 2 / sum(hann)
  float scale = 2.0f / hann_sum;
  for (int k = 0; k < N_BINS; ++k) {
    float re = fft_buf[2*k];
    float im = fft_buf[2*k + 1];
    amp_buf[k] = sqrtf(re * re + im * im) * scale;
  }
  amp_buf[0] = 0.0f;  // remove DC (igual ao Python)
}

// ---------- Helpers de busca espectral ----------

static float find_peak_freq_in_band(float target_hz, float halfwidth_hz) {
  float f_lo = target_hz - halfwidth_hz;
  float f_hi = target_hz + halfwidth_hz;
  int idx_peak = -1;
  for (int i = 1; i < N_BINS; ++i) {
    float f = i * df_hz;
    if (f < f_lo) continue;
    if (f > f_hi) break;
    if (idx_peak < 0 || amp_buf[i] > amp_buf[idx_peak]) idx_peak = i;
  }
  if (idx_peak < 0) return NAN;
  return idx_peak * df_hz;
}

static float amp_at_target_freq(float freq_alvo) {
  if (isnan(freq_alvo)) return NAN;
  int idx = (int)roundf(freq_alvo / df_hz);
  if (idx < 0 || idx >= N_BINS) return NAN;
  return amp_buf[idx];
}

static float band_energy(float f_lo, float f_hi) {
  bool tem_algum = false;
  float sum = 0.0f;
  for (int i = 1; i < N_BINS; ++i) {
    float f = i * df_hz;
    if (f >= f_lo && f <= f_hi) {
      sum += amp_buf[i] * amp_buf[i];
      tem_algum = true;
    }
  }
  return tem_algum ? sum : NAN;
}

// ---------- Extracao por eixo ----------

static void extract_features_axis(const float* signal, float* feat) {
  // ----- Tempo -----
  float rms  = compute_rms(signal, N_SAMPLES);
  float peak = compute_peak(signal, N_SAMPLES);
  feat[ 0] = rms;
  feat[ 1] = peak;
  feat[ 2] = compute_kurtosis(signal, N_SAMPLES);
  feat[ 3] = (rms > 0.0f) ? (peak / rms) : NAN;

  // ----- Espectro -----
  compute_amplitude_spectrum(signal);

  // 2o estagio
  float fm_real = find_peak_freq_in_band(FM2_TEORICO_HZ, LARGURA_BUSCA_FM_HZ);
  feat[ 4] = fm_real;
  feat[ 5] = amp_at_target_freq(fm_real - FCSD2_HZ);
  feat[ 6] = amp_at_target_freq(fm_real + FCSD2_HZ);
  feat[ 7] = amp_at_target_freq(fm_real - FCSL2_HZ);
  feat[ 8] = amp_at_target_freq(fm_real + FCSL2_HZ);

  for (int h = 1; h <= N_HARM_FM; ++h) {
    float f_alvo = (float)h * fm_real;
    feat[8 + h] = band_energy(f_alvo - LARGURA_BANDA_HARM_HZ,
                              f_alvo + LARGURA_BANDA_HARM_HZ);
  }

  // 1o estagio
  float fm1_real = find_peak_freq_in_band(FM1_TEORICO_HZ, LARGURA_BUSCA_FM_HZ);
  feat[14] = fm1_real;
  feat[15] = amp_at_target_freq(fm1_real + FCSD1_HZ);
  feat[16] = amp_at_target_freq(fm1_real + FCSL1_HZ);

  for (int h = 1; h <= N_HARM_FM1; ++h) {
    float f_alvo = (float)h * fm1_real;
    feat[16 + h] = band_energy(f_alvo - LARGURA_BANDA_HARM_HZ,
                               f_alvo + LARGURA_BANDA_HARM_HZ);
  }
}

void features_extract(const float* sx, const float* sy, const float* sz,
                      float* out) {
  if (!init_ok && !features_init()) return;
  extract_features_axis(sx, out + 0 * FEATURES_PER_AXIS);
  extract_features_axis(sy, out + 1 * FEATURES_PER_AXIS);
  extract_features_axis(sz, out + 2 * FEATURES_PER_AXIS);
}
