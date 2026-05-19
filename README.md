# Planetary Gearbox Fault Diagnosis on ESP32 with TensorFlow Lite Micro

End-to-end pipeline for fault diagnosis in planetary gearboxes using accelerometer vibration data, a compact MLP trained in TensorFlow, and on-device inference with TensorFlow Lite Micro on the ESP32-S3.

The project covers the full workflow:
1. Signal segmentation and kinematic-frequency–guided feature extraction
2. Training and INT8 quantization of a Keras model
3. Code generation for embedded deployment (model, scaler, validation vectors)
4. Bit-exact validation of the ported model on the microcontroller
5. Real-time FFT and feature extraction in pure C on the ESP32

---

## Fault Classes

| ID | Class                | Description                                              |
|----|----------------------|----------------------------------------------------------|
| 0  | Normal               | Healthy gearbox                                          |
| 1  | Surface Wear         | Distributed wear across tooth surfaces                   |
| 2  | Cracked Tooth        | Localized fatigue crack at the tooth root                |
| 3  | Chipped Tooth        | Partial loss of tooth material                           |
| 4  | Missing Tooth        | Complete absence of a tooth                              |

---

## Repository Layout

```
.
├── main/                              ESP-IDF firmware
│   ├── main.cc                        FreeRTOS entry point
│   ├── main_functions.cc              Inference loop + validation
│   ├── features.{cc,h}                Pure-C FFT + feature extraction
│   ├── model.{cc,h}                   Quantized TFLite model (g_model[])
│   ├── scaler_params.h                StandardScaler params + kinematic constants
│   ├── test_vectors.h                 20 INT8 vectors for model-level validation
│   ├── test_signals.{bin,h}           5 raw segments for feature-level validation
│   ├── CMakeLists.txt                 Component registration + EMBED_FILES
│   └── idf_component.yml              Managed components (esp-tflite-micro)
├── notebook/
│   ├── gearbox_fault_esp32_tflite.ipynb   End-to-end training notebook
│   ├── predictive_maintenance_fft.ipynb   Exploratory analysis (XAI/SHAP/LLM)
│   └── data/                              Vibration dataset (.npy)
├── CMakeLists.txt                     Top-level ESP-IDF project
├── sdkconfig.defaults                 Default SDK configuration
└── README.md
```

---

## Signal Pipeline

Both training (Python) and inference (C) follow the same processing chain to guarantee numerical parity:

```
raw vibration (X, Y, Z @ 10 kHz)
        │
        ▼  segmentation
1-segment window = 8192 samples (~0.82 s, power of two for radix-2 FFT)
        │
        ▼  per axis: time-domain stats
        │    RMS, peak value, kurtosis, crest factor          (4 features)
        │
        ▼  per axis: FFT (Hann window, single-sided amplitude)
        │    fm_real peak in Fm2 ± 15 Hz                       (1 feature)
        │    sidebands at fm_real ± Fcsd2, fm_real ± Fcsl2     (4 features)
        │    energy in harmonics 1..5 of fm_real               (5 features)
        │    fm1_real peak in Fm1 ± 15 Hz                      (1 feature)
        │    sidebands at fm1_real + Fcsd1, fm1_real + Fcsl1   (2 features)
        │    energy in harmonics 1..5 of fm1_real              (5 features)
        │    = 22 features per axis × 3 axes = 66 features
        │
        ▼  StandardScaler normalization
        │
        ▼  INT8 quantization (input scale/zero-point from TFLite)
        │
        ▼  TFLite Micro inference: MLP 66 → 64 → 32 → 5
        │    Ops: FullyConnected (×3), Softmax (×1)
        │
        ▼
argmax → predicted class + probabilities
```

### Kinematic Frequencies

Computed from the gearbox geometry at 1500 RPM input speed and used to steer the spectral feature extraction toward physically meaningful bands:

| Symbol     | Value (Hz) | Meaning                                        |
|------------|-----------:|------------------------------------------------|
| Fm1        | 416.67     | 1st-stage gear mesh frequency                  |
| Fm2        | 91.15      | 2nd-stage gear mesh frequency                  |
| Fcsd1      | 20.83      | 1st-stage distributed-fault characteristic     |
| Fcsd2      | 3.26       | 2nd-stage distributed-fault characteristic     |
| Fcsl1      | 62.50      | 1st-stage local-fault characteristic           |
| Fcsl2      | 13.02      | 2nd-stage local-fault characteristic           |

---

## Training Workflow (Python / Colab)

The full pipeline lives in `notebook/gearbox_fault_esp32_tflite.ipynb` and is structured into 14 sections covering environment setup, data loading, segmentation, kinematic parameter computation, feature extraction, MLP training, TFLite conversion, INT8 quantization, validation, and code generation.

### Running the Notebook

1. Open `gearbox_fault_esp32_tflite.ipynb` in Google Colab (a free GPU runtime is sufficient).
2. The data-loading cell auto-detects Colab and downloads the dataset folder via `gdown`. For local execution, place the `.npy` files in `notebook/data/`.
3. Run all cells. Training completes in roughly 30 seconds on a free Colab GPU.

### Generated Artifacts

After execution the notebook writes the following files to `MODEL_DIR` (`/content/gearbox_models` in Colab):

| File                  | Description                                                       |
|-----------------------|-------------------------------------------------------------------|
| `gearbox_float.tflite`| Float32 TFLite model (reference)                                  |
| `gearbox_int8.tflite` | INT8-quantized model deployed to the ESP32                        |
| `model.cc`            | C byte array of `gearbox_int8.tflite` (compiled into firmware)    |
| `scaler_params.h`     | `SCALER_MEAN`, `SCALER_SCALE`, kinematic frequencies, quant params|
| `test_vectors.h`      | 20 pre-quantized samples for model-port validation                |
| `test_signals.bin`    | Raw signals + expected features for feature-extraction validation |
| `test_signals.h`      | Struct layout + extern declarations for the embedded binary blob  |

Copy these files into `main/` to update the firmware.

---

## Firmware (ESP-IDF)

### Requirements

- ESP-IDF v5.0 or newer (tested with v6.0.1)
- Target: ESP32-S3 (other ESP32 variants supported with sufficient RAM/flash)
- Managed components (installed automatically via the IDF Component Manager):
  - `espressif/esp-tflite-micro`

### Build and Flash

```bash
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

### Memory Footprint

| Section                 | Size      |
|-------------------------|-----------|
| INT8 model (`g_model`)  | ~7 KB     |
| Tensor arena            | 4 KB      |
| FFT working buffers     | ~112 KB   |
| `test_signals.bin`      | ~480 KB (validation only — can be removed for production) |

---

## Validation Strategy

The project implements three independent validation layers, each verifying a different part of the pipeline.

### Layer 1 — Model-Port Validation (bit-exact)

The notebook exports 20 INT8 inputs and their expected INT8 outputs from the Python TFLite interpreter into `test_vectors.h`. The firmware feeds each input to TFLite Micro and compares the output byte-by-byte. A `bit_exact = OK` result on every vector confirms the model was correctly serialized, the schema version matches, the op resolver includes every required kernel, and the tensor arena is large enough.

Expected log output:

```
=== Modelo carregado ===
Input  : dtype=9  bytes=66
Output : dtype=9  bytes=5
Arena usada: 1756 / 4096 bytes

[#0] true=0 pred=0 expected=0  classe=OK  bit-exact=OK
[#1] true=0 pred=0 expected=0  classe=OK  bit-exact=OK
...
Bit-exato      : 20 / 20
Classe correta : 20 / 20
Veredito       : PASS (bit-exato)
```

### Layer 2 — Feature-Extraction Validation

The notebook exports 5 raw segments (one per class) plus the 66-feature vector computed by Python into `test_signals.bin`. The firmware reads each segment from flash, runs the C feature extractor, and compares the output against the expected vector. Typical results:

| Metric        | Value             | Interpretation                            |
|---------------|-------------------|-------------------------------------------|
| `max_abs`     | < 3 × 10⁻⁵        | Float32 noise — well within tolerance     |
| `max_rel`     | < 0.01 %          | No systematic drift                       |
| Worst feature | always `kurtosis` | Expected: 4th moment amplifies round-off  |

### Layer 3 — End-to-End Inference

With Layers 1 and 2 green, the firmware chains them together: raw signal → C feature extraction → normalization → INT8 quantization → TFLite Micro inference. Each inference completes in ~50–60 ms (FFT + features) plus ~5–10 ms (model), well within the 5-second period used during demonstration.

Sample output:

```
[#3] true=3 (Dente Lascado)  extracao=50 ms
  Comparacao features (C vs Python):
    max_abs : 0.000020  (feature [46])
    -> kurtosis_Z   C=3.338190  Py=3.338210  delta=-0.000020
  Features calculadas em C (unidades reais):
    feature                  X            Y            Z
    rms              0.08417      0.18738      0.24169
    peak_value       0.35507      0.86893      1.01224
    ...
    fm1_real(Hz)   417.48047    417.48047    417.48047
  Inferencia=10 ms  pred=3 (Dente Lascado)  prob=0.961  classe=OK
  Probabilidades:
    [0] Normal                 0.0000
    [1] Desgaste Superficial   0.0000
    [2] Dente Trincado         0.0000
    [3] Dente Lascado          0.9609
    [4] Dente Ausente          0.0000
```

---

## FFT Implementation Notes

The firmware uses a pure-C iterative radix-2 Cooley-Tukey FFT (`fft_radix2` in `features.cc`) instead of `esp-dsp`. The decision was driven by initialization conflicts: when `esp-tflite-micro` is linked, the `dsps_fft2r_init_fc32` global state may already be claimed by `esp-nn` or another internal user, causing our own init call to return `ESP_ERR_DSP_REINITIALIZED` (0x70003) with no reliable way to reclaim the tables.

The pure-C implementation is self-contained, has no global state, and runs an 8192-point FFT in roughly 50 ms on the ESP32-S3 at 240 MHz — far within the inference budget for this application. Switching to `esp-dsp` for a roughly 10× speed-up is straightforward once the init-order issue is solved upstream.

---

## Configuration Flags

In `main/main_functions.cc`:

| Macro                       | Default | Purpose                                                                        |
|-----------------------------|---------|--------------------------------------------------------------------------------|
| `INFERENCIA_VIA_FEATURES_C` | `1`     | `1` = features extracted in C; `0` = use pre-quantized vectors from `test_vectors.h` |
| `MOSTRAR_FEATURES`          | `1`     | `1` = print the 22×3 feature table per inference                               |
| `kTensorArenaSize`          | `4096`  | Bytes reserved for the tensor arena                                            |

In `main/main.cc`:

| Macro                     | Default | Purpose                                  |
|---------------------------|---------|------------------------------------------|
| `INFERENCIA_PERIODO_MS`   | `5000`  | Delay between inferences (simulated period) |

---

## Dataset

The vibration data used in this project comes from the **ICPHM'23 Benchmark Vibration Dataset Applicable in Machine Learning for Systems' Health Monitoring**, which provides triaxial accelerometer recordings (X, Y, Z) of a planetary gearbox operated under five health conditions at 10 kHz sampling rate.

> S. M. Tayyab, S. Asghar, A. Chatterton and A. Thomson, *"ICPHM'23 Benchmark Vibration Dataset Applicable in Machine Learning for Systems' Health Monitoring"*, IEEE International Conference on Prognostics and Health Management (ICPHM), 2023.
> [https://ieeexplore.ieee.org/document/10626846](https://ieeexplore.ieee.org/document/10626846)

---

## License

Source files derived from the TensorFlow Lite Micro Hello World example retain the Apache 2.0 license headers from upstream. All original work in this repository is released under the same Apache 2.0 license.
