    // Auto-gerado pelo notebook de treinamento.
    // 1) Calcule as 66 features na mesma ordem de FEATURE_NAMES.
    // 2) Normalize:  norm[i] = (feat[i] - SCALER_MEAN[i]) / SCALER_SCALE[i]
    // 3) Quantize:   q[i]    = round(norm[i] / INPUT_SCALE + INPUT_ZERO_POINT)  [clip em -128..127]
    // 4) Copie q[] para input->data.int8[].
    // 5) Após Invoke(), output->data.int8[k] -> probabilidade ~= (out_k - OUTPUT_ZERO_POINT)*OUTPUT_SCALE
    #ifndef SCALER_PARAMS_H_
    #define SCALER_PARAMS_H_

    #define N_FEATURES 66
    #define N_CLASSES  5

    // ---- Quantização da entrada (int8) ----
    const float INPUT_SCALE       = 0.03863877f;
    const int   INPUT_ZERO_POINT  = -60;

    // ---- Quantização da saída (int8) ----
    const float OUTPUT_SCALE      = 0.00390625f;
    const int   OUTPUT_ZERO_POINT = -128;

    // ---- Frequências cinemáticas usadas na extração de features ----
    const float FS_HZ             = 10000.0000f;
    const int   AMOSTRAS_POR_SEG  = 8192;
    const float FM1_TEORICO_HZ    = 416.666667f;
    const float FM2_TEORICO_HZ    = 91.145833f;
    const float FCSD1_HZ          = 20.833333f;
    const float FCSD2_HZ          = 3.255208f;
    const float FCSL1_HZ          = 62.500000f;
    const float FCSL2_HZ          = 13.020833f;
    const float LARGURA_BUSCA_FM_HZ      = 15.0000f;
    const float LARGURA_BANDA_HARM_HZ    = 15.0000f;

    // ---- Ordem das features (use exatamente esta ordem ao montar o vetor) ----
    // [ 0] rms_x
// [ 1] peak_value_x
// [ 2] kurtosis_x
// [ 3] crest_factor_x
// [ 4] fm_real_x
// [ 5] amp_fcsd_esq_x
// [ 6] amp_fcsd_dir_x
// [ 7] amp_fcsl_esq_x
// [ 8] amp_fcsl_dir_x
// [ 9] amp_fm_h1_x
// [10] amp_fm_h2_x
// [11] amp_fm_h3_x
// [12] amp_fm_h4_x
// [13] amp_fm_h5_x
// [14] fm1_real_x
// [15] amp_fcsd1_dir_x
// [16] amp_fcsl1_dir_x
// [17] amp_fm1_h1_x
// [18] amp_fm1_h2_x
// [19] amp_fm1_h3_x
// [20] amp_fm1_h4_x
// [21] amp_fm1_h5_x
// [22] rms_y
// [23] peak_value_y
// [24] kurtosis_y
// [25] crest_factor_y
// [26] fm_real_y
// [27] amp_fcsd_esq_y
// [28] amp_fcsd_dir_y
// [29] amp_fcsl_esq_y
// [30] amp_fcsl_dir_y
// [31] amp_fm_h1_y
// [32] amp_fm_h2_y
// [33] amp_fm_h3_y
// [34] amp_fm_h4_y
// [35] amp_fm_h5_y
// [36] fm1_real_y
// [37] amp_fcsd1_dir_y
// [38] amp_fcsl1_dir_y
// [39] amp_fm1_h1_y
// [40] amp_fm1_h2_y
// [41] amp_fm1_h3_y
// [42] amp_fm1_h4_y
// [43] amp_fm1_h5_y
// [44] rms_z
// [45] peak_value_z
// [46] kurtosis_z
// [47] crest_factor_z
// [48] fm_real_z
// [49] amp_fcsd_esq_z
// [50] amp_fcsd_dir_z
// [51] amp_fcsl_esq_z
// [52] amp_fcsl_dir_z
// [53] amp_fm_h1_z
// [54] amp_fm_h2_z
// [55] amp_fm_h3_z
// [56] amp_fm_h4_z
// [57] amp_fm_h5_z
// [58] fm1_real_z
// [59] amp_fcsd1_dir_z
// [60] amp_fcsl1_dir_z
// [61] amp_fm1_h1_z
// [62] amp_fm1_h2_z
// [63] amp_fm1_h3_z
// [64] amp_fm1_h4_z
// [65] amp_fm1_h5_z

    const float SCALER_MEAN[66] = {
    0.08584656f, 0.34948692f, 3.05162001f, 4.06962765f,
    83.94516367f, 0.00213635f, 0.00168671f, 0.00274501f,
    0.00153346f, 0.00015560f, 0.00012643f, 0.00007628f,
    0.00038629f, 0.00190036f, 414.80722908f, 0.00715117f,
    0.00397755f, 0.00287949f, 0.00097670f, 0.00011409f,
    0.00071356f, 0.00014763f, 0.18517532f, 0.80971918f,
    3.35636871f, 4.36708076f, 87.26667827f, 0.00206216f,
    0.00193336f, 0.00224111f, 0.00172142f, 0.00015428f,
    0.00005972f, 0.00009237f, 0.00131870f, 0.00328994f,
    414.62405206f, 0.00989313f, 0.00378819f, 0.00642873f,
    0.00244000f, 0.00445031f, 0.00330377f, 0.00064940f,
    0.23547886f, 1.07392583f, 3.35998117f, 4.55554836f,
    90.81716414f, 0.00206854f, 0.00201082f, 0.00217737f,
    0.00207072f, 0.00016951f, 0.00011907f, 0.00017614f,
    0.00334311f, 0.00439823f, 415.99644866f, 0.01388397f,
    0.00833087f, 0.00953004f, 0.00697228f, 0.00298372f,
    0.00496310f, 0.00108608f
};

    const float SCALER_SCALE[66] = {
    0.00521551f, 0.03960892f, 0.12633402f, 0.37772030f,
    7.14970033f, 0.00116393f, 0.00086753f, 0.00156163f,
    0.00079096f, 0.00006278f, 0.00005198f, 0.00002909f,
    0.00074424f, 0.00100178f, 7.83509092f, 0.00388865f,
    0.00237522f, 0.00073255f, 0.00044190f, 0.00005144f,
    0.00042044f, 0.00009190f, 0.01519355f, 0.10878249f,
    0.16232550f, 0.40577344f, 7.85111772f, 0.00109122f,
    0.00102567f, 0.00122640f, 0.00095632f, 0.00007186f,
    0.00002325f, 0.00003791f, 0.00219804f, 0.00263555f,
    7.86470797f, 0.00578130f, 0.00200883f, 0.00182521f,
    0.00101776f, 0.00198893f, 0.00176647f, 0.00032463f,
    0.01562842f, 0.13841914f, 0.16310449f, 0.45933048f,
    8.73710760f, 0.00104518f, 0.00103617f, 0.00122282f,
    0.00111653f, 0.00005621f, 0.00003925f, 0.00006567f,
    0.00400725f, 0.00326585f, 8.00897393f, 0.00778675f,
    0.00498716f, 0.00282530f, 0.00300738f, 0.00130496f,
    0.00278740f, 0.00066073f
};

    // Saída do modelo (índice = classe):
    // 0: Normal | 1: Desgaste Superficial | 2: Dente Trincado
    // 3: Dente Lascado | 4: Dente Ausente

    #endif  // SCALER_PARAMS_H_
