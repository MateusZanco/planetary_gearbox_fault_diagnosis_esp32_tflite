    // Auto-gerado pelo notebook de treinamento.
    // 1) Calcule as 57 features na mesma ordem de FEATURE_NAMES.
    // 2) Normalize:  norm[i] = (feat[i] - SCALER_MEAN[i]) / SCALER_SCALE[i]
    // 3) Quantize:   q[i]    = round(norm[i] / INPUT_SCALE + INPUT_ZERO_POINT)  [clip em -128..127]
    // 4) Copie q[] para input->data.int8[].
    // 5) Após Invoke(), output->data.int8[k] -> probabilidade ~= (out_k - OUTPUT_ZERO_POINT)*OUTPUT_SCALE
    #ifndef SCALER_PARAMS_H_
    #define SCALER_PARAMS_H_

    #define N_FEATURES 57
    #define N_CLASSES  5

    // ---- Quantização da entrada (int8) ----
    const float INPUT_SCALE       = 0.05156828f;
    const int   INPUT_ZERO_POINT  = -82;

    // ---- Quantização da saída (int8) ----
    const float OUTPUT_SCALE      = 0.00390625f;
    const int   OUTPUT_ZERO_POINT = -128;

    // ---- Frequências cinemáticas usadas na extração de features ----
    const float FS_HZ             = 10000.0000f;
    const int   AMOSTRAS_POR_SEG  = 10000;
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
// [14] amp_fm_h6_x
// [15] amp_fm_h7_x
// [16] fm1_real_x
// [17] amp_fcsd1_dir_x
// [18] amp_fcsl1_dir_x
// [19] rms_y
// [20] peak_value_y
// [21] kurtosis_y
// [22] crest_factor_y
// [23] fm_real_y
// [24] amp_fcsd_esq_y
// [25] amp_fcsd_dir_y
// [26] amp_fcsl_esq_y
// [27] amp_fcsl_dir_y
// [28] amp_fm_h1_y
// [29] amp_fm_h2_y
// [30] amp_fm_h3_y
// [31] amp_fm_h4_y
// [32] amp_fm_h5_y
// [33] amp_fm_h6_y
// [34] amp_fm_h7_y
// [35] fm1_real_y
// [36] amp_fcsd1_dir_y
// [37] amp_fcsl1_dir_y
// [38] rms_z
// [39] peak_value_z
// [40] kurtosis_z
// [41] crest_factor_z
// [42] fm_real_z
// [43] amp_fcsd_esq_z
// [44] amp_fcsd_dir_z
// [45] amp_fcsl_esq_z
// [46] amp_fcsl_dir_z
// [47] amp_fm_h1_z
// [48] amp_fm_h2_z
// [49] amp_fm_h3_z
// [50] amp_fm_h4_z
// [51] amp_fm_h5_z
// [52] amp_fm_h6_z
// [53] amp_fm_h7_z
// [54] fm1_real_z
// [55] amp_fcsd1_dir_z
// [56] amp_fcsl1_dir_z

    const float SCALER_MEAN[57] = {
    0.08577619f, 0.35491985f, 3.05101534f, 4.13635213f,
    83.94563662f, 0.00185423f, 0.00156376f, 0.00250232f,
    0.00136495f, 0.00015722f, 0.00012740f, 0.00007819f,
    0.00037621f, 0.00188757f, 0.00043317f, 0.00029226f,
    414.35622318f, 0.00663720f, 0.00352537f, 0.18487235f,
    0.82303707f, 3.35892716f, 4.44531239f, 87.18168813f,
    0.00186024f, 0.00177811f, 0.00197011f, 0.00163299f,
    0.00015246f, 0.00006042f, 0.00009380f, 0.00134482f,
    0.00348671f, 0.00038141f, 0.00042675f, 414.04577969f,
    0.00928656f, 0.00352352f, 0.23531453f, 1.08896958f,
    3.36592446f, 4.62139083f, 91.45779685f, 0.00188211f,
    0.00186335f, 0.00200055f, 0.00193484f, 0.00017223f,
    0.00011896f, 0.00018584f, 0.00362476f, 0.00445541f,
    0.00197959f, 0.00244534f, 415.24463519f, 0.01282504f,
    0.00732871f
};

    const float SCALER_SCALE[57] = {
    0.00506450f, 0.03914676f, 0.11174106f, 0.37442287f,
    7.08904474f, 0.00102946f, 0.00082172f, 0.00149043f,
    0.00070366f, 0.00006002f, 0.00004824f, 0.00002930f,
    0.00071894f, 0.00095965f, 0.00028974f, 0.00011178f,
    7.68424623f, 0.00357770f, 0.00199257f, 0.01505594f,
    0.11007952f, 0.14885279f, 0.40067289f, 7.92200225f,
    0.00095025f, 0.00089508f, 0.00104583f, 0.00084937f,
    0.00005992f, 0.00002204f, 0.00003468f, 0.00224008f,
    0.00271346f, 0.00017230f, 0.00021318f, 7.45578623f,
    0.00545410f, 0.00187711f, 0.01540258f, 0.13921336f,
    0.15110808f, 0.45048460f, 8.91507420f, 0.00097468f,
    0.00095990f, 0.00106148f, 0.00100036f, 0.00005400f,
    0.00003498f, 0.00006810f, 0.00423110f, 0.00338373f,
    0.00096789f, 0.00099888f, 7.85519587f, 0.00715212f,
    0.00420579f
};

    // Saída do modelo (índice = classe):
    // 0: Normal | 1: Desgaste Superficial | 2: Dente Trincado
    // 3: Dente Lascado | 4: Dente Ausente

    #endif  // SCALER_PARAMS_H_
