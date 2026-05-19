#
# Main component — pipeline FFT + inferencia TFLite Micro
#

idf_component_register(
    SRCS main.cc main_functions.cc model.cc features.cc
    PRIV_REQUIRES spi_flash
    INCLUDE_DIRS ""
    EMBED_FILES "test_signals.bin")
