#
# Main component — programa de validacao do modelo TFLite Micro
#

idf_component_register(
    SRCS main.cc main_functions.cc model.cc
    PRIV_REQUIRES spi_flash
    INCLUDE_DIRS "")
