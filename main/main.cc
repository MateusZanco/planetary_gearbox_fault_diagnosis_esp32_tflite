#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "main_functions.h"

// Periodo entre inferencias (simula janela de 1 s do sinal real)
#define INFERENCIA_PERIODO_MS  1000

extern "C" void app_main(void) {
  setup();
  while (true) {
    loop();
    vTaskDelay(pdMS_TO_TICKS(INFERENCIA_PERIODO_MS));
  }
}
