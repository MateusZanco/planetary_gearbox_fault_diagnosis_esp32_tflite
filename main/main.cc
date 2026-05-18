#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "main_functions.h"

extern "C" void app_main(void) {
  setup();
  loop();  // roda os test_vectors uma unica vez

  // Mantem a task viva para preservar o log no UART
  while (true) {
    vTaskDelay(pdMS_TO_TICKS(10000));
  }
}
