idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES dink mmrt esp_partition esp_timer heap spi_flash esp_driver_usb_serial_jtag vfs
)
