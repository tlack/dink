# Dink's engine as an ESP-IDF component. It builds on MMRT (BabyTalk's runtime, the esp32/babytalk
# submodule): the text encoder's conv trunk is a .mmrt graph on MMRT's int8 kernels, and the
# dense float32 layers use mmrt_f32.h. Dink's own code is the wiring, the harness and numbers.
idf_component_register(
    SRCS "dink.c" "dink_harness.c"
    INCLUDE_DIRS "."
    REQUIRES mmrt
)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
