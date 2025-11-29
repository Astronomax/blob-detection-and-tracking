check_include_file(arm_neon.h HAVE_ARM_NEON_H)

if(EXISTS "/proc/cpuinfo")
    file(READ "/proc/cpuinfo" CPUINFO)
    if(CPUINFO MATCHES "neon" OR CPUINFO MATCHES "asimd")
        set(HAVE_NEON TRUE)
        message(STATUS "NEON support detected")
    else()
        set(HAVE_NEON FALSE)
        message(STATUS "NEON support not detected")
    endif()
else()
    set(HAVE_NEON FALSE)
endif()

set(RASPBERRYPI FALSE)

if(EXISTS "/proc/device-tree/model")
    file(READ "/proc/device-tree/model" DEVICE_TREE_MODEL)
    if(DEVICE_TREE_MODEL MATCHES "Raspberry Pi")
        set(RASPBERRYPI TRUE)
        message(STATUS "Detected Raspberry Pi: ${DEVICE_TREE_MODEL}")
    endif()
endif()

if(NOT RASPBERRYPI AND EXISTS "/proc/cpuinfo")
    file(READ "/proc/cpuinfo" CPUINFO)
    if(CPUINFO MATCHES "Raspberry Pi")
        set(RASPBERRYPI TRUE)
        message(STATUS "Detected Raspberry Pi via cpuinfo")
    endif()
endif()

if(RASPBERRYPI)
    message(STATUS "Configuring for ARM architecture")

    find_program(LSCPU_EXECUTABLE lscpu)
    if(LSCPU_EXECUTABLE)
        execute_process(
            COMMAND ${LSCPU_EXECUTABLE}
            OUTPUT_VARIABLE CPU_INFO
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    else()
        set(CPU_INFO "")
    endif()

    if(DEVICE_TREE_MODEL MATCHES "Pi 4" OR DEVICE_TREE_MODEL MATCHES "Pi 400" OR DEVICE_TREE_MODEL MATCHES "Pi 3")
        set(TARGET_CPU "cortex-a72")
        set(TARGET_ARCH "armv8-a+crc+simd")
        message(STATUS "Detected Raspberry Pi 3/4/400 - using Cortex-A72 optimizations")
    elseif(DEVICE_TREE_MODEL MATCHES "Pi 2")
        set(TARGET_CPU "cortex-a7")
        set(TARGET_ARCH "armv7-a")
        message(STATUS "Detected Raspberry Pi 2 - using Cortex-A7 optimizations")
    elseif(DEVICE_TREE_MODEL MATCHES "Pi 1" OR DEVICE_TREE_MODEL MATCHES "Pi Zero")
        set(TARGET_CPU "arm1176jzf-s")
        set(TARGET_ARCH "armv6zk")
        message(STATUS "Detected Raspberry Pi 1/Zero - using ARM1176 optimizations")
    else()
        # Default values
        set(TARGET_CPU "native")
        set(TARGET_ARCH "armv8-a")
        message(STATUS "Unknown Raspberry Pi model, using default values")
    endif()

    set(RASPBERRY_COMMON_COMPILE_OPTIONS
        -march=${TARGET_ARCH}
        -mcpu=${TARGET_CPU}
        -mtune=${TARGET_CPU}
    )

    set(RASPBERRY_RELEASE_COMPILE_OPTIONS
        -ftree-vectorize
        -ftree-vectorizer-verbose=1
        -funsafe-math-optimizations
        -mfloat-abi=hard
        -falign-functions=32
        -falign-loops=32
        -funroll-loops
    )

    if(HAVE_NEON)
        list(APPEND RASPBERRY_COMMON_COMPILE_OPTIONS
            -mfpu=neon
            -mvectorize-with-neon-quad
        )
    endif()
endif()
