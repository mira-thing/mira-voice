# Cross-build toolchain for the Car Thing C binaries 
FROM debian:trixie-slim

ARG TF_REF=v2.17.0
ARG ENABLE_XNNPACK=OFF

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    crossbuild-essential-arm64 \
    cmake \
    ninja-build \
    git \
    ca-certificates \
    python3 \
    pkg-config \
 && rm -rf /var/lib/apt/lists/*

ENV CROSS=aarch64-linux-gnu
# strict ARMv8.0-A baseline
ENV ARMFLAGS="-march=armv8-a+crypto+crc -mtune=cortex-a53 -moutline-atomics -O3"

WORKDIR /build
RUN git clone --depth 1 --branch ${TF_REF} https://github.com/tensorflow/tensorflow.git

# cross-compile the TFLite C API shared library
RUN cmake -S tensorflow/tensorflow/lite/c -B tflite_build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
        -DCMAKE_C_COMPILER=${CROSS}-gcc \
        -DCMAKE_CXX_COMPILER=${CROSS}-g++ \
        -DCMAKE_C_FLAGS="${ARMFLAGS}" \
        -DCMAKE_CXX_FLAGS="${ARMFLAGS}" \
        -DTFLITE_ENABLE_XNNPACK=${ENABLE_XNNPACK} \
        -DTFLITE_ENABLE_RUY=ON \
        -DCMAKE_FIND_ROOT_PATH=/usr/${CROSS} \
        -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
        -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
        -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
 && cmake --build tflite_build -j"$(nproc)" --target tensorflowlite_c

# static tinyalsa for mic capture
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    linux-libc-dev-arm64-cross \
 && rm -rf /var/lib/apt/lists/* \
 && git clone --depth 1 https://github.com/tinyalsa/tinyalsa.git \
 && ${CROSS}-gcc ${ARMFLAGS} -Itinyalsa/include -c tinyalsa/src/*.c \
 && ${CROSS}-ar rcs /build/libtinyalsa_static.a *.o \
 && rm -f *.o

# build the wake harness against the TFLite C API + static tinyalsa
COPY src/oww_wake.c /build/oww_wake.c
RUN ${CROSS}-gcc ${ARMFLAGS} \
        -I tensorflow -I tinyalsa/include \
        -o /build/oww_wake /build/oww_wake.c \
        -Ltflite_build -ltensorflowlite_c \
        -L/build -ltinyalsa_static \
        -lm -lpthread -ldl \
 && ${CROSS}-strip /build/oww_wake
