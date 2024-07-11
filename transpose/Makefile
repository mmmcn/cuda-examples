.PHONY: all clean

BINARY_DIR="./binary"

all:
	mkdir -p ${BINARY_DIR}
	nvcc -arch=sm_80 transpose.cu -o ${BINARY_DIR}/transpose

clean:
	rm -rf ${BINARY_DIR}
