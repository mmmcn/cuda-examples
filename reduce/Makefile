.PHONY: all clean

BINARY_DIR="./binary"

all:
	mkdir -p ${BINARY_DIR}
	nvcc -std=c++17 reduce_sum.cu -o ${BINARY_DIR}/reduce_sum

clean:
	rm -rf ${BINARY_DIR}
