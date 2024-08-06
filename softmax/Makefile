.PHONY: all clean

BINARY_DIR="./binary"

all:
	mkdir -p ${BINARY_DIR}
	nvcc -std=c++17 softmax_fwd.cu -o ${BINARY_DIR}/softmax_fwd

clean:
	rm -rf ${BINARY_DIR}
