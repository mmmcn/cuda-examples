.PHONY: all clean

BINARY_DIR="./binary"

all:
	mkdir -p ${BINARY_DIR}
	nvcc -std=c++17 -arch=sm_89 embedding_bwd.cu -o ${BINARY_DIR}/embedding_bwd

clean:
	rm -rf ${BINARY_DIR}
