.PHONY: all build clean profile

CMAKE := cmake

BUILD_DIR := build
BENCHMARK_DIR := benchmark_results

all: build

build:
	@mkdir -p $(BUILD_DIR)
	nvcc -std=c++17 sgemm.cu -o ${BUILD_DIR}/sgemm


clean:
	@rm -rf $(BUILD_DIR)


# Usage: make profile KERNEL=<integer> PREFIX=<optional string>
profile: build
	@mkdir -p $(BENCHMARK_DIR)
	@ncu --launch-skip 3 --set full --export $(BENCHMARK_DIR)/$(PREFIX)kernel_$(KERNEL) --force-overwrite $(BUILD_DIR)/sgemm $(KERNEL)
