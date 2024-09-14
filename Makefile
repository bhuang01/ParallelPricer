NVCC := nvcc
NVCC_FLAGS := -O3 -arch=sm_75
INCLUDES := -I./include

TARGET := monte_carlo_sim

SRC_DIR := src
OBJ_DIR := obj

SRCS := $(wildcard $(SRC_DIR)/*.cu)
OBJS := $(SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $@

clean:
	rm -rf $(TARGET) $(OBJ_DIR)

.PHONY: all clean