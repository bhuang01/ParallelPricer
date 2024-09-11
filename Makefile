NVCC := nvcc
NVCC_FLAGS := -O3 -arch=sm_75
INCLUDES := -I./include

TARGET := monte_carlo_sim

SRCS := main.cu random_generator.cu stock_simulation.cu utils.cu
OBJS := $(SRCS:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

%.o: src/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean