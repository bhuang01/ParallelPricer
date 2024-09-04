NVCC := nvcc
NVCC_FLAGS := -O3 -arch=sm_75  # Adjust the architecture as needed

TARGET := stock_sim

all: $(TARGET)

$(TARGET): stock_sim.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean