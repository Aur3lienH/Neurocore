CC = g++
NVCC = nvcc

TARGET_EXEC ?= final_program
BUILD_DIR ?= ./build
SRC_DIR ?= ./src

FLAGS ?= -std=c++17 -pthread -O3 -mavx -march=native -lsfml-graphics -lsfml-window -lsfml-system
NVCC_FLAGS ?= -lsfml-graphics -lsfml-window -lsfml-system
#find cpp and cu files in src directory
SRCS := $(shell find $(SRC_DIR) -name *.cpp -or -name *.cu)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

all : $(BUILD_DIR)/$(TARGET_EXEC)

$(BUILD_DIR)/$(TARGET_EXEC) : $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $@

$(BUILD_DIR)/%.cpp.o : %.cpp
	@mkdir -p $(@D)
	$(CC) $(FLAGS) -c $< -o $@

$(BUILD_DIR)/%.cu.o : %.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@


.PHONY: clean

clean:
	rm -f ${BUILD_DIR}/${TARGET_EXEC}
	rm -d -r ${BUILD_DIR}/src
