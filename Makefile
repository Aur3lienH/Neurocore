CC = g++ -std=c++17 -pthread

TARGET_EXEC ?= final_program
BUILD_DIR ?= ./build
SRC_DIR ?= ./src

FLAGS ?= -O3 -pg

SRCS := $(shell find $(SRC_DIR) -name *.cpp)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

all : $(BUILD_DIR)/$(TARGET_EXEC)

$(BUILD_DIR)/$(TARGET_EXEC) : $(OBJS)
	$(CC) $(FLAGS) $(OBJS) -o $@

$(BUILD_DIR)/%.cpp.o : %.cpp
	@mkdir -p $(@D)
	$(CC) $(FLAGS) -c $< -o $@


.PHONY: clean

clean:
	rm -f ${BUILD_DIR}/${TARGET_EXEC}
	rm -d -r ${BUILD_DIR}/src
