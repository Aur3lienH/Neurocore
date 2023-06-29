CC = g++

TARGET_EXEC ?= final_program
BUILD_DIR ?= ./build
SRC_DIR ?= ./src

FLAGS ?= -O3

SRCS := $(shell find $(SRC_DIR) -name *.cpp)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

all : $(BUILD_DIR)/$(TARGET_EXEC)

$(BUILD_DIR)/$(TARGET_EXEC) : $(OBJS)
	$(CC) $(OBJS) -o $@ $(FLAGS)

$(BUILD_DIR)/%.cpp.o : %.cpp
	@mkdir -p $(@D)
	$(CC) -c $< -o $@ $(FLAGS)


.PHONY: clean

clean:
	rm -f ${BUILD_DIR}/${TARGET_EXEC}
	rm -d -r ${BUILD_DIR}/src
