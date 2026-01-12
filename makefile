.PHONY = clean all

VPATH=src
BUILD_DIR:=build
TEST_DIR:=test
FLAGS:=-g -Wall -Werror

TARGETS:=common perceptron layer neuralnetwork trainer
SOURCES:=$(addsuffix .cpp, $(TARGETS))
OBJECTS:=$(addsuffix $(BUILD_DIR)/, $(SOURCES:.cpp=.o))
OUT_LIB:=$(BUILD_DIR)/libneuralnetwork.so
TEST:= #TODO: here enumerate the test executables

_TARGETS:=$(addprefix $(BUILD_DIR)/, $(EXECUTABLES))

all: $(BUILD_DIR) $(OUT_LIB) $(TEST_DIR) $(TEST)

shared_object: $(BUILD_DIR) $(OUT_LIB)


#TODO: todavia el makefile esta incompleto

$(BUILD_DIR):
	mkdir -p $@

$(TEST_DIR):
	mkdir -p $@

$(TARGETS): %.cpp:


clean:
	rm -r $(BUILD_DIR)
