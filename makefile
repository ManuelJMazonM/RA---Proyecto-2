.PHONY: clean all

VPATH=src test
FLAGS:=-g -Wall -Werror
LINKER_FLAGS:=-Wl,-rpath,'$$ORIGIN'

BUILD_DIR:=build
SOURCES:= common perceptron layer neuralnetwork trainer
OUT_LIB:= libneuralnetwork.so
TEST:= and-perceptron or-perceptron xor-nn

OBJ_DIR:=$(BUILD_DIR)/obj
OBJECTS:=$(addprefix $(OBJ_DIR)/, $(addsuffix .o, $(SOURCES)))
OUT_LIB:=$(BUILD_DIR)/$(OUT_LIB)
TEST:=$(addprefix $(BUILD_DIR)/, $(TEST))
INCLUDE_DIR:=src/include


all: lib test

lib: build_dirs $(OUT_LIB)

test: lib $(TEST)

test-exe: lib test
	for t in $(BUILD_DIR)/*; do ./$$t; done


build_dirs:
	mkdir -p $(OBJ_DIR)


$(OUT_LIB): $(OBJECTS)
	g++ -shared $^ -o $@


$(OBJECTS): $(OBJ_DIR)/%.o: %.cpp
	g++ $(FLAGS) -fPIC -c $< -o $@


$(TEST): $(BUILD_DIR)/%: %.cpp
	g++ $(FLAGS) -I$(INCLUDE_DIR) -L$(BUILD_DIR) $< -lneuralnetwork $(LINKER_FLAGS) -o $@


clean:
	rm -r $(BUILD_DIR)
