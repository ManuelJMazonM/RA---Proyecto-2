.PHONY: clean all lib test

VPATH=src test
FLAGS:=-g -Wall -Werror
BUILD_DIR:=build
SOURCES:= common perceptron layer neuralnetwork trainer
TEST:= and-perceptron or-perceptron xor-nn experimentos_datasets experimentos_rapidos

OBJ_DIR:=$(BUILD_DIR)/obj
OBJECTS:=$(addprefix $(OBJ_DIR)/, $(addsuffix .o, $(SOURCES)))
TEST:=$(addprefix $(BUILD_DIR)/, $(TEST))
INCLUDE_DIR:=src/include

# Detectar sistema operativo
ifeq ($(OS),Windows_NT)
    # Windows: libreria estatica
    OUT_LIB:=$(BUILD_DIR)/libneuralnetwork.a
    LINK_CMD = g++ $(FLAGS) -I$(INCLUDE_DIR) $< $(OUT_LIB) -o $@
    LIB_CMD = ar rcs $@ $^
else
    # Linux: libreria dinamica
    OUT_LIB:=$(BUILD_DIR)/libneuralnetwork.so
    LINKER_FLAGS:=-Wl,-rpath,'$$ORIGIN'
    LINK_CMD = g++ $(FLAGS) -I$(INCLUDE_DIR) -L$(BUILD_DIR) $< -lneuralnetwork $(LINKER_FLAGS) -o $@
    LIB_CMD = g++ -shared $^ -o $@
endif


all: lib test

lib: build_dirs $(OUT_LIB)

test: lib $(TEST)

test-exe: lib test
	for t in $(BUILD_DIR)/*; do ./$$t; done


build_dirs:
	mkdir -p $(OBJ_DIR)


$(OUT_LIB): $(OBJECTS)
	$(LIB_CMD)


$(OBJECTS): $(OBJ_DIR)/%.o: %.cpp
	g++ $(FLAGS) -fPIC -I$(INCLUDE_DIR) -c $< -o $@


$(TEST): $(BUILD_DIR)/%: %.cpp lib
	$(LINK_CMD)


clean:
	rm -rf $(BUILD_DIR)
