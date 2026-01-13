.PHONY: clean all

VPATH=src
BUILD_DIR:=build
TEST_DIR:=test
FLAGS:=-g -Wall -Werror

OBJ_DIR:=$(BUILD_DIR)/obj

TARGETS:=common perceptron layer neuralnetwork trainer
SOURCES:=$(addsuffix .cpp, $(TARGETS))
OBJECTS:=$(addprefix $(OBJ_DIR)/, $(SOURCES:.cpp=.o))
OUT_LIB:=$(BUILD_DIR)/libneuralnetwork.so
TEST:= #TODO: here enumerate the test executables


all: lib #tests

lib: build_dirs $(OUT_LIB)

build_dirs:
	mkdir -p $(OBJ_DIR)


#TODO: a loop that compiles and executes tests (inside bulid dir)
#TEST=$(addprefix $(TEST_DIR), $(TEST))
#tests: lib


$(OBJ_DIR)/%.o: %.cpp
	g++ $(FLAGS) -fPIC -c $< -o $@


$(OUT_LIB): $(OBJECTS)
	g++ -shared $^ -o $@


clean:
	rm -r $(BUILD_DIR)
