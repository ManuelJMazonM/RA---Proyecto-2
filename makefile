.PHONY: clean all lib test experimentos

VPATH=src test
FLAGS:=-g -Wall -Werror
BUILD_DIR:=build
OBJ_DIR:=$(BUILD_DIR)/obj
INCLUDE_DIR:=src/include

# Archivos fuente
SOURCES:= common perceptron layer neuralnetwork trainer
OBJECTS:=$(addprefix $(OBJ_DIR)/, $(addsuffix .o, $(SOURCES)))

# Detectar sistema operativo
ifeq ($(OS),Windows_NT)
    # Windows: libreria estatica, ejecutables con .exe
    OUT_LIB:=$(BUILD_DIR)/libneuralnetwork.a
    EXE_EXT:=.exe
    LINK_FLAGS:=
else
    # Linux: libreria dinamica con rpath
    OUT_LIB:=$(BUILD_DIR)/libneuralnetwork.so
    EXE_EXT:=
    LINK_FLAGS:=-Wl,-rpath,'$$ORIGIN'
endif


all: lib test
	@echo "Compilacion completada en $(if $(filter Windows_NT,$(OS)),Windows,Linux)!"

lib: build_dirs $(OUT_LIB)
	@echo "Libreria creada: $(OUT_LIB)"

test: lib $(BUILD_DIR)/and-perceptron$(EXE_EXT)
	@echo "Tests compilados!"


build_dirs:
	@mkdir -p $(OBJ_DIR)


# Crear libreria (estatica en Windows, dinamica en Linux)
ifeq ($(OS),Windows_NT)
$(OUT_LIB): $(OBJECTS)
	ar rcs $@ $^
else
$(OUT_LIB): $(OBJECTS)
	g++ -shared $^ -o $@
endif


# Compilar objetos
$(OBJ_DIR)/common.o: src/common.cpp
	g++ $(FLAGS) -fPIC -I$(INCLUDE_DIR) -c $< -o $@

$(OBJ_DIR)/perceptron.o: src/perceptron.cpp
	g++ $(FLAGS) -fPIC -I$(INCLUDE_DIR) -c $< -o $@

$(OBJ_DIR)/layer.o: src/layer.cpp
	g++ $(FLAGS) -fPIC -I$(INCLUDE_DIR) -c $< -o $@

$(OBJ_DIR)/neuralnetwork.o: src/neuralnetwork.cpp
	g++ $(FLAGS) -fPIC -I$(INCLUDE_DIR) -c $< -o $@

$(OBJ_DIR)/trainer.o: src/trainer.cpp
	g++ $(FLAGS) -fPIC -I$(INCLUDE_DIR) -c $< -o $@


# Compilar test
$(BUILD_DIR)/and-perceptron$(EXE_EXT): test/and-perceptron.cpp $(OUT_LIB)
ifeq ($(OS),Windows_NT)
	g++ $(FLAGS) -I$(INCLUDE_DIR) $< $(OUT_LIB) -o $@
else
	g++ $(FLAGS) -I$(INCLUDE_DIR) -L$(BUILD_DIR) $< -lneuralnetwork $(LINK_FLAGS) -o $@
endif


# Compilar experimentos
experimentos: lib
ifeq ($(OS),Windows_NT)
	g++ -O2 -Wall -I$(INCLUDE_DIR) test/experimentos_rapidos.cpp $(OUT_LIB) -o $(BUILD_DIR)/experimentos_rapidos.exe
	@echo "Ejecuta: build\\experimentos_rapidos.exe"
else
	g++ -O2 -Wall -I$(INCLUDE_DIR) -L$(BUILD_DIR) test/experimentos_rapidos.cpp -lneuralnetwork $(LINK_FLAGS) -o $(BUILD_DIR)/experimentos_rapidos
	@echo "Ejecuta: ./build/experimentos_rapidos"
endif


clean:
ifeq ($(OS),Windows_NT)
	@if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
else
	rm -rf $(BUILD_DIR)
endif
