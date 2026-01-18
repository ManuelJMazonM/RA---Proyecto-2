# Proyecto 2: Machine Learning
## Razonamiento Automático 2025/26

## Estructura del proyecto

```
Proyecto2_ML_Completo/
├── src/                    # Código fuente de la librería
│   ├── include/
│   │   └── neuralnetwork.hpp   # Header público (para usar la librería)
│   ├── common.cpp/hpp      # Funciones comunes, carga de datasets
│   ├── perceptron.cpp/hpp  # Implementación del Perceptrón
│   ├── layer.cpp/hpp       # Capa de neuronas
│   ├── neuralnetwork.cpp/hpp # Red Neuronal completa
│   └── trainer.cpp/hpp     # Entrenador (Backprop + PLA)
│
├── test/                   # Programas de prueba
│   ├── experimentos_rapidos.cpp    # ⭐ TEST PRINCIPAL (rápido)
│   ├── experimentos_datasets.cpp   # Test exhaustivo (más lento)
│   ├── and-perceptron.cpp          # Test básico AND
│   └── ...
│
├── datasets/               # Datasets UCI
│   ├── wine.csv            # Wine (13 features, 3 clases)
│   ├── seeds.csv           # Seeds (7 features, 3 clases)
│   └── breast_cancer.csv   # Breast Cancer (10 features, 2 clases)
│
├── iris.csv                # Dataset Iris (4 features, 3 clases)
├── makefile                # Sistema de compilación
└── README.md               # Este archivo
```

## Compilación

```bash
# Compilar todo (librería + tests básicos)
make all

# Compilar el test de experimentos rápidos
g++ -O2 -Wall -Isrc/include -Lbuild test/experimentos_rapidos.cpp -lneuralnetwork -Wl,-rpath,'$ORIGIN' -o build/experimentos_rapidos

# Compilar el test exhaustivo
g++ -O2 -Wall -Isrc/include -Lbuild test/experimentos_datasets.cpp -lneuralnetwork -Wl,-rpath,'$ORIGIN' -o build/experimentos_datasets
```

## Ejecución

```bash
# Ejecutar experimentos rápidos (recomendado, ~5 segundos)
./build/experimentos_rapidos

# Ejecutar experimentos exhaustivos (más lento)
./build/experimentos_datasets
```

## Resultados esperados

### Perceptrón - Puertas Lógicas
| Puerta | Accuracy |
|--------|----------|
| AND    | 100%     |
| OR     | 100%     |
| XOR    | 50% (no linealmente separable) |

### Red Neuronal - Datasets UCI
| Dataset        | Topología | Train Acc | Val Acc |
|----------------|-----------|-----------|---------|
| Iris           | 4-8-3     | ~97%      | ~100%   |
| Wine           | 13-8-3    | ~100%     | ~100%   |
| Seeds          | 7-8-3     | ~97%      | ~96%    |
| Breast Cancer  | 10-8-2    | ~97%      | ~100%   |

## Datasets UCI utilizados

Los datasets provienen del UCI Machine Learning Repository:

- **Iris**: https://archive.ics.uci.edu/dataset/53/iris
- **Wine**: https://archive.ics.uci.edu/dataset/109/wine
- **Seeds**: https://archive.ics.uci.edu/dataset/236/seeds
- **Breast Cancer**: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

## Uso programático

### Entrenar un Perceptrón
```cpp
#include "neuralnetwork.hpp"

// Datos de entrenamiento (AND)
vector<vector<double>> inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
vector<int> outputs = {0, 0, 0, 1};

// Crear y entrenar
Perceptron p(2);  // 2 inputs
Trainer t;
t.train(p, inputs, outputs, 0.1, 1000, ActivationType::STEP);

// Predecir
double result = p.predict({1, 1}, ActivationType::STEP);  // → positivo (clase 1)
```

### Entrenar una Red Neuronal
```cpp
#include "neuralnetwork.hpp"

// Cargar dataset (col target=4, con cabecera, 20% validación)
Dataset ds = loadDataset("iris.csv", 4, ',', true, 0.2);

// Crear red: 4 inputs → 8 hidden → 3 outputs
NeuralNetwork nn({4, 8, 3}, ActivationType::SIGMOID);

// Entrenar
Trainer trainer(0.1);  // learning rate = 0.1
trainer.train(nn, ds, 500);  // 500 epochs

// Evaluar
double acc = trainer.test_acc(nn, ds.val_inputs, ds.val_targets);
cout << "Validation accuracy: " << acc << "%" << endl;

// Predecir
int clase = nn.predict(ds.val_inputs[0]);  // → 0, 1, o 2
```

## Tecnologías implementadas

| Nivel | Tecnología | Puntos |
|-------|------------|--------|
| 0 | Perceptrón | 0.35 |
| 1 | Red Neuronal + Backpropagation | 1.25 |
| - | Funciones de activación (SIGMOID, TANH, RELU, STEP) | +0.10 |
| - | Validación (split 80/20) | +0.15 |

## Notas importantes

1. **El archivo iris.csv debe estar en el directorio raíz** (donde ejecutas el programa)
2. **Los otros datasets deben estar en la carpeta datasets/**
3. Si ves errores de "no se pudo cargar", verifica las rutas de los CSV
4. La librería se compila como `.so` (shared object), necesita el flag `-Wl,-rpath`
