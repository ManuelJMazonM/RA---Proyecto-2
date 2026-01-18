/**
 * Experimentos ML - Proyecto 2
 * Test exhaustivo con todos los datasets y configuraciones
 */

#include "neuralnetwork.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <functional>
#include <cmath>

using namespace std;

double measure_time(function<void()> func) {
    auto start = chrono::high_resolution_clock::now();
    func();
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(end - start).count();
}

int main() {
    cout << fixed << setprecision(1);
    
    // EXPERIMENTO 1: PERCEPTRON - PUERTAS LOGICAS
    cout << "EXPERIMENTO 1: PERCEPTRON - PUERTAS LOGICAS\n\n";
    cout << "Objetivo: Verificar que el perceptron aprende funciones linealmente separables\n\n";
    cout << "| Puerta | Epochs | Accuracy | Tiempo (ms) | Lin. Separable |\n";
    
    vector<vector<double>> logic_inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
    vector<tuple<string, vector<int>, bool>> gates = {
        {"AND", {0, 0, 0, 1}, true},
        {"OR", {0, 1, 1, 1}, true},
        {"NAND", {1, 1, 1, 0}, true},
        {"NOR", {1, 0, 0, 0}, true},
        {"XOR", {0, 1, 1, 0}, false}
    };
    
    for (auto& gate : gates) {
        Perceptron p(2);
        Trainer t;
        double time_ms;
        int epochs;
        
        time_ms = measure_time([&]() {
            epochs = t.train(p, logic_inputs, get<1>(gate), 0.1, 1000, ActivationType::STEP);
        });
        
        int correct = 0;
        for (size_t i = 0; i < logic_inputs.size(); i++) {
            double pred = p.predict(logic_inputs[i], ActivationType::STEP);
            if ((pred > 0 ? 1 : 0) == get<1>(gate)[i]) correct++;
        }
        double acc = 100.0 * correct / logic_inputs.size();
        
        cout << "| " << setw(6) << get<0>(gate) 
             << " | " << setw(6) << epochs
             << " | " << setw(7) << acc << "%"
             << " | " << setw(11) << time_ms
             << " | " << setw(14) << (get<2>(gate) ? "SI" : "NO") << " |\n";
    }
    
    // EXPERIMENTO 2: LEARNING RATE
    cout << "\nEXPERIMENTO 2: PERCEPTRON - INFLUENCIA DEL LEARNING RATE\n\n";
    cout << "Objetivo: Analizar como afecta el learning rate a la convergencia\n\n";
    cout << "| Learning Rate | Epochs hasta 100% | Tiempo (ms) | Observacion |\n";
    
    vector<int> and_outputs = {0, 0, 0, 1};
    vector<pair<double, string>> lrs = {
        {0.001, "Lento"},
        {0.010, "Lento"},
        {0.050, "Optimo"},
        {0.100, "Optimo"},
        {0.300, "Optimo"},
        {0.500, "Optimo"},
        {1.000, "Posible oscilacion"}
    };
    
    for (auto& lr_pair : lrs) {
        Perceptron p(2);
        Trainer t;
        double time_ms;
        int epochs;
        
        time_ms = measure_time([&]() {
            epochs = t.train(p, logic_inputs, and_outputs, lr_pair.first, 1000, ActivationType::STEP);
        });
        
        cout << "| " << setw(13) << lr_pair.first
             << " | " << setw(17) << epochs
             << " | " << setw(11) << time_ms
             << " | " << setw(11) << lr_pair.second << " |\n";
    }
    
    // EXPERIMENTO 3: XOR
    cout << "\nEXPERIMENTO 3: RED NEURONAL - RESOLVIENDO XOR\n\n";
    cout << "Objetivo: Demostrar que una red con capa oculta resuelve XOR\n\n";
    cout << "| Topologia  | Epochs | Train Acc | Tiempo (ms) | Resultado |\n";
    
    // Crear dataset XOR
    Dataset xor_ds;
    xor_ds.train_inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
    xor_ds.train_targets = {0, 1, 1, 0};
    xor_ds.val_inputs = xor_ds.train_inputs;
    xor_ds.val_targets = xor_ds.train_targets;
    xor_ds.num_classes = 2;
    
    vector<pair<vector<int>, int>> xor_configs = {
        {{2,2,2}, 500}, {{2,2,2}, 1000}, {{2,2,2}, 2000},
        {{2,3,2}, 500}, {{2,3,2}, 1000}, {{2,3,2}, 2000},
        {{2,4,2}, 500}, {{2,4,2}, 1000}, {{2,4,2}, 2000},
        {{2,8,2}, 500}, {{2,8,2}, 1000}, {{2,8,2}, 2000},
        {{2,4,4,2}, 500}, {{2,4,4,2}, 1000}, {{2,4,4,2}, 2000}
    };
    
    for (auto& config : xor_configs) {
        NeuralNetwork nn(config.first, ActivationType::SIGMOID);
        Trainer trainer(0.5);
        double time_ms;
        
        time_ms = measure_time([&]() {
            trainer.train(nn, xor_ds, config.second);
        });
        
        int correct = 0;
        for (size_t i = 0; i < xor_ds.train_inputs.size(); i++) {
            if (nn.predict(xor_ds.train_inputs[i]) == xor_ds.train_targets[i]) correct++;
        }
        double acc = 100.0 * correct / xor_ds.train_inputs.size();
        
        string topo_str;
        for (size_t i = 0; i < config.first.size(); i++) {
            if (i > 0) topo_str += "-";
            topo_str += to_string(config.first[i]);
        }
        
        string resultado = (acc == 100) ? "RESUELTO" : (acc >= 75 ? "Parcial" : "Fallo");
        
        cout << "| " << setw(10) << topo_str
             << " | " << setw(6) << config.second
             << " | " << setw(8) << acc << "%"
             << " | " << setw(11) << time_ms
             << " | " << setw(9) << resultado << " |\n";
    }
    
    // EXPERIMENTO 4: IRIS
    cout << "\nEXPERIMENTO 4: DATASET IRIS (UCI)\n\n";
    cout << "Fuente: https://archive.ics.uci.edu/dataset/53/iris\n";
    cout << "Descripcion: Clasificacion de 3 especies de flores (150 muestras, 4 features)\n\n";
    
    Dataset iris = loadDataset("iris.csv", 4, ',', true, 0.2);
    if (!iris.train_inputs.empty()) {
        cout << "Dataset cargado: " << iris.train_inputs.size() << " train, " 
             << iris.val_inputs.size() << " val, " << iris.num_classes << " clases\n\n";
        cout << "| Topologia    | LR   | Epochs | Train Acc | Val Acc | Tiempo (s) |\n";
        
        vector<tuple<vector<int>, double, int>> iris_configs = {
            {{4,4,3}, 0.10, 200},
            {{4,8,3}, 0.10, 200},
            {{4,8,3}, 0.10, 500},
            {{4,8,3}, 0.10, 1000},
            {{4,16,3}, 0.10, 500},
            {{4,8,8,3}, 0.10, 500},
            {{4,8,3}, 0.05, 500},
            {{4,8,3}, 0.20, 500}
        };
        
        for (auto& config : iris_configs) {
            NeuralNetwork nn(get<0>(config), ActivationType::SIGMOID);
            Trainer trainer(get<1>(config));
            
            auto start = chrono::high_resolution_clock::now();
            trainer.train(nn, iris, get<2>(config));
            auto end = chrono::high_resolution_clock::now();
            double time_s = chrono::duration<double>(end - start).count();
            
            double train_acc = trainer.test_acc(nn, iris.train_inputs, iris.train_targets);
            double val_acc = trainer.test_acc(nn, iris.val_inputs, iris.val_targets);
            
            string topo_str;
            for (size_t i = 0; i < get<0>(config).size(); i++) {
                if (i > 0) topo_str += "-";
                topo_str += to_string(get<0>(config)[i]);
            }
            
            cout << "| " << setw(12) << topo_str
                 << " | " << setw(4) << get<1>(config)
                 << " | " << setw(6) << get<2>(config)
                 << " | " << setw(8) << train_acc << "%"
                 << " | " << setw(6) << val_acc << "%"
                 << " | " << setw(10) << setprecision(3) << time_s << " |\n";
        }
    }
    
    // EXPERIMENTO 5: WINE
    cout << "\nEXPERIMENTO 5: DATASET WINE (UCI)\n\n";
    cout << "Fuente: https://archive.ics.uci.edu/dataset/109/wine\n";
    cout << "Descripcion: Clasificacion de 3 tipos de vino italiano (178 muestras, 13 features)\n\n";
    
    Dataset wine = loadDataset("datasets/wine.csv", 0, ',', true, 0.2);
    if (!wine.train_inputs.empty()) {
        cout << "Dataset cargado: " << wine.train_inputs.size() << " train, "
             << wine.val_inputs.size() << " val, " << wine.num_classes << " clases\n\n";
        cout << "| Topologia      | LR   | Epochs | Train Acc | Val Acc |\n";
        
        vector<tuple<vector<int>, double, int>> wine_configs = {
            {{13,8,3}, 0.10, 500},
            {{13,16,3}, 0.10, 500},
            {{13,8,8,3}, 0.10, 500},
            {{13,16,8,3}, 0.10, 1000}
        };
        
        for (auto& config : wine_configs) {
            NeuralNetwork nn(get<0>(config), ActivationType::SIGMOID);
            Trainer trainer(get<1>(config));
            trainer.train(nn, wine, get<2>(config));
            
            double train_acc = trainer.test_acc(nn, wine.train_inputs, wine.train_targets);
            double val_acc = trainer.test_acc(nn, wine.val_inputs, wine.val_targets);
            
            string topo_str;
            for (size_t i = 0; i < get<0>(config).size(); i++) {
                if (i > 0) topo_str += "-";
                topo_str += to_string(get<0>(config)[i]);
            }
            
            cout << "| " << setw(14) << topo_str
                 << " | " << setw(4) << get<1>(config)
                 << " | " << setw(6) << get<2>(config)
                 << " | " << setw(8) << train_acc << "%"
                 << " | " << setw(6) << val_acc << "% |\n";
        }
    }
    
    // EXPERIMENTO 6: SEEDS
    cout << "\nEXPERIMENTO 6: DATASET SEEDS (UCI)\n\n";
    cout << "Fuente: https://archive.ics.uci.edu/dataset/236/seeds\n";
    cout << "Descripcion: Clasificacion de 3 variedades de trigo (210 muestras, 7 features)\n\n";
    
    Dataset seeds = loadDataset("datasets/seeds.csv", 7, ',', true, 0.2);
    if (!seeds.train_inputs.empty()) {
        cout << "Dataset cargado: " << seeds.train_inputs.size() << " train, "
             << seeds.val_inputs.size() << " val, " << seeds.num_classes << " clases\n\n";
        cout << "| Topologia    | Epochs | Train Acc | Val Acc |\n";
        
        vector<pair<vector<int>, int>> seeds_configs = {
            {{7,8,3}, 500},
            {{7,16,3}, 500},
            {{7,8,8,3}, 500},
            {{7,16,3}, 1000}
        };
        
        for (auto& config : seeds_configs) {
            NeuralNetwork nn(config.first, ActivationType::SIGMOID);
            Trainer trainer(0.1);
            trainer.train(nn, seeds, config.second);
            
            double train_acc = trainer.test_acc(nn, seeds.train_inputs, seeds.train_targets);
            double val_acc = trainer.test_acc(nn, seeds.val_inputs, seeds.val_targets);
            
            string topo_str;
            for (size_t i = 0; i < config.first.size(); i++) {
                if (i > 0) topo_str += "-";
                topo_str += to_string(config.first[i]);
            }
            
            cout << "| " << setw(12) << topo_str
                 << " | " << setw(6) << config.second
                 << " | " << setw(8) << train_acc << "%"
                 << " | " << setw(6) << val_acc << "% |\n";
        }
    }
    
    // EXPERIMENTO 7: BREAST CANCER
    cout << "\nEXPERIMENTO 7: BREAST CANCER WISCONSIN (UCI)\n\n";
    cout << "Fuente: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic\n";
    cout << "Descripcion: Diagnostico de cancer de mama (569 muestras, 10 features)\n\n";
    
    Dataset bc = loadDataset("datasets/breast_cancer.csv", 0, ',', true, 0.2);
    if (!bc.train_inputs.empty()) {
        cout << "Dataset cargado: " << bc.train_inputs.size() << " train, "
             << bc.val_inputs.size() << " val, " << bc.num_classes << " clases\n\n";
        cout << "| Topologia    | Epochs | Train Acc | Val Acc |\n";
        
        vector<pair<vector<int>, int>> bc_configs = {
            {{10,8,2}, 500},
            {{10,16,2}, 500},
            {{10,8,4,2}, 500},
            {{10,16,2}, 1000}
        };
        
        for (auto& config : bc_configs) {
            NeuralNetwork nn(config.first, ActivationType::SIGMOID);
            Trainer trainer(0.1);
            trainer.train(nn, bc, config.second);
            
            double train_acc = trainer.test_acc(nn, bc.train_inputs, bc.train_targets);
            double val_acc = trainer.test_acc(nn, bc.val_inputs, bc.val_targets);
            
            string topo_str;
            for (size_t i = 0; i < config.first.size(); i++) {
                if (i > 0) topo_str += "-";
                topo_str += to_string(config.first[i]);
            }
            
            cout << "| " << setw(12) << topo_str
                 << " | " << setw(6) << config.second
                 << " | " << setw(8) << train_acc << "%"
                 << " | " << setw(6) << val_acc << "% |\n";
        }
    }
    
    // EXPERIMENTO 8: FUNCIONES DE ACTIVACION
    cout << "\nEXPERIMENTO 8: COMPARACION DE FUNCIONES DE ACTIVACION\n\n";
    cout << "Objetivo: Comparar rendimiento de SIGMOID, TANH y RELU\n\n";
    
    if (!iris.train_inputs.empty()) {
        cout << "| Activacion | Train Acc | Val Acc | Observacion |\n";
        
        vector<tuple<string, ActivationType, string>> activations = {
            {"SIGMOID", ActivationType::SIGMOID, "Estable"},
            {"TANH", ActivationType::TANH, "Converge rapido"},
            {"RELU", ActivationType::RELU, "Dying neurons"}
        };
        
        for (auto& act : activations) {
            NeuralNetwork nn({4,8,3}, get<1>(act));
            Trainer trainer(0.1);
            trainer.train(nn, iris, 500);
            
            double train_acc = trainer.test_acc(nn, iris.train_inputs, iris.train_targets);
            double val_acc = trainer.test_acc(nn, iris.val_inputs, iris.val_targets);
            
            cout << "| " << setw(10) << get<0>(act)
                 << " | " << setw(8) << train_acc << "%"
                 << " | " << setw(6) << val_acc << "%"
                 << " | " << setw(11) << get<2>(act) << " |\n";
        }
    }
    
    // EXPERIMENTO 9: OVERFITTING
    cout << "\nEXPERIMENTO 9: DETECCION DE OVERFITTING\n\n";
    cout << "Objetivo: Identificar sobreajuste comparando train vs validation accuracy\n\n";
    
    if (!iris.train_inputs.empty()) {
        cout << "| Red           | Epochs | Train Acc | Val Acc | Diferencia | Estado |\n";
        
        vector<tuple<string, vector<int>, int>> overfit_configs = {
            {"4-4-3 (pequeña)", {4,4,3}, 100},
            {"4-4-3 (pequeña)", {4,4,3}, 500},
            {"4-4-3 (pequeña)", {4,4,3}, 2000},
            {"4-32-3 (grande)", {4,32,3}, 100},
            {"4-32-3 (grande)", {4,32,3}, 500},
            {"4-32-3 (grande)", {4,32,3}, 2000}
        };
        
        for (auto& config : overfit_configs) {
            NeuralNetwork nn(get<1>(config), ActivationType::SIGMOID);
            Trainer trainer(0.1);
            trainer.train(nn, iris, get<2>(config));
            
            double train_acc = trainer.test_acc(nn, iris.train_inputs, iris.train_targets);
            double val_acc = trainer.test_acc(nn, iris.val_inputs, iris.val_targets);
            double diff = train_acc - val_acc;
            string estado = (abs(diff) > 10) ? "OVERFITTING" : "OK";
            
            cout << "| " << setw(13) << get<0>(config)
                 << " | " << setw(6) << get<2>(config)
                 << " | " << setw(8) << train_acc << "%"
                 << " | " << setw(6) << val_acc << "%"
                 << " | " << setw(9) << showpos << diff << "%" << noshowpos
                 << " | " << setw(6) << estado << " |\n";
        }
    }
    
    return 0;
}
