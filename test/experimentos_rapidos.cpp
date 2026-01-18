/**
 * Experimentos ML - Proyecto 2
 * Test rápido (~5 segundos)
 */

#include "neuralnetwork.hpp"
#include <iostream>
#include <vector>

using namespace std;

int main() {
    cout << "EXPERIMENTOS ML - Proyecto 2\n\n";
    
    // 1. PERCEPTRON - PUERTAS LOGICAS
    cout << "1. PERCEPTRON - PUERTAS LOGICAS\n";
    cout << "| Gate | Accuracy | Separable |\n";
    
    vector<vector<double>> logic_inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
    vector<pair<string, vector<int>>> gates = {
        {"AND", {0, 0, 0, 1}},
        {"OR", {0, 1, 1, 1}},
        {"XOR", {0, 1, 1, 0}}
    };
    
    for (auto& gate : gates) {
        Perceptron p(2);
        Trainer t;
        t.train(p, logic_inputs, gate.second, 0.1, 1000, ActivationType::STEP);
        
        int correct = 0;
        for (size_t i = 0; i < logic_inputs.size(); i++) {
            double pred = p.predict(logic_inputs[i], ActivationType::STEP);
            if ((pred > 0 ? 1 : 0) == gate.second[i]) correct++;
        }
        double acc = 100.0 * correct / logic_inputs.size();
        cout << "| " << gate.first << " | " << acc << "% | " 
             << (acc == 100 ? "SI" : "NO") << " |\n";
    }
    
    // 2. PERCEPTRON - LEARNING RATE
    cout << "\n2. PERCEPTRON - LEARNING RATE\n";
    cout << "| LR | Epochs |\n";
    
    vector<double> lrs = {0.01, 0.1, 0.5};
    vector<int> and_outputs = {0, 0, 0, 1};
    
    for (double lr : lrs) {
        Perceptron p(2);
        Trainer t;
        int epochs = t.train(p, logic_inputs, and_outputs, lr, 1000, ActivationType::STEP);
        cout << "| " << lr << " | " << epochs << " |\n";
    }
    
    // 3. RED NEURONAL - XOR (usando Dataset)
    cout << "\n3. RED NEURONAL - XOR\n";
    cout << "| Topologia | Epochs | Acc |\n";
    
    // Crear dataset XOR manualmente
    Dataset xor_ds;
    xor_ds.train_inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
    xor_ds.train_targets = {0, 1, 1, 0};
    xor_ds.val_inputs = xor_ds.train_inputs;
    xor_ds.val_targets = xor_ds.train_targets;
    xor_ds.num_classes = 2;
    
    vector<vector<int>> topos = {{2,4,2}, {2,8,2}};
    
    for (auto& topo : topos) {
        NeuralNetwork nn(topo, ActivationType::SIGMOID);
        Trainer trainer(0.5);
        trainer.train(nn, xor_ds, 1000);
        
        int correct = 0;
        for (size_t i = 0; i < xor_ds.train_inputs.size(); i++) {
            if (nn.predict(xor_ds.train_inputs[i]) == xor_ds.train_targets[i]) correct++;
        }
        double acc = 100.0 * correct / xor_ds.train_inputs.size();
        
        string topo_str = to_string(topo[0]);
        for (size_t i = 1; i < topo.size(); i++) topo_str += "-" + to_string(topo[i]);
        cout << "| " << topo_str << " | 1000 | " << acc << "% |\n";
    }
    
    // 4. DATASET IRIS
    cout << "\n4. DATASET IRIS (UCI)\n";
    cout << "Fuente: https://archive.ics.uci.edu/dataset/53/iris\n";
    
    Dataset iris = loadDataset("iris.csv", 4, ',', true, 0.2);
    if (iris.train_inputs.empty()) {
        cout << "Error: No se pudo cargar iris.csv\n";
    } else {
        cout << "Cargado: " << iris.train_inputs.size() << " train, " << iris.val_inputs.size() << " val\n";
        cout << "| Topologia | Train | Val |\n";
        
        vector<vector<int>> iris_topos = {{4,4,3}, {4,8,3}, {4,16,3}};
        for (auto& topo : iris_topos) {
            NeuralNetwork nn(topo, ActivationType::SIGMOID);
            Trainer trainer(0.1);
            trainer.train(nn, iris, 500);
            
            double train_acc = trainer.test_acc(nn, iris.train_inputs, iris.train_targets);
            double val_acc = trainer.test_acc(nn, iris.val_inputs, iris.val_targets);
            
            string topo_str = to_string(topo[0]);
            for (size_t i = 1; i < topo.size(); i++) topo_str += "-" + to_string(topo[i]);
            cout << "| " << topo_str << " | " << train_acc << "% | " << val_acc << "% |\n";
        }
    }
    
    // 5. DATASET WINE
    cout << "\n5. DATASET WINE (UCI)\n";
    cout << "Fuente: https://archive.ics.uci.edu/dataset/109/wine\n";
    
    Dataset wine = loadDataset("datasets/wine.csv", 0, ',', true, 0.2);
    if (wine.train_inputs.empty()) {
        cout << "Error: No se pudo cargar datasets/wine.csv\n";
    } else {
        cout << "Cargado: " << wine.train_inputs.size() << " train\n";
        NeuralNetwork nn({13,8,3}, ActivationType::SIGMOID);
        Trainer trainer(0.1);
        trainer.train(nn, wine, 500);
        double train_acc = trainer.test_acc(nn, wine.train_inputs, wine.train_targets);
        double val_acc = trainer.test_acc(nn, wine.val_inputs, wine.val_targets);
        cout << "Topologia 13-8-3: Train=" << train_acc << "% Val=" << val_acc << "%\n";
    }
    
    // 6. DATASET SEEDS
    cout << "\n6. DATASET SEEDS (UCI)\n";
    cout << "Fuente: https://archive.ics.uci.edu/dataset/236/seeds\n";
    
    Dataset seeds = loadDataset("datasets/seeds.csv", 7, ',', true, 0.2);
    if (seeds.train_inputs.empty()) {
        cout << "Error: No se pudo cargar datasets/seeds.csv\n";
    } else {
        cout << "Cargado: " << seeds.train_inputs.size() << " train\n";
        NeuralNetwork nn({7,8,3}, ActivationType::SIGMOID);
        Trainer trainer(0.1);
        trainer.train(nn, seeds, 500);
        double train_acc = trainer.test_acc(nn, seeds.train_inputs, seeds.train_targets);
        double val_acc = trainer.test_acc(nn, seeds.val_inputs, seeds.val_targets);
        cout << "Topologia 7-8-3: Train=" << train_acc << "% Val=" << val_acc << "%\n";
    }
    
    // 7. DATASET BREAST CANCER
    cout << "\n7. DATASET BREAST CANCER (UCI)\n";
    cout << "Fuente: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic\n";
    
    Dataset bc = loadDataset("datasets/breast_cancer.csv", 0, ',', true, 0.2);
    if (bc.train_inputs.empty()) {
        cout << "Error: No se pudo cargar datasets/breast_cancer.csv\n";
    } else {
        cout << "Cargado: " << bc.train_inputs.size() << " train\n";
        NeuralNetwork nn({10,8,2}, ActivationType::SIGMOID);
        Trainer trainer(0.1);
        trainer.train(nn, bc, 500);
        double train_acc = trainer.test_acc(nn, bc.train_inputs, bc.train_targets);
        double val_acc = trainer.test_acc(nn, bc.val_inputs, bc.val_targets);
        cout << "Topologia 10-8-2: Train=" << train_acc << "% Val=" << val_acc << "%\n";
    }
    
    // 8. FUNCIONES DE ACTIVACION
    cout << "\n8. FUNCIONES DE ACTIVACION\n";
    if (!iris.train_inputs.empty()) {
        cout << "| Funcion | Train | Val |\n";
        
        vector<pair<string, ActivationType>> activations = {
            {"SIGMOID", ActivationType::SIGMOID},
            {"TANH", ActivationType::TANH}
        };
        
        for (auto& act : activations) {
            NeuralNetwork nn({4,8,3}, act.second);
            Trainer trainer(0.1);
            trainer.train(nn, iris, 500);
            double train_acc = trainer.test_acc(nn, iris.train_inputs, iris.train_targets);
            double val_acc = trainer.test_acc(nn, iris.val_inputs, iris.val_targets);
            cout << "| " << act.first << " | " << train_acc << "% | " << val_acc << "% |\n";
        }
    }

    
    return 0;
}
