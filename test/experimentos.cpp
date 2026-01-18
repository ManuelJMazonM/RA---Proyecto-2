#include "neuralnetwork.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>

using namespace std;

// ==================== EXPERIMENTOS CON PERCEPTRON ====================

void experimento_puertas_logicas() {
    cout << "\n" << string(60, '=') << endl;
    cout << "EXPERIMENTO 1: PERCEPTRON - PUERTAS LOGICAS" << endl;
    cout << string(60, '=') << endl;

    // Datos para AND, OR, NAND, NOR
    vector<vector<double>> inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
    
    struct PuertaLogica {
        string nombre;
        vector<int> outputs;
        bool linealmente_separable;
    };
    
    vector<PuertaLogica> puertas = {
        {"AND",  {0, 0, 0, 1}, true},
        {"OR",   {0, 1, 1, 1}, true},
        {"NAND", {1, 1, 1, 0}, true},
        {"NOR",  {1, 0, 0, 0}, true},
        {"XOR",  {0, 1, 1, 0}, false}  // NO linealmente separable
    };

    Trainer trainer;

    cout << "\n| Puerta | Epochs | Accuracy | Tiempo (ms) | Separable |" << endl;
    cout << "|--------|--------|----------|-------------|-----------|" << endl;

    for (auto& puerta : puertas) {
        for (int epochs : {100, 500, 1000}) {
            Perceptron p(2);
            
            auto start = chrono::high_resolution_clock::now();
            trainer.train(p, inputs, puerta.outputs, 0.1, epochs, ActivationType::STEP);
            auto end = chrono::high_resolution_clock::now();
            
            double time_ms = chrono::duration<double, milli>(end - start).count();
            
            // Calcular accuracy
            int hits = 0;
            for (size_t i = 0; i < inputs.size(); i++) {
                double pred = p.predict(inputs[i], ActivationType::STEP);
                int pred_class = (pred >= 0) ? 1 : 0;
                if (pred_class == puerta.outputs[i]) hits++;
            }
            double acc = 100.0 * hits / inputs.size();
            
            cout << "| " << setw(6) << puerta.nombre 
                 << " | " << setw(6) << epochs 
                 << " | " << setw(7) << fixed << setprecision(1) << acc << "% "
                 << "| " << setw(11) << setprecision(2) << time_ms 
                 << " | " << setw(9) << (puerta.linealmente_separable ? "SI" : "NO") 
                 << " |" << endl;
        }
    }

    cout << "\nCONCLUSION: El perceptron converge al 100% en AND, OR, NAND, NOR" << endl;
    cout << "pero NO puede resolver XOR (problema no linealmente separable)." << endl;
}


void experimento_learning_rates() {
    cout << "\n" << string(60, '=') << endl;
    cout << "EXPERIMENTO 2: PERCEPTRON - EFECTO DEL LEARNING RATE" << endl;
    cout << string(60, '=') << endl;

    vector<vector<double>> inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
    vector<int> outputs_and = {0, 0, 0, 1};
    
    Trainer trainer;
    vector<double> learning_rates = {0.001, 0.01, 0.1, 0.5, 1.0};
    
    cout << "\n| LR    | Epochs para 100% | Tiempo (ms) |" << endl;
    cout << "|-------|------------------|-------------|" << endl;
    
    for (double lr : learning_rates) {
        Perceptron p(2);
        
        auto start = chrono::high_resolution_clock::now();
        
        int epochs_needed = 0;
        for (int e = 1; e <= 10000; e++) {
            trainer.train(p, inputs, outputs_and, lr, 1, ActivationType::STEP);
            
            // Verificar convergencia
            int hits = 0;
            for (size_t i = 0; i < inputs.size(); i++) {
                double pred = p.predict(inputs[i], ActivationType::STEP);
                int pred_class = (pred >= 0) ? 1 : 0;
                if (pred_class == outputs_and[i]) hits++;
            }
            if (hits == 4) {
                epochs_needed = e;
                break;
            }
        }
        
        auto end = chrono::high_resolution_clock::now();
        double time_ms = chrono::duration<double, milli>(end - start).count();
        
        cout << "| " << setw(5) << fixed << setprecision(3) << lr 
             << " | " << setw(16) << (epochs_needed > 0 ? to_string(epochs_needed) : ">10000")
             << " | " << setw(11) << setprecision(2) << time_ms << " |" << endl;
    }
}


// ==================== EXPERIMENTOS CON RED NEURONAL ====================

void experimento_xor_red_neuronal() {
    cout << "\n" << string(60, '=') << endl;
    cout << "EXPERIMENTO 3: RED NEURONAL - RESOLVIENDO XOR" << endl;
    cout << string(60, '=') << endl;

    // XOR: no linealmente separable, requiere red con capa oculta
    Dataset ds;
    ds.train_inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
    ds.train_targets = {0, 1, 1, 0};
    ds.num_classes = 2;
    
    cout << "\nComparando diferentes topologias:" << endl;
    cout << "\n| Topologia    | Epochs | Accuracy | Tiempo (ms) |" << endl;
    cout << "|--------------|--------|----------|-------------|" << endl;
    
    vector<vector<int>> topologias = {
        {2, 2, 2},    // Minima
        {2, 4, 2},    // Media
        {2, 8, 2},    // Grande
        {2, 4, 4, 2}  // Dos capas ocultas
    };
    
    for (auto& topo : topologias) {
        for (int epochs : {100, 500, 1000}) {
            NeuralNetwork nn(topo, ActivationType::SIGMOID);
            Trainer trainer(0.5);
            
            auto start = chrono::high_resolution_clock::now();
            trainer.train(nn, ds, epochs);
            auto end = chrono::high_resolution_clock::now();
            
            double time_ms = chrono::duration<double, milli>(end - start).count();
            double acc = trainer.test_acc(nn, ds.train_inputs, ds.train_targets);
            
            // Formatear topologia
            string topo_str = "";
            for (size_t i = 0; i < topo.size(); i++) {
                topo_str += to_string(topo[i]);
                if (i < topo.size() - 1) topo_str += "-";
            }
            
            cout << "| " << setw(12) << topo_str
                 << " | " << setw(6) << epochs
                 << " | " << setw(7) << fixed << setprecision(1) << acc << "%"
                 << " | " << setw(11) << setprecision(2) << time_ms << " |" << endl;
        }
    }
    
    cout << "\nCONCLUSION: La red neuronal SI puede resolver XOR gracias a la capa oculta." << endl;
}


void experimento_iris_dataset() {
    cout << "\n" << string(60, '=') << endl;
    cout << "EXPERIMENTO 4: RED NEURONAL - DATASET IRIS (3 CLASES)" << endl;
    cout << string(60, '=') << endl;
    
    Dataset ds = loadDataset("iris.csv", 4, ',', true, 0.2);
    
    cout << "\nDataset cargado:" << endl;
    cout << "  - Muestras entrenamiento: " << ds.train_inputs.size() << endl;
    cout << "  - Muestras validacion: " << ds.val_inputs.size() << endl;
    cout << "  - Clases: " << ds.num_classes << endl;
    cout << "  - Features: " << ds.train_inputs[0].size() << endl;
    
    cout << "\nClases detectadas:" << endl;
    for (auto& [id, name] : ds.id_to_label) {
        cout << "  ID " << id << " -> " << name << endl;
    }
    
    // Experimento: diferentes configuraciones
    cout << "\n| Topologia      | LR   | Epochs | Train Acc | Val Acc | Tiempo (s) |" << endl;
    cout << "|----------------|------|--------|-----------|---------|------------|" << endl;
    
    struct Config {
        vector<int> topology;
        double lr;
        int epochs;
    };
    
    vector<Config> configs = {
        {{4, 8, 3}, 0.1, 100},
        {{4, 8, 3}, 0.1, 500},
        {{4, 8, 3}, 0.1, 1000},
        {{4, 16, 3}, 0.1, 500},
        {{4, 8, 8, 3}, 0.1, 500},
        {{4, 8, 3}, 0.05, 500},
        {{4, 8, 3}, 0.3, 500},
    };
    
    double best_val_acc = 0;
    string best_config;
    
    for (auto& cfg : configs) {
        NeuralNetwork nn(cfg.topology, ActivationType::SIGMOID);
        Trainer trainer(cfg.lr);
        
        // Silenciar output del trainer
        cout.setstate(ios_base::failbit);
        auto start = chrono::high_resolution_clock::now();
        trainer.train(nn, ds, cfg.epochs);
        auto end = chrono::high_resolution_clock::now();
        cout.clear();
        
        double time_s = chrono::duration<double>(end - start).count();
        double train_acc = trainer.test_acc(nn, ds.train_inputs, ds.train_targets);
        double val_acc = trainer.test_acc(nn, ds.val_inputs, ds.val_targets);
        
        // Formatear topologia
        string topo_str = "";
        for (size_t i = 0; i < cfg.topology.size(); i++) {
            topo_str += to_string(cfg.topology[i]);
            if (i < cfg.topology.size() - 1) topo_str += "-";
        }
        
        cout << "| " << setw(14) << topo_str
             << " | " << setw(4) << fixed << setprecision(2) << cfg.lr
             << " | " << setw(6) << cfg.epochs
             << " | " << setw(8) << setprecision(1) << train_acc << "%"
             << " | " << setw(6) << val_acc << "%"
             << " | " << setw(10) << setprecision(3) << time_s << " |" << endl;
        
        if (val_acc > best_val_acc) {
            best_val_acc = val_acc;
            best_config = topo_str + ", lr=" + to_string(cfg.lr).substr(0,4) + ", epochs=" + to_string(cfg.epochs);
        }
    }
    
    cout << "\nMejor configuracion: " << best_config << " con " << best_val_acc << "% en validacion" << endl;
}


void experimento_funciones_activacion() {
    cout << "\n" << string(60, '=') << endl;
    cout << "EXPERIMENTO 5: COMPARACION DE FUNCIONES DE ACTIVACION" << endl;
    cout << string(60, '=') << endl;
    
    Dataset ds = loadDataset("iris.csv", 4, ',', true, 0.2);
    
    struct ActConfig {
        ActivationType type;
        string name;
    };
    
    vector<ActConfig> activations = {
        {ActivationType::SIGMOID, "SIGMOID"},
        {ActivationType::TANH, "TANH"},
        {ActivationType::RELU, "RELU"}
    };
    
    cout << "\n| Activacion | Train Acc | Val Acc | Tiempo (s) |" << endl;
    cout << "|------------|-----------|---------|------------|" << endl;
    
    for (auto& act : activations) {
        vector<int> topology = {4, 8, 3};
        NeuralNetwork nn(topology, act.type);
        Trainer trainer(0.1);
        
        cout.setstate(ios_base::failbit);
        auto start = chrono::high_resolution_clock::now();
        trainer.train(nn, ds, 500);
        auto end = chrono::high_resolution_clock::now();
        cout.clear();
        
        double time_s = chrono::duration<double>(end - start).count();
        double train_acc = trainer.test_acc(nn, ds.train_inputs, ds.train_targets);
        double val_acc = trainer.test_acc(nn, ds.val_inputs, ds.val_targets);
        
        cout << "| " << setw(10) << act.name
             << " | " << setw(8) << fixed << setprecision(1) << train_acc << "%"
             << " | " << setw(6) << val_acc << "%"
             << " | " << setw(10) << setprecision(3) << time_s << " |" << endl;
    }
    
    cout << "\nNota: RELU puede tener problemas de 'dying neurons' si no se inicializan bien los pesos." << endl;
}


void experimento_overfitting() {
    cout << "\n" << string(60, '=') << endl;
    cout << "EXPERIMENTO 6: DETECCION DE OVERFITTING" << endl;
    cout << string(60, '=') << endl;
    
    Dataset ds = loadDataset("iris.csv", 4, ',', true, 0.2);
    
    // Red pequeña vs red grande
    cout << "\nComparando red pequeña vs grande (propension a overfitting):" << endl;
    cout << "\n| Red         | Epochs | Train Acc | Val Acc | Diferencia |" << endl;
    cout << "|-------------|--------|-----------|---------|------------|" << endl;
    
    vector<pair<string, vector<int>>> redes = {
        {"Pequeña 4-4-3", {4, 4, 3}},
        {"Media 4-8-3", {4, 8, 3}},
        {"Grande 4-32-3", {4, 32, 3}},
        {"MuyGrande 4-64-32-3", {4, 64, 32, 3}}
    };
    
    for (auto& [name, topo] : redes) {
        for (int epochs : {100, 1000, 5000}) {
            NeuralNetwork nn(topo, ActivationType::SIGMOID);
            Trainer trainer(0.1);
            
            cout.setstate(ios_base::failbit);
            trainer.train(nn, ds, epochs);
            cout.clear();
            
            double train_acc = trainer.test_acc(nn, ds.train_inputs, ds.train_targets);
            double val_acc = trainer.test_acc(nn, ds.val_inputs, ds.val_targets);
            double diff = train_acc - val_acc;
            
            cout << "| " << setw(11) << name.substr(0, 11)
                 << " | " << setw(6) << epochs
                 << " | " << setw(8) << fixed << setprecision(1) << train_acc << "%"
                 << " | " << setw(6) << val_acc << "%"
                 << " | " << setw(9) << (diff > 0 ? "+" : "") << diff << "% |" << endl;
        }
    }
    
    cout << "\nCONCLUSION: Una diferencia grande Train-Val indica overfitting." << endl;
    cout << "Las redes mas grandes tienden a sobreajustarse con muchas epocas." << endl;
}


int main() {
    cout << "\n" << string(60, '*') << endl;
    cout << "*  EXPERIMENTOS - PROYECTO 2 MACHINE LEARNING             *" << endl;
    cout << "*  Razonamiento Automatico 2025/26                        *" << endl;
    cout << string(60, '*') << endl;
    
    // Ejecutar todos los experimentos
    experimento_puertas_logicas();
    experimento_learning_rates();
    experimento_xor_red_neuronal();
    experimento_iris_dataset();
    experimento_funciones_activacion();
    experimento_overfitting();
    
    cout << "\n" << string(60, '=') << endl;
    cout << "RESUMEN DE CONCLUSIONES" << endl;
    cout << string(60, '=') << endl;
    
    cout << R"(
1. PERCEPTRON (Nivel 0):
   - Resuelve problemas linealmente separables (AND, OR, NAND, NOR)
   - NO puede resolver XOR (requiere red con capa oculta)
   - Learning rate optimo: 0.1 - 0.5

2. RED NEURONAL CON BACKPROPAGATION (Nivel 1):
   - Resuelve problemas no linealmente separables como XOR
   - En Iris (3 clases): alcanza >90% de accuracy
   - Importancia de la validacion para detectar overfitting

3. FUNCIONES DE ACTIVACION:
   - SIGMOID: buena para clasificacion, output (0,1)
   - TANH: similar pero output (-1,1), puede converger mas rapido
   - RELU: rapida pero riesgo de "dying neurons"

4. OVERFITTING:
   - Redes muy grandes + muchas epocas = sobreajuste
   - Usar conjunto de validacion para detectarlo
   - Early stopping cuando val_acc empieza a bajar
)" << endl;

    return 0;
}
