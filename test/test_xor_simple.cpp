#include "neuralnetwork.hpp"
#include <iostream>
using namespace std;

int main() {
    Dataset ds;
    ds.train_inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
    ds.train_targets = {0, 1, 1, 0};
    ds.num_classes = 2;
    
    cout << "Creando red 2-2-2..." << endl;
    vector<int> topo = {2, 2, 2};
    NeuralNetwork nn(topo, ActivationType::SIGMOID);
    
    cout << "Capas creadas: " << nn.layers.size() << endl;
    for (size_t i = 0; i < nn.layers.size(); i++) {
        cout << "  Capa " << i << ": " << nn.layers[i].neurons.size() << " neuronas" << endl;
        if (!nn.layers[i].neurons.empty()) {
            cout << "    Pesos por neurona: " << nn.layers[i].neurons[0].weights.size() << endl;
        }
    }
    
    cout << "\nProbando predict..." << endl;
    int result = nn.predict(ds.train_inputs[0]);
    cout << "Resultado: " << result << endl;
    
    cout << "\nCreando trainer..." << endl;
    Trainer trainer(0.5);
    
    cout << "Entrenando 10 epochs..." << endl;
    trainer.train(nn, ds, 10);
    
    cout << "OK!" << endl;
    return 0;
}
