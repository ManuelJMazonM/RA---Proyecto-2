// main.cpp
#include <iostream>
#include "perceptron.hpp"
using namespace std;

int main() {
    int num_inputs;
    
    // Pedir número de entradas
    cout << "¿Cuántas entradas tendrá el perceptrón? ";
    cin >> num_inputs;
    
    // Crear perceptrón
    Perceptron p(num_inputs);
    
    // Mostrar pesos generados
    cout << "\nPesos generados:" << endl;
    for (int i = 0; i < num_inputs; ++i) {
        cout << "  w[" << i << "] = " << p.weights[i] << endl;
    }
    cout << "  bias = " << p.bias << endl;
    
    // Pedir valores de entrada
    vector<double> inputs(num_inputs);
    cout << "\nIntroduce los " << num_inputs << " valores de entrada:" << endl;
    for (int i = 0; i < num_inputs; ++i) {
        cout << "  x[" << i << "] = ";
        cin >> inputs[i];
    }
    
    // Elegir función de activación
    cout << "\nElige función de activación:" << endl;
    cout << "  1. SIGMOID" << endl;
    cout << "  2. TANH" << endl;
    cout << "  3. RELU" << endl;
    cout << "  4. STEP" << endl;
    cout << "Opción: ";
    
    int opcion;
    cin >> opcion;
    
    ActivationType activation;
    switch (opcion) {
        case 1: activation = ActivationType::SIGMOID; break;
        case 2: activation = ActivationType::TANH; break;
        case 3: activation = ActivationType::RELU; break;
        case 4: activation = ActivationType::STEP; break;
        default: activation = ActivationType::SIGMOID;
    }
    
    // Calcular y mostrar resultado
    double output = p.predict(inputs, activation);
    cout << "\nResultado: " << output << endl;
    
    return 0;
}