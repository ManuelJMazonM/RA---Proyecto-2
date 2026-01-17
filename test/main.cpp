#include <iostream>
#include <vector>
#include <string>
#include "common.hpp"
#include "trainer.hpp"
#include "neuralnetwork.hpp"
#include "perceptron.hpp"
using namespace std;

int main(){
	int option;
	string datasetPath;
	size_t targetCol;
	char delim;
	int hasHeader;
	double valSplit = 0.2;

	cout << "=== BIENVENIDO AL ENTRENADOR DE BOTS 2026 ===" << endl;
    cout << "1. Perceptron (PLA) - Clasificacion Binaria" << endl;
    cout << "2. Neural Network (Backpropagation) - Multiclase" << endl;

    do {
        cout << "\nSeleccione algoritmo (1 o 2): ";
        cin >> option;
    } while (option != 1 && option != 2);

    valSplit = (option == 1) ? 0.0 : 0.2; //Si se elige el perceptrón no se aplica la validación

    // --- CONFIGURACIÓN DEL DATASET ---
    cout << "\n--- Configuracion del Dataset ---" << endl;
    cout << "Ruta del archivo (ej: data/iris.csv): ";
    cin >> datasetPath;
    cout << "Indice de la columna objetivo (empezando en 0): ";
    cin >> targetCol;
    cout << "Delimitador (ej: , o ;): ";
    cin >> delim;
    cout << "¿Tiene cabecera? (1=Si, 0=No): ";
    cin >> hasHeader;

    Dataset ds = loadDataset(datasetPath, targetCol, delim, (hasHeader == 1), valSplit);

    if (ds.train_inputs.empty()) {
        cerr << "Error critico: No se han podido cargar datos." << endl;
        return 1;
    }
    
    if (option == 1 && ds.num_classes > 2) {
        cout << "\n[¡CUIDADO!] Has seleccionado Perceptron pero el dataset tiene " << ds.num_classes << " clases." << endl;
        cout << "El Perceptron simple solo funciona para clasificacion binaria (2 clases)." << endl;
        cout << "Deseas continuar asumiendo que es Clase 0 vs el resto? (1=Si, 0=No): ";
        int confirm;
        cin >> confirm;
        if (confirm == 0) return 0; // Abortar para que elijan Neural Network
    }

	Trainer trainer(0.1);

	// --- EJECUCIÓN ---
    if (option == 1) {
        cout << "\n> Configurando Perceptron..." << endl;
        int epochs;
        cout << "Numero de epocas: ";
        cin >> epochs;

        // Inicializar perceptron con el numero de entradas detectado
        size_t num_inputs = ds.train_inputs[0].size();
        Perceptron p(num_inputs);

        cout << "Entrenando..." << endl;
        double acc = trainer.train(p, ds.train_inputs, ds.train_targets, 0.1, epochs, ActivationType::STEP);
        cout << "\nRESULTADO FINAL - Accuracy Test: " << acc << "%" << endl;

    } else if (option == 2) {
        cout << "\n> Configurando Red Neuronal..." << endl;
        int epochs, hidden_neurons;
        cout << "Numero de epocas: ";
        cin >> epochs;
        cout << "Neuronas en la capa oculta: ";
        cin >> hidden_neurons;

        // Definir topología: [Entradas, Oculta, Salidas]
        size_t num_inputs = ds.train_inputs[0].size();
        vector<int> topology = {(int)num_inputs, hidden_neurons, ds.num_classes};
        
        NeuralNetwork nn(topology, ActivationType::SIGMOID);

        cout << "Entrenando con validacion..." << endl;
        // Esta funcion ahora usa tu validacion interna cada 10 epocas
        trainer.train(nn, ds, epochs);
    }

    cout << "\nProceso finalizado. ¡Suerte con los experimentos!" << endl;
	return 0;
}