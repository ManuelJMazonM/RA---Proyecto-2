#include "common.hpp"
#include <iostream>

int main(){
	std::cout << "--- Probando carga de Dataset ---" << std::endl;
    
    // Intenta cargar el iris.csv (ajusta los parámetros según el CSV que bajes)
    // Supongamos: target en col 4, con cabecera, separador coma, 20% validación
    Dataset ds = loadDataset("iris.csv", 4, ',', true, 0.2);

    std::cout << "Clases detectadas: " << ds.num_classes << std::endl;
    std::cout << "Muestras de entrenamiento: " << ds.train_inputs.size() << std::endl;
    std::cout << "Muestras de validacion: " << ds.val_inputs.size() << std::endl;

    // Comprobar el diccionario de etiquetas
    std::cout << "\nDiccionario de clases:" << std::endl;
    for (auto const& [id, nombre] : ds.id_to_label) {
        std::cout << "  ID " << id << " -> " << nombre << std::endl;
    }

    if (!ds.train_inputs.empty()) {
        std::cout << "\nEjemplo primera muestra (Inputs): ";
        for(double val : ds.train_inputs[0]) std::cout << val << " ";
        
        int class_id = ds.train_targets[0];
        std::cout << "\nEjemplo primera muestra (Target One-Hot): ";
		std::cout << " (Especie: " << ds.id_to_label[class_id] << ")" << std::endl;    }

    return 0;
}
