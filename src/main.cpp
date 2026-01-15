int main(){
	int option;
	std::cout << "Hola, estás a punto de entrenar un bot. ¿Qué algoritmo de aprendizaje quieres usar?" << std::endl;
	std::cout << "1. Perceptón (PLA)" << std::endl << "2. Neural Network (Backpropagation)" << std::endl;
	
	do{
		std::cout << "Seleccione (1) o (2):" << std::endl; 
		cin >> option;

		std::string datasetPath;
		std::cout << "¡Buena elección! ¿Y con qué dataset quieres entrenar a este bot? (Ej.: datasetXOR.csv)" << std::endl;
		std::cin >> datasetPath;


		//CARGAR EL DATASET Y "TRANSFORMARLO" COMO SEA NECESARIO
		// call loadDataset()

		if (option == 1){
			//Establecer datos de entrada 
			//Procesar dataset
			//Llamar al algoritmo de PLA del perceptrón
		}

		else if (option == 2){
			//
		}

		else{
			std::cout << "Opción inválida." << std::endl;
		}
	}while (option!=1 && option!=2);
	

	return 0;
}