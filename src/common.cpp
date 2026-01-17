#include "common.hpp"
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <map>

namespace Activations {
	//Funciones de Activación
	double apply(double s, ActivationType type){
		switch (type){
		case ActivationType::SIGMOID:
			return 1.0 / (1.0 + std::exp(-s));
		case ActivationType::TANH:
			return std::tanh(s);
		case ActivationType::RELU:
			return std::max(0.0, s); 
		case ActivationType::STEP:
			return (s >= 0.0) ? 1.0 : -1.0;
		default: 
			return s;
		}
	}


	//Derivadas
	double applyDerivative(double f_s, ActivationType type){
		switch (type){
		case ActivationType::SIGMOID:
			return f_s * (1.0 - f_s);
		case ActivationType::TANH:
			return 1.0 - (f_s * f_s);
		case ActivationType::RELU:
			return (f_s > 0.0) ? 1.0 : 0.0;
		case ActivationType::STEP:
			return 0.0; //NO se usa en BackProp, así que no necesitamos la derivada
		default: 
			return 1.0;
		}	
	}
<<<<<<< Updated upstream
}
=======
}

Dataset loadDataset(const std::string& filename, int target_col, char delimiter, bool has_header, double val_split){
	Dataset ds;
	std::vector<std::vector<std::string>> raw_data;
	std::ifstream file(filename);
	std::string line;

	
	if (has_header){
		std::getline(file, line);
	}

	while(std::getline(file, line)){
		std::stringstream ss(line);
		std::string celda;
		std::vector<std::string> row;
		
		while(std::getline(ss, celda, delimiter)){
			row.push_back(celda);
		}
		if(!row.empty()){
			raw_data.push_back(row); 
		}
	}

	std::shuffle(raw_data.begin(), raw_data.end(), std::mt19937(std::random_device()()));

	std::map<std::string, int> label_to_id;
	int class_counter = 0;
	for (const auto& row : raw_data){
		std::string label = row[target_col];
		if (label_to_id.find(label) == label_to_id.end()){
			label_to_id[label] = class_counter++;
		}
	}
	ds.num_classes = class_counter;

	for (const auto& row : raw_data){
		std::vector<double> inputs;
		for (int i=0; i < row.size(); ++i){
			if (i==target_col){
				continue
			}
			inputs.push_back(std::stod(row[i]));
		}

		std::vector<double> target_vec(ds.num_classes, 0.0);
		target_vec[label_to_id[row[target_col]]] = 1.0;

		if(ds.val_inputs.size() < raw_data.size() * val_split){
			ds.val_inputs.push_back(inputs);
			ds.val_targets.push_back(target_vec);
		}
		else{
			ds.train_inputs.push_back(inputs);
			ds.train_targets.push_back(target_vec);
		}
	}

	return ds;
}
>>>>>>> Stashed changes
