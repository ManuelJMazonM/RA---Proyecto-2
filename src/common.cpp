#include "common.hpp"
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <map>

namespace Activations {
	double apply(double s, ActivationType type) {
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

	double applyDerivative(double f_s, ActivationType type) {
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
}

Dataset loadDataset(const std::string& filename, size_t target_col, char delimiter, bool has_header, double val_split){
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

	std::set<std::string> unique_labels;
	for (const auto& row : raw_data){
		unique_labels.insert(row[target_col]);
	}

	std::map<std::string, int> label_to_id;
	int class_counter = 0;
	for (const auto& label : unique_labels){
		label_to_id[label] = class_counter;
		ds.id_to_label[class_counter] = label;
		class_counter++;
	}
	ds.num_classes = class_counter;

	std::shuffle(raw_data.begin(), raw_data.end(), std::mt19937(std::random_device()()));

	for (const auto& row : raw_data){
		std::vector<double> inputs;
		for (size_t i=0; i < row.size(); ++i){
			if (i==target_col){
				continue;
			}
			inputs.push_back(std::stod(row[i]));
		}

		int class_id = label_to_id[row[target_col]];

		if(ds.val_inputs.size() < raw_data.size() * val_split){
			ds.val_inputs.push_back(inputs);
			ds.val_targets.push_back(class_id);
		} else{
			ds.train_inputs.push_back(inputs);
			ds.train_targets.push_back(class_id);
		}
	}

	return ds;
}

