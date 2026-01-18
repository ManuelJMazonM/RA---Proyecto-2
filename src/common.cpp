#include "common.hpp"
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <map>
#include <limits>

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


	size_t num_samples = raw_data.size();
	size_t num_features = raw_data[0].size() -1;

	std::vector<std::vector<double>> all_inputs(num_samples, std::vector<double>(num_features));
	std::vector<int> all_targets(num_samples);

	std::vector<double> min_vals(num_features, std::numeric_limits<double>::max());
	std::vector<double> max_vals(num_features, std::numeric_limits<double>::lowest());

	for (size_t i = 0; i < num_samples; ++i) {
        size_t feat_idx = 0;
        for (size_t j = 0; j < raw_data[i].size(); ++j) {
            if (j == target_col) {
                all_targets[i] = label_to_id[raw_data[i][j]];
                continue;
            }
            double val = std::stod(raw_data[i][j]);
            all_inputs[i][feat_idx] = val;
            
            if (val < min_vals[feat_idx]) min_vals[feat_idx] = val;
            if (val > max_vals[feat_idx]) max_vals[feat_idx] = val;
            feat_idx++;
        }
    }

    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            double range = max_vals[j] - min_vals[j];
            if (range > 0.0) {
                all_inputs[i][j] = (all_inputs[i][j] - min_vals[j]) / range;
            }
        }
    }

    size_t val_limit = static_cast<size_t>(num_samples * val_split);
    for (size_t i = 0; i < num_samples; ++i) {
        if (ds.val_inputs.size() < val_limit) {
            ds.val_inputs.push_back(all_inputs[i]);
            ds.val_targets.push_back(all_targets[i]);
        } else {
            ds.train_inputs.push_back(all_inputs[i]);
            ds.train_targets.push_back(all_targets[i]);
        }
    }

	return ds;
}
