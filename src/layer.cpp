#include "layer.hpp"
#include "common.hpp"

Layer::Layer(int num_neurons, int inputs_per_neuron){
	for (int i=0; i < num_neurons; ++i){
		neurons.push_back(Perceptron(inputs_per_neuron));
	}
}

std::vector<double> Layer::forward(const std::vector<double>& inputs, ActivationType activation){
	this->last_inputs = inputs;  //Guardamos los valores de entrada para el entrenamiento

	std::vector<double> outputs;
	this->last_outputs.clear();

	for (size_t i=0; i < neurons.size(); ++i){
		//REVISAR función PREDICT  en perceptrón 
		double output = neurons[i].predict(inputs, activation);

		ouputs.push_back(output);
		this->last_outputs.push_back(output);
	}

	return outputs;
}