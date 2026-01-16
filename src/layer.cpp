#include "layer.hpp"
#include "common.hpp"

Layer::Layer(int num_neurons, int inputs_per_neuron) : last_outputs(num_neurons) {
	for (int i=0; i < num_neurons; ++i){
		neurons.push_back(Perceptron(inputs_per_neuron));
	}
}

std::vector<double> Layer::forward(const std::vector<double>& inputs, const ActivationType activation){
	last_inputs = inputs;
	last_outputs.clear();

	for (size_t i=0; i < neurons.size(); ++i) {
		double output = neurons[i].predict(inputs, activation);
		last_outputs.push_back(output);
	}

	return last_outputs;
}
