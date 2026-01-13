#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "NeuralNetwork.hpp"
#include <vector>

//TODO: trainer will probably only be function helpers to loop-train neural networks

class Trainer{
private:
	double learning_rate;

	std::vector<std::vector<double>> computeDeltas(NeuralNetwork& nn, const std::vector<double>& target, ActivationType activation);

	void applyGradients(NeuralNetwork& nn, const std::vector<std::vector<double>>& deltas, double eta);

public:
	Trainer(double lr = 0.1) : learning_rate(lr){}

	void trainPLA(Perceptron& p, const std::vector<std::vector<double>>& data, const std::vector<int>& labels);

	void trainBackprop(NeuralNetwork& nn, const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& targets, 
						int epochs, ActivationType activation);

};

#endif
