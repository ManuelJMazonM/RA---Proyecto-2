#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "neuralnetwork.hpp"
#include <vector>

using std::vector;

class Trainer{
private:
	double learning_rate;

	vector<vector<double>> computeDeltas(NeuralNetwork& nn, const vector<double>& target) const;
	void applyGradients(NeuralNetwork& nn, const vector<vector<double>>& deltas, double eta) const;

public:
	Trainer(double lr = 0.1) : learning_rate(lr){}

	double train(
      NeuralNetwork& nn,
      const vector<vector<double>>& data,
      const vector<int>& targets,
      const int epochs) const;

  double train(
      Perceptron& perceptron,
      const vector<vector<double>>& training_inputs,
      const vector<double>& training_outputs,
      double learning_rate = 0.1,
      int max_epochs = 1000,
      ActivationType activation = ActivationType::STEP) const;
  
  double test_acc(
      NeuralNetwork& network,
      const vector<vector<double>>& test_inputs,
      const vector<double>& test_outputs) const;
};

#endif
