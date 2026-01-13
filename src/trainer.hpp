#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "NeuralNetwork.hpp"
#include <vector>

using std::vector;
using std::cout;
using std::cerr;
using std::abs;


class Trainer{
private:
	double learning_rate;

	vector<vector<double>> computeDeltas(NeuralNetwork& nn, const vector<double>& target, ActivationType activation);
	void applyGradients(NeuralNetwork& nn, const vector<vector<double>>& deltas, double eta);

public:
	Trainer(double lr = 0.1) : learning_rate(lr){}

	void train(
      NeuralNetwork& nn,
      const vector<vector<double>>& data,
      const vector<vector<double>>& targets, 
			int epochs, ActivationType activation);

  bool train(
      Perceptron& perceptron,
      const vector<vector<double>>& training_inputs,
      const vector<double>& training_outputs,
      double learning_rate = 0.1,
      int max_epochs = 1000,
      ActivationType activation = ActivationType::STEP);
  
  double test_acc(
      NeuralNetwork& network,
      const vector<vector<double>>& test_inputs,
      const vector<double>& test_outputs);
};

#endif
