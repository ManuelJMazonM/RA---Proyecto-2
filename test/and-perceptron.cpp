#include "neuralnetwork.hpp"

#include <iostream>

using std::cout;
using std::endl;

int main()
{
  vector<vector<double>> dataset_in = { {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0} };
  vector<double> dataset_out = { 0.0, 0.0, 0.0, 1.0 };
  Perceptron p(2);
  Trainer t;

  t.train(p, dataset_in, dataset_out);
  double hits = 0;
  for(size_t i=0; i < dataset_out.size(); i++)
  {
    double result = p.predict(dataset_in[i], ActivationType::STEP);
    if((dataset_out[i] > 0) == (result > 0)) hits++;
  }
  cout << "Accuracy for AND gate: " << (hits / (double)dataset_out.size()) * 100.0 << endl;

  return 0;
}
