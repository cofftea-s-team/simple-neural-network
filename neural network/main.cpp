// neural network.cpp : Ten plik zawiera funkcję „main”. W nim rozpoczyna się i kończy wykonywanie programu.

#include <iostream>
#include <iomanip>
#include <fstream>

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;

#include "network.hpp"
#include "activations.hpp"
#include "losses.hpp"
#include "optimizers.hpp"

#include <Windows.h>
#include "progress_bar.h"

using namespace network;
using namespace pipeline;
#include <deque>

template <size_t _Batch, size_t M>
inline double accuracy(const tensor<_Batch, M>& _Predictions, const tensor<_Batch, M>& _Targets) {
	double correct = 0;
	for (size_t i = 0; i < _Batch; ++i) {
		auto max = _Predictions[i][0];
		size_t max_index = 0;
		for (size_t j = 1; j < M; ++j) {
			if (_Predictions[i][j] > max) {
				max = _Predictions[i][j];
				max_index = j;
			}
		}
		if (_Targets[i][max_index] == 1) {
			++correct;
		}
	}
	return correct / _Batch;
}

int main()
{
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(0);
	cout << std::fixed << std::setprecision(5);
	auto obj = new nnetwork<
		linear<784, 16>,
		relu,
		linear<16, 10>,
		softmax
	>();
	auto& net = *obj;
	sizeof(net);

	constexpr int batch = 256;
	constexpr int _Epoch = 40;
	
	for (int epoch = 0; epoch < _Epoch; ++epoch) {
		progress_bar bar(32);
		ifstream x_file("train_x.txt");
		ifstream y_file("train_y.txt");
		while (bar) {
			tensor<batch, 784> train_data;
			tensor<batch, 10> train_results(0);
			for (int i = 0; i < batch; ++i) {
				for (int j = 0; j < 784; ++j) {
					x_file >> train_data[i][j];
				}
				int x;
				y_file >> x;
				train_results[i][x] = 1;
			}

			auto preds = net.forward(train_data);
			net.backward<xentropy_loss, sgd>(train_results);
			bar.update([](double x, double y, double z) {
				cout << "loss: " << x << ", lr: " << y << ", acc: " << z;
				},
				xentropy.compute(preds, train_results), sgd::current_lr, accuracy(preds, train_results)
					);
		}
		x_file.close();
		y_file.close();
	}

	ifstream x_valid("train_x.txt");
	ifstream y_valid("train_y.txt");
	tensor<10000, 784> x_valid_tensor;
	tensor<10000, 10> y_valid_tensor;
	for (int i = 0; i < 10000; ++i) {
		for (int j = 0; j < 784; ++j) {
			x_valid >> x_valid_tensor[i][j];
		}
		int y;
		y_valid >> y;
		y_valid_tensor[i][y] = 1;
	}
	x_valid.close();
	y_valid.close();

	auto preds = net.forward(x_valid_tensor);
	cout << "accuracy: " << accuracy(preds, y_valid_tensor) << endl;

}
