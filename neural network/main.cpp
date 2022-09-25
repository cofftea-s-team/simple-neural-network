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

bool confirm_save() {
	cout << "save data to file? (Y/N) ";
	char input;
	std::cin >> input;
	return input == 'Y' || input == 'y';
}


int main()
{
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(0);
	cout << std::fixed << std::setprecision(2);
	auto obj = new nnetwork<
		linear<784, 512>,
		relu,
		linear<512, 256>,
		relu,
		linear<256, 10>,
		softmax
	>();
	auto& net = *obj;
	sizeof(net);
	//nnfile().load(net, "netdata.cofftea");
	//net.reset_layers<2>();
	//net.freeze_layers<0>();

	ifstream x_file("train_x.txt");
	ifstream y_file("train_y.txt");

	constexpr int batch = 128;
	constexpr int _Epochs = 20;
	
	for (int epoch = 0; epoch < _Epochs; ++epoch) {
		progress_bar bar(10);
		ifstream x_file("train_x.txt");
		ifstream y_file("train_y.txt");
		int _Iter = 0;
		size_t _Iters_total = 20;
		double _Acc_sum = 0;
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
			_Acc_sum += accuracy(preds, train_results);
			if (_Iter % _Iters_total == 19) {
				bar.update([](double x, double y, double z, int e) {
					cout << std::setprecision(5) << "loss: " << x
						<< ", lr: " << y << std::setprecision(2)
						<< ", acc: " << z * 100 << "%, epochs: " << e;
					},
					xentropy.compute(preds, train_results),
					sgd::current_lr,
					_Acc_sum / _Iters_total,
					epoch
				);
				_Acc_sum = 0;
			}
			++_Iter;
		}
		std::cout << endl;
		x_file.close();
		y_file.close();
	}
	
	ifstream x_valid("train_x.txt");
	ifstream y_valid("train_y.txt");
	tensor<1000, 784> x_valid_tensor;
	tensor<1000, 10> y_valid_tensor;
	for (int i = 0; i < 1000; ++i) {
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
