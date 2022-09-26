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
#include "nnfile.hpp"
#include "progress_bar.h"

using namespace network;
using namespace pipeline;


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

using nn = nnetwork<
	linear<784, 512>,
	relu,
	linear<512, 256>,
	relu,
	linear<256, 128>,
	relu,
	linear<128, 10>,
	softmax
>;


inline void save_tmp(nn& net) {
	nnfile().save(net, "nettmp.cofftea");
}

void test_coffee_data(nn& net) {
	tensor<10, 784> _Input;
	for (int i = 0; i < 10; ++i) {
		ifstream file((char(i + '0') + std::string(".txt")));

		for (int j = 0; j < 784; ++j) {
			file >> _Input[i][j];
		}
		file.close();
	}
	net.predict(_Input).print();
}

void make_tests(nn& net) {
	ifstream x_valid("test_x.txt");
	ifstream y_valid("test_y.txt");
	tensor<10000, 784> x_valid_tensor;
	tensor<10000, 10> y_valid_tensor(0);
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
void train(nn& net) {
	ifstream x_file("train_x.txt");
	ifstream y_file("train_y.txt");

	constexpr int batch = 512;
	constexpr int _Epochs = 10;

	for (int epoch = 0; epoch < _Epochs; ++epoch) {
		progress_bar bar(9);
		ifstream x_file("train_x.txt");
		ifstream y_file("train_y.txt");
		int _Iter = 0;
		size_t _Iters_total = 10;
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
			net.backward<xentropy_loss, adam>(train_results);
			_Acc_sum += accuracy(preds, train_results);
			if (_Iter % _Iters_total == _Iters_total - 1) {
				bar.update([](double x, double y, double z, int e) {
					cout << std::setprecision(5) << "loss: " << x
						<< ", lr: " << y << std::setprecision(2)
						<< ", acc: " << z * 100 << "%, epochs: " << e;
					},
					xentropy.compute(preds, train_results),
						adam::current_lr,
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
		save_tmp(net);
	}
}
void train_siedem(nn& net) {
	net.freeze_layers<0, 1, 2>();

	tensor<1, 784> siedem_iks;
	tensor<1, 10> siedem_prawda(0);
	siedem_prawda[0][7] = 1;
	ifstream file("7.txt");
	for (int i = 0; i < 784; ++i) file >> siedem_iks[0][i];
	progress_bar bar(20);
	adam::lr = 5e-6;
	for (int i = 0; bar; ++i) {
		auto przewidywany_wynik = net.forward(siedem_iks);
		net.backward<xentropy_loss, adam>(siedem_prawda);

		if (i % 45 == 0) bar.update();
	}
}



int main()
{
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(0);
	cout << std::fixed << std::setprecision(2);
	nn net;
	sizeof(net);
	//nnfile().load(net, "netdata.cofftea");
	//net.reset_layers<2>();
	//adam::lr = 0.001;
	//train_siedem(net);
	train(net);
	test_coffee_data(net);
	make_tests(net);


	if (confirm_save()) 
		nnfile().save(net, "netdata.cofftea");
	
}
