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
#include "nnfile.hpp"

using namespace network;
using namespace pipeline;
#include <deque>

template <size_t _Batch, size_t M>
inline double accuracy(const tensor<_Batch, M>& _Predictions, const tensor<_Batch, M>& _Targets) {
	double _Acc = 0;
	for (int i = 0; i < _Batch; ++i) {
		int _Max = 0;
		int _Max2 = 0;
		for (int j = 0; j < M; ++j) {
			if (_Predictions[i][j] > _Predictions[i][_Max]) {
				_Max = j;
			}
			if (_Targets[i][j] > _Targets[i][_Max2]) {
				_Max2 = j;
			}
		}
		if (_Max == _Max2) {
			++_Acc;
		}
	}
	return _Acc / _Batch;
}

bool confirm_save() {
	cout << "save data to file? (Y/N)" << endl;
	char input;
	std::cin >> input;
	return input == 'Y' || input == 'y';
}

int main()
{
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(0);
	cout << std::fixed << std::setprecision(5);
	auto obj = new nnetwork <
		linear<784, 128>,
		relu,
		linear<128, 48>,
		relu,
		linear<48, 10>,
		softmax
	>();
	auto& net = *obj;
	sizeof(net);
	nnfile().load(net, "netdata.cofftea");
	//net.reset_layers<2>();
	//net.freeze_layers<0, 1, 2>();
	ifstream x_file("train_x.txt");
	ifstream y_file("train_y.txt");

	constexpr int batch = 16;
	
	progress_bar bar(90);
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
		//for (int j = 0; j < 10; ++j) {
		//	ifstream test(char(j + '0') + std::string(".txt"));
		//	for (int i = 0; i < 784; ++i) {
		//		test >> train_data[j][i];
		//	}
		//	train_results[j][j] = 1;
		//	test.close();
		//}
		for (int i = 0; i < 64; ++i) {
			auto preds = net.forward(train_data);
			net.backward<xentropy_loss, sgd>(train_results);
		}
		
		auto preds = net.forward(train_data);
		//net.backward<xentropy_loss, sgd>(train_results);
		bar.update([](double x, double y, double z) { 
				cout << "loss: " << x << ", lr: " << y << ", acc: " << z; 
			}, 
			xentropy.compute(preds, train_results), 
			sgd::current_lr, 
			accuracy(preds, train_results)
		);
	}
	x_file.close();
	y_file.close();
	
	if (confirm_save()) {
		nnfile().save(net, "netdata.cofftea");
	}
	//return 1;

	// provide the test

	//nnfile().load(net, "netdata.cofftea");
	using std::string;
	tensor<10, 784> data;
	for (int j = 0; j < 10; ++j) {
		ifstream test(char(j + '0') + string(".txt"));
		for (int i = 0; i < 784; ++i) {
			test >> data[j][i];
		}
		test.close();
	}
	net.predict(data).print();

}
