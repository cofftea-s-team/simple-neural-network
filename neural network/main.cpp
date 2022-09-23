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


int main()
{
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(0);
	auto obj = new nnetwork<
		linear<784, 16>,
		relu,
		linear<16, 10>,
		softmax
	>();
	auto& net = *obj;
	sizeof(net);

	ifstream x_file("train_x.txt");
	ifstream y_file("train_y.txt");

	constexpr int batch = 32;
	
	progress_bar bar(64);
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
		for (int i = 0; i < 4500; ++i) {
			auto preds = net.forward(train_data);
			net.backward<xentropy_loss, sgd>(train_results);
		}
		
		auto preds = net.forward(train_data);
		net.backward<xentropy_loss, sgd>(train_results);
		bar.update([](double x) { cout << "loss: " << x; }, xentropy.compute(preds, train_results));
	}
	x_file.close();
	y_file.close();
	
	ifstream x_test_file("test_x.txt");
	ifstream y_test_file("test_y.txt");



	// provide the test
	ifstream test("5.txt");
	tensor<1, 784> data;
	for (int i = 0; i < 784; ++i) {
		test >> data[0][i];
	}
	net.forward(data).print();

}
