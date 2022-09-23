// neural network.cpp : Ten plik zawiera funkcję „main”. W nim rozpoczyna się i kończy wykonywanie programu.
//

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

int main()
{
	nnetwork<
		linear<784, 16>,
		relu,
		linear<16, 32>,
		relu,
		linear<32, 16>,
		relu,
		linear<16, 10>,
		sigmoid
	> net;
	
	sizeof(net);

	ifstream x_file("train_x.txt");
	ifstream y_file("train_x.txt");

	constexpr int batch = 8;
	
	progress_bar bar(6250);
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

		for (int i = 0; i < 1000; ++i) {
			auto preds = net.forward(train_data);
			net.backward<xentropy_loss, sgd>(train_results);
		}
		
		auto preds = net.forward(train_data);
		net.backward<xentropy_loss, sgd>(train_results);
		bar.update(xentropy.compute(preds, train_results));
	}
	
	


	
	


	
	

}
