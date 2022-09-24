#pragma once
 
#include <utility.h>
#include <linkedlist.h>
#include "utils.hpp"

namespace network {
	using any_tensor = _Empty;
	struct sgd {
		template <size_t N, size_t M>
		inline static void update(tensor<N, M>& _Weights, const tensor<N, M>& _Ders) {
			if (decay != 0 && iters % layer_count == 0) {
				current_lr = lr * (1. / (1. + decay * (2 * iters / layer_count)));
			}
			if (momentum == 0) {
				auto it = _Weights.begin();
				auto it2 = _Ders.begin();
				for (; it != _Weights.end(); ++it, ++it2) {
					*it -= *it2 * lr;
				}
				++iters;
				return;
			}
			if (tensor_momentums.size() != layer_count) {
				auto t = new tensor<N, M>(0);
				tensor_momentums.push_front(reinterpret_cast<any_tensor*>(t));
			}
			auto _M_it = reinterpret_cast<tensor<N, M>*>(tensor_momentums.front())->begin();
			auto _W_it = _Weights.begin();
			auto _D_it = _Ders.begin();

			for (; _W_it != _Weights.end(); ++_W_it, ++_D_it, ++_M_it) {
				*_W_it -= current_lr * *_D_it + momentum * *_M_it;
				*_M_it = *_D_it;
			}

			auto front_ptr = tensor_momentums.front();
			tensor_momentums.pop_front();

			tensor_momentums.push_back(front_ptr);
			

			++iters;
		}
		
		static size_t iters;
		static double current_lr;
		static double lr;
		static double decay;
		static double momentum;
		static size_t layer_count;
		static linkedlist<any_tensor*> tensor_momentums;
	};
	inline double sgd::lr = 0.1;
	inline double sgd::decay = 0.000001;
	inline double sgd::momentum = 0.7;
	inline size_t sgd::iters = 0;
	inline size_t sgd::layer_count = 0;
	inline double sgd::current_lr = 0;
	inline linkedlist<any_tensor*> sgd::tensor_momentums;
}