#pragma once
 
#include <utility.h>
#include <linkedlist.h>
#include "utils.hpp"

namespace cuda_network {
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
				*_M_it = -current_lr * *_D_it + momentum * *_M_it;
				*_W_it += *_M_it;
			}

			auto front_ptr = tensor_momentums.front();
			tensor_momentums.pop_front();

			tensor_momentums.push_back(front_ptr);
			

			++iters;
		}

		static void reset() {
			lr = 1;
			momentum = 0.9;
			iters = 0;
			current_lr = 0;
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
	inline double sgd::decay = 5e-3;
	inline double sgd::momentum = 0.9;
	inline size_t sgd::iters = 0;
	inline size_t sgd::layer_count = 0;
	inline double sgd::current_lr = 0;
	inline linkedlist<any_tensor*> sgd::tensor_momentums;

	struct rmsprop {
		template <size_t N, size_t M>
		inline static void update(tensor<N, M>& _Weights, const tensor<N, M>& _Ders) {
			if (decay != 0 && iters % layer_count == 0) {
				current_lr = lr * (1. / (1. + decay * (2 * iters / layer_count)));
			}
			if (tensor_cache.size() != layer_count) {
				auto t = new tensor<N, M>(0);
				tensor_cache.push_front(reinterpret_cast<any_tensor*>(t));
			}
			auto _M_it = reinterpret_cast<tensor<N, M>*>(tensor_cache.front())->begin();
			auto _W_it = _Weights.begin();
			auto _D_it = _Ders.begin();

			for (; _W_it != _Weights.end(); ++_W_it, ++_D_it, ++_M_it) {
				*_M_it = rho * *_M_it + (1 - rho) * *_D_it * *_D_it;
				*_W_it += -current_lr * *_D_it / (sqrt(*_M_it) + epsilon);
			}

			auto front_ptr = tensor_cache.front();
			tensor_cache.pop_front();

			tensor_cache.push_back(front_ptr);
			++iters;
		}

		static void reset() {
			lr = 0.001;
			epsilon = 1e-7;
			rho = 0.9;
			iters = 0;
			current_lr = 0;
		}

		static size_t iters;
		static double current_lr;
		static double lr;
		static double decay;
		static double epsilon;
		static double rho;
		static size_t layer_count;
		static linkedlist<any_tensor*> tensor_cache;
	};
	inline double rmsprop::lr = 0.001;
	inline double rmsprop::decay = 1e-4;
	inline double rmsprop::epsilon = 1e-7;
	inline double rmsprop::rho = 0.9;
	inline size_t rmsprop::iters = 0;
	inline size_t rmsprop::layer_count = 0;
	inline double rmsprop::current_lr = 0;
	inline linkedlist<any_tensor*> rmsprop::tensor_cache;

	struct adam {
		template <size_t N, size_t M>
		inline static void update(tensor<N, M>& _Weights, const tensor<N, M>& _Ders) {
			if (decay != 0 && iters % layer_count == 0) {
				current_lr = lr * (1. / (1. + decay * (2 * iters / layer_count)));
			}
			if (tensor_cache.size() != layer_count) {
				tensor_cache.push_front(reinterpret_cast<any_tensor*>(new tensor<N, M>(0)));
				tensor_momentums.push_front(reinterpret_cast<any_tensor*>(new tensor<N, M>(0)));
			}
			
			auto _M_it = reinterpret_cast<tensor<N, M>*>(tensor_momentums.front())->begin();
			auto _V_it = reinterpret_cast<tensor<N, M>*>(tensor_cache.front())->begin();

			auto _W_it = _Weights.begin();
			auto _D_it = _Ders.begin();

			for (; _W_it != _Weights.end(); ++_W_it, ++_D_it, ++_M_it, ++_V_it) {
				*_M_it = beta1 * *_M_it + (1 - beta1) * *_D_it;
				*_V_it = beta2 * *_V_it + (1 - beta2) * *_D_it * *_D_it;

				auto _M_fixed = *_M_it / (1 - pow(beta1, iters + 1));
				auto _V_fixed = *_V_it / (1 - pow(beta2, iters + 1));

				*_W_it += -current_lr * _M_fixed / (sqrt(_V_fixed) + epsilon);
			}

			auto front_ptr = tensor_cache.front();
			tensor_cache.pop_front();
			tensor_cache.push_back(front_ptr);
			
			front_ptr = tensor_momentums.front();
			tensor_momentums.pop_front();
			tensor_momentums.push_back(front_ptr);
			++iters;
		}

		static void reset() {
			lr = 0.0001;
			epsilon = 1e-7;
			beta1 = 0.9;
			beta2 = 0.999;
			iters = 0;
			current_lr = 0;
		}

		static size_t iters;
		static double current_lr;
		static double lr;
		static double decay;
		static double epsilon;
		static double beta1;
		static double beta2;
		static size_t layer_count;
		static linkedlist<any_tensor*> tensor_cache;
		static linkedlist<any_tensor*> tensor_momentums;
	};
	inline double adam::lr = 1e-3;
	inline double adam::decay = 1e-4;
	inline double adam::epsilon = 1e-7;
	inline double adam::beta1 = 0.9;
	inline double adam::beta2 = 0.999;
	inline size_t adam::iters = 0;
	inline size_t adam::layer_count = 0;
	inline double adam::current_lr = 0;
	inline linkedlist<any_tensor*> adam::tensor_cache;
	inline linkedlist<any_tensor*> adam::tensor_momentums;
}