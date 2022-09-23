#pragma once
 
#include <utility.h>
#include "utils.hpp"

namespace network {
	struct sgd {
		template <size_t N, size_t M>
		inline static void update(tensor<N, M>& _Weights, const tensor<N, M>& _Ders) {
			auto it = _Weights.begin();
			auto it2 = _Ders.begin();
			for (; it != _Weights.end(); ++it, ++it2) {
				*it -= *it2 * learning_rate;
			}
		}

		static constexpr double learning_rate = 0.05;
	};
}