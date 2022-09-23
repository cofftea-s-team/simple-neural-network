#pragma once

#include <utility.h>
#include <matrix.h>
#include "utils.hpp"

namespace network {

	template <size_t N, size_t M>
	struct linear
	{
		inline linear()
			: biases(0) {
			fill_randn(weights);
		}
		template <size_t _Batch>
		inline auto forward(const tensor<_Batch, N>& _Inputs) const {
			return _Inputs * weights + biases;
		}
		template <size_t _Batch>
		inline auto backward(const tensor<_Batch, M>& _Ders) {
			return _Ders * pipeline::transpose(weights);
		}
		template <class _Optimizer, size_t _Batch>
		inline void update(tensor<_Batch, N>& _Input, const tensor<_Batch, M>& _Ders) {
			auto backwarded = pipeline::transpose(_Input) * _Ders;
			_Optimizer::update(weights, move(backwarded)); // update weights
			_Optimizer::update(biases, _Sum_rows(_Ders)); // update biases
		}
	private:
		template <size_t _Batch>
		inline auto _Sum_rows(const tensor<_Batch, M>& _Ders) const {
			tensor<1, M> _Sum(0);
			for (int i = 0; i < _Batch; ++i) {
				for (int j = 0; j < M; ++j) {
					_Sum[0][i] += _Ders[i][j];
				}
			}
			return _Sum;
		}

		tensor<N, M> weights;
		tensor<1, M> biases;
	};
}