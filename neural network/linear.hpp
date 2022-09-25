#pragma once

#include <utility.h>
#include <matrix.h>
#include "utils.hpp"

namespace network {

	template <size_t N, size_t M>
	struct linear
	{
		inline linear()
			: _Biases(0) {
			fill_randn(_Weights);
		}
		template <size_t _Batch>
		inline auto forward(const tensor<_Batch, N>& _Inputs) const {
			return dot_a_b_add_c(_Inputs, _Weights, _Biases);
			//return _Inputs * _Weights + _Biases;
		}
		template <size_t _Batch>
		inline auto backward(const tensor<_Batch, M>& _Ders) {
			//return _Ders.dot_create_dyn(pipeline::transpose(_Weights));
			//return new auto(_Ders * pipeline::transpose(_Weights)); // default
			return dot_a_transpose_b(_Ders, _Weights);
		}
		template <class _Optimizer, size_t _Batch>
		inline void update(tensor<_Batch, N>& _Input, const tensor<_Batch, M>& _Ders) {
			if (_Freezed) return;
			//auto backwarded = pipeline::transpose(_Input) * _Ders;
			auto backwarded = dot_transpose_a_b(_Input, _Ders);
			_Optimizer::update(_Weights, move(backwarded)); // update weights
			_Optimizer::update(_Biases, _Sum_rows(_Ders)); // update biases
		}
		inline void reset() {
			fill_randn(_Weights);
			for (auto&& e : _Biases) e = 0;
		}
		inline void freeze(bool _Val = true) {
			_Freezed = _Val;
		}
	private:
		template <size_t _Batch>
		inline auto _Sum_rows(const tensor<_Batch, M>& _Ders) const {
			tensor<1, M> _Sum(0);
			for (int i = 0; i < _Batch; ++i) {
				for (int j = 0; j < M; ++j) {
					_Sum[0][j] += _Ders[i][j];
				}
			}
			return _Sum;
		}
		bool _Freezed = false;
		tensor<N, M> _Weights;
		tensor<1, M> _Biases;
	};
}