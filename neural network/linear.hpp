#pragma once

#include <utility.h>
#include <matrix.h>
#include "utils.hpp"

namespace network {

	template <size_t N, size_t M>
	struct linear
	{
		inline linear() {
			tensor<N, M> w;
			fill_randn(w);
			tensor<1, M> b(0);
			_Weights = alloc_in_cuda<N * M>(reinterpret_cast<double*>(w.data()));
			_Biases = alloc_in_cuda<M>(reinterpret_cast<double*>(b.data()));

			tensor<N, M> w2;
			copy_to_host<N* M>(_Weights, reinterpret_cast<double*>(w2.data()));
		}
		inline ~linear() {
			cudaFree(_Weights);
			_Weights = nullptr;
			cudaFree(_Biases);
			_Biases = nullptr; 
		}
		template <size_t _Batch>
		inline auto forward(const tensor<_Batch, N>& _Inputs) const {
			return dot_a_b_add_c<_Batch, N, M>(_Inputs, _Weights, _Biases);
		}
		template <size_t _Batch>
		inline auto backward(const tensor<_Batch, M>& _Ders) {
			return dot_a_transpose_b<_Batch, M, N>(_Ders, _Weights);
		}
		template <class _Optimizer, size_t _Batch>
		inline void update(tensor<_Batch, N>& _Input, const tensor<_Batch, M>& _Ders) {
			if (_Freezed) return;
			//auto backwarded = pipeline::transpose(_Input) * _Ders;
			auto backwarded = dot_transpose_a_b(_Input, _Ders);
			
			tensor<N, M> w;
			copy_to_host<N * M>(_Weights, reinterpret_cast<double*>(&w));
			_Optimizer::update(w, move(backwarded)); // update weights
			copy_to_cuda<N * M>(reinterpret_cast<double*>(w.data()), _Weights);

			tensor<1, M> b;
			copy_to_host<M>(_Biases, reinterpret_cast<double*>(b.data()));
			_Optimizer::update(b, _Sum_rows(_Ders)); // update biases
			copy_to_cuda<M>(reinterpret_cast<double*>(b.data()), _Biases);
		}
		inline void reset() {
			tensor<N, M> w;
			fill_randn(w);
			tensor<1, M> b(0);
			_Weights = alloc_in_cuda<N * M>(reinterpret_cast<double*>(w.data()));
			_Biases = alloc_in_cuda<M>(reinterpret_cast<double*>(b.data()));
		}
		inline void freeze(bool _Val = true) {
			_Freezed = _Val;
		}
		void check() const {
			check_cuda_ptr(_Weights);
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

		double* _Weights;
		double* _Biases;
	};
}