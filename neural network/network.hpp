#pragma once

#ifndef __NETWORK_HPP__
#define __NETWORK_HPP__

#include <utility.h>
#include <matrix.h>
#include <vector.h>

#include "utils.hpp"
#include "linear.hpp"
#include "losses.hpp"

namespace network {

	template <class... _Args>
	class nnetwork {
	public:
		using linears = typename evens<std::tuple<_Args...>>::type;
		using activations = typename odds<std::tuple<_Args...>>::type;

		template <size_t _Batch, size_t M>
		auto forward(const tensor<_Batch, M>& _Input) {
			for (int i = 0; i < _Forward_results.size(); ++i) {
				if ((i & 1) == 0) delete _Forward_results[i];
			}
			_Forward_results.clear();
			_Forward_results.push_back(reinterpret_cast<any_tensor*>(new auto(_Input)));
			for_each([&]<size_t N1, size_t M1>(const linear<N1, M1>& _Layer, auto&& _Fn) -> void {
				auto& t = *reinterpret_cast<tensor<_Batch, N1>*>(_Forward_results.back());
				_Forward_results.emplace_back(reinterpret_cast<any_tensor*>(&t));
				_Forward_results.emplace_back(reinterpret_cast<any_tensor*>(new auto(_Fn.forward_apply(_Layer.forward(t)))));
			});

			return return_last<_Batch>(std::get<std::tuple_size_v<linears> - 1>(_Layers));
		}
		template <class _Loss_fn, class _Optimizer_fn, size_t _Batch, size_t M>
		auto backward(const tensor<_Batch, M>& _Output) {
			auto& _Preds = *reinterpret_cast<tensor<_Batch, M>*>(_Forward_results.back());
			_Forward_results.emplace_back(reinterpret_cast<any_tensor*>(new auto(_Loss_fn::backward(_Preds, _Output))));
			for_each<false, std::tuple_size_v<linears> - 1>([&]<size_t N1, size_t M1>(linear<N1, M1>& _Layer, auto&& _Fn) -> void {
				auto& err = *reinterpret_cast<tensor<_Batch, M1>*>(_Forward_results.back()); // error
				_Forward_results.pop_back();
				auto& fwd = *reinterpret_cast<tensor<_Batch, M1>*>(_Forward_results.back()); // wyjscie z aktywacyjnej
				_Forward_results.pop_back();

				auto& inputs = *reinterpret_cast<tensor<_Batch, N1>*>(_Forward_results.back()); // wejscie do layera
				_Forward_results.pop_back();
				
				auto fn_bwd = _Fn.backward_apply(fwd);
				auto error = fn_bwd.mul(err);
				
				_Forward_results.push_back(reinterpret_cast<any_tensor*>(new auto(_Layer.backward(error))));
				_Layer.update<_Optimizer_fn>(inputs, error);

				delete &err;
				delete &fwd;
			});
			for (auto&& ptr : _Forward_results) delete ptr;
			_Forward_results.clear();
		}

	private:
		template <size_t _Batch, size_t N, size_t M>
		inline auto& return_last(linear<N, M>) {
			return *reinterpret_cast<tensor<_Batch, M>*>(_Forward_results.back());
		}

		template<bool _Increment = true, int I = 0, class _Lambda>
		inline typename std::enable_if<I == std::tuple_size_v<linears> || I == -1, void>::type
			for_each(_Lambda&& _Func)
		{ }

		template<bool _Increment = true, int I = 0, class _Lambda>
		inline typename std::enable_if<I < std::tuple_size_v<linears> && I >= 0, void>::type
			for_each(_Lambda&& _Func)
		{
			auto& layer = std::get<I>(_Layers);
			auto& fn = std::get<I>(_Fns);

			_Func(layer, fn);
			
			if constexpr (_Increment) {
				for_each<true, I + 1>(_Func);
			} else {
				for_each<false, I - 1>(_Func);
			}
		}
		using any_tensor = _Empty;
		
		linears _Layers;
		activations _Fns;
		vector<any_tensor*> _Forward_results;
	//	vector<any_tensor*> _Backward_results;
	};
}


#endif // 
