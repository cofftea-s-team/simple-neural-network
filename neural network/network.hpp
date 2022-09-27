#pragma once

#ifndef __NETWORK_HPP__
#define __NETWORK_HPP__

#include <utility.h>
#include <matrix.h>
#include <vector.h>

#include "utils.hpp"
#include "linear.hpp"
#include "losses.hpp"
#include "optimizers.hpp"

namespace cuda_network {

	template <class... _Args>
	class nnetwork {
	public:
		using linears = typename evens<std::tuple<_Args...>>::type;
		using activations = typename odds<std::tuple<_Args...>>::type;
		friend class nnfile;

		constexpr nnetwork() {
			sgd::layer_count = sizeof...(_Args);
			adam::layer_count = sizeof...(_Args);
			adamw::layer_count = sizeof...(_Args);
		}

		template <size_t _Batch, size_t M>
		inline auto predict(const tensor<_Batch, M>& _Input) {
			auto _Preds = forward(_Input);
			_Preds.print();
			// clean up data for backward...
			_Clean_up_forward_data();

			return argmax(_Preds);
		}

		template <size_t _Batch, size_t M>
		inline auto forward(const tensor<_Batch, M>& _Input) {
			_Clean_up_forward_data();
			
			_Forward_results.push_back(reinterpret_cast<any_tensor*>(cuda_object_create<tensor<_Batch, M>>(_Input)));
			for_each([&]<size_t N1, size_t M1>(const linear<N1, M1>& _Layer, auto&& _Fn) -> void {
				auto& t = *reinterpret_cast<tensor<_Batch, N1>*>(_Forward_results.back());
				_Forward_results.emplace_back(reinterpret_cast<any_tensor*>(&t));
				auto fwd = _Layer.forward(t);
				_Fn.forward_apply(fwd);
				_Forward_results.emplace_back(reinterpret_cast<any_tensor*>(cuda_object_create<tensor<_Batch, M1>>(move(fwd))));
			});

			return _Return_last<_Batch>(std::get<std::tuple_size_v<linears> - 1>(_Layers));
		}
		
		template <class _Loss_fn, class _Optimizer_fn, size_t _Batch, size_t M>
		inline auto backward(const tensor<_Batch, M>& _Output) {
			auto& _Preds = *reinterpret_cast<tensor<_Batch, M>*>(_Forward_results.back());
			_Forward_results.emplace_back(reinterpret_cast<any_tensor*>(cuda_object_create<tensor<_Batch, M>>(_Loss_fn::backward(_Preds, _Output))));
			for_each<false, std::tuple_size_v<linears> - 1>([&]<size_t N1, size_t M1>(linear<N1, M1>& _Layer, auto&& _Fn) -> void {

				auto& err = *reinterpret_cast<tensor<_Batch, M1>*>(_Forward_results.back()); // error
				_Forward_results.pop_back();
				auto& fwd = *reinterpret_cast<tensor<_Batch, M1>*>(_Forward_results.back()); // wyjscie z aktywacyjnej
				_Forward_results.pop_back();

				auto& inputs = *reinterpret_cast<tensor<_Batch, N1>*>(_Forward_results.back()); // wejscie do layera
				_Forward_results.pop_back();
				
				_Fn.backward_apply(fwd);
				fwd.mul(err); // error
				//mul_a_b(fwd, err); // not so efficient :(

				_Forward_results.push_back(reinterpret_cast<any_tensor*>(_Layer.backward(fwd)));

				_Layer.update<_Optimizer_fn>(inputs, fwd);

				//delete &inputs;
				
				cuda_object_destroy(&err);
				cuda_object_destroy(&fwd);
			});
			for (auto&& ptr : _Forward_results) 
				cuda_object_destroy(ptr);
			_Forward_results.clear();
		}
		
		template <size_t _Batch, size_t M>
		inline auto argmax(const tensor<_Batch, M>& _Preds) const {
			tensor<_Batch, 1> _Res;
			for (int i = 0; i < _Batch; ++i) {
				unsigned int _Max_idx = 0;
				for (int j = 0; j < M; ++j) {
					if (_Preds[i][j] > _Preds[i][_Max_idx]) _Max_idx = j;
				}
				_Res[i][0] = _Max_idx;
			}
			return _Res;
		}
		
		template <size_t... _Indices>
		inline void reset_layers() {
			_Reset_layers<_Indices...>();
		}
		
		template <size_t... _Indices>
		inline void freeze_layers() {
			_Freeze_layers<_Indices...>();
		}

	private:
		template <size_t _Batch, size_t N, size_t M>
		inline auto& _Return_last(linear<N, M>&) {
			return *reinterpret_cast<tensor<_Batch, M>*>(_Forward_results.back());
		}

		inline void _Reload() {
			new(&_Forward_results) vector<any_tensor*>();
			for_each([&]<size_t N1, size_t M1>(linear<N1, M1>& _Layer, auto&& _Fn) -> void {
				new (&_Fn) remove_reference_t<decltype(_Fn)>();
				_Layer.freeze(false);
			});
		}

		template <size_t _Idx, size_t... _Indices>
		inline void _Reset_layers() {
			auto& _Layer = std::get<_Idx>(_Layers);
			_Layer.reset();

			if constexpr (sizeof...(_Indices) > 0) 
				_Reset_layers<_Indices...>();
		}

		template <size_t _Idx, size_t... _Indices>
		inline void _Freeze_layers() {
			auto& _Layer = get<_Idx>(_Layers);
			_Layer.freeze();
			sgd::layer_count -= 2;
			
			if constexpr (sizeof...(_Indices) > 0) {
				_Freeze_layers<_Indices...>();
			}
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
		inline void _Clean_up_forward_data() {
			for (int i = 0; i < _Forward_results.size(); ++i) {
				if ((i & 1) == 0) cuda_object_destroy(_Forward_results[i]);
			}
			_Forward_results.clear();
		}
		
		using any_tensor = tensor<1, 1>;
		
		linears _Layers;
		activations _Fns;
		vector<any_tensor*> _Forward_results;
	};
}


#endif // 
