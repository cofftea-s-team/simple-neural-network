#pragma once

#include <utility.h>
#include <algorithms.h>
#include <cmath>
#include "utils.hpp"

namespace network {
	
	struct xentropy_loss {	
		template <range _Range>
		constexpr double compute(_Range&& _Preds, _Range&& _Output) const {
			double _Sum = 0;
			auto _It = _Preds.begin();
			auto _It2 = _Output.begin();
			for (; _It != _Preds.end(); ++_It, ++_It2) {
				_Sum += *_It2 * std::log(*_It);
			}
			return -_Sum;
		}
		template <size_t N, size_t M>
		static constexpr auto backward(const tensor<N, M>& _Preds, const tensor<N, M>& _Output) {
			auto _Res = _Preds;
			for (int i = 0; i < N; ++i) {
				for (int j = 0; j < M; ++j) {
					if (_Output[i][j]) {
						_Res[i][j] -= 1;
					}
					_Res[i][j] /= N;
				}
			}
			return _Res;
		}
	};
	constexpr xentropy_loss xentropy;

	struct mse_loss {
		template <range _Range>
		constexpr double compute(_Range&& _Preds, _Range&& _Output) const {
			double _Sum = 0;
			auto _It = _Preds.begin();
			auto _It2 = _Output.begin();
			for (; _It != _Preds.end(); ++_It, ++_It2) {
				_Sum += std::pow(*_It - *_It2, 2);
			}
			return _Sum / 2;
		}
	};
	constexpr mse_loss mse;

	
	template <class _Ty, class... Args>
	concept loss_fn = requires(_Ty _Loss, Args... _Args) {
		_Loss.backward(forward<Args>(_Args)...);
	};
}