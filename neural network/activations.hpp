#pragma once
#include <utility.h>
#include <algorithms.h>
#include <cmath>
#include "utils.hpp"
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define NOMINMAX
#include <windows.h>

namespace cuda_network {
	
	struct base_activation {
		template <range _Range>
		inline void forward_apply(_Range& _Rng) const {
			for (auto&& e : _Rng) e = forward(e);
		}
		template <range _Range>
		inline void backward_apply(_Range& _Rng) const {
			for (auto&& e : _Rng) e = backward(e);
		}
		virtual inline double forward(double x) const = 0;
		virtual inline double backward(double x) const = 0;
	};

	struct sigmoid : public base_activation {
		inline double forward(double x) const override {
			return 1.0 / (1 + std::exp(-x));
		}

		constexpr double backward(double x) const override {
			return x * (1 - x);
		}
	};

	struct tanh : public base_activation {
		inline double forward(double x) const override {
			return std::tanh(x);
		}

		constexpr double backward(double x) const override {
			return 1 - x * x;
		}
	};

	struct relu : public base_activation {
		constexpr double forward(double x) const override {
			return ::max(0.0, x);
		}

		constexpr double backward(double x) const override {
			return x > 0 ? 1 : 0;
		}
	};

	struct softmax : public base_activation {
		template <range _Range>
		inline void forward_apply(_Range& _Rng) const {
			for (int i = 0; i < _Rng.rows(); ++i) {
				double _Sum = 0;
				for (auto&& e : _Rng[i]) _Sum += e = std::exp(e);
				for (auto&& e : _Rng[i]) {
					e /= _Sum;
#ifdef DEBUG
					assert(std::isnan(e) == false, "[softmax] NaN detected!");
#endif
				}
			}
		}
		constexpr double forward(double x) const override {
			return x;
		}
		constexpr double backward(double x) const override {
			return x * (1 - x);
		}
	};

	template <class fn>
	concept activation_fn = is_same<fn, softmax> || 
		is_base_of_v<base_activation, fn> && requires(fn _Fn, double x) {
		_Fn.forward(x);
	};
}