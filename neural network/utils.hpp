#pragma once

#include <tuple>
#include <random>
#include <utility.h>
#include "kernel.cuh"

namespace network {
	template<bool>
	struct zero_or_one {
		template<class E> using type = std::tuple<E>;
	};

	template<>
	struct zero_or_one<false> {
		template<class E> using type = std::tuple<>;
	};

	template<class Tuple, class = std::make_index_sequence<std::tuple_size<Tuple>::value>>
	struct evens;

	template<class... Es, size_t... Is>
	struct evens<std::tuple<Es...>, std::index_sequence<Is...>> {
		using type = decltype(std::tuple_cat(
			std::declval<typename zero_or_one<Is % 2 == 0>::template type<Es>>()...
		));
	};

	template<class Tuple, class = std::make_index_sequence<std::tuple_size<Tuple>::value>>
	struct odds;

	template<class... Es, size_t... Is>
	struct odds<std::tuple<Es...>, std::index_sequence<Is...>> {
		using type = decltype(std::tuple_cat(
			std::declval<typename zero_or_one<Is % 2 == 1>::template type<Es>>()...
		));
	};

	using pipeline::range;

	template <range _Range>
	inline void fill_randn(_Range& _Rng) {
		std::random_device device{};
		std::mt19937 engine(device());
		std::normal_distribution<double> normal(0, 1);

		for (auto& _Val : _Rng) {
			_Val = normal(engine) * 0.1;
		}
	};

	template <size_t N, size_t M>
	using tensor = matrix<double, N, M>;

	template <size_t N, size_t M, size_t M1>
	inline auto dot_a_b_add_c(const tensor<N, M>& A, const tensor<M, M1>& B, const tensor<1, M1>& C) {
		tensor<N, M1> _Res;
		auto a = reinterpret_cast<const double*>(A.data());
		auto b = reinterpret_cast<const double*>(B.data());
		auto c = reinterpret_cast<const double*>(C.data());
		auto d = reinterpret_cast<double*>(_Res.data());
		__dot_a_b_add_c(a, b, c, d, N, M, M1);
		return _Res;
	}

	template <size_t N, size_t M, size_t M1>
	inline auto dot_a_transpose_b(const tensor<N, M>& A, const tensor<M1, M>& B) {
		auto _Res = new tensor<N, M1>();
		auto a = reinterpret_cast<const double*>(A.data());
		auto b = reinterpret_cast<const double*>(B.data());
		auto d = reinterpret_cast<double*>(_Res->data());
		__dot_a_transpose_b(a, b, d, N, M, M1);
		return _Res;
	}

	template <size_t N, size_t M, size_t M1>
	inline auto dot_transpose_a_b(const tensor<M, N>& A, const tensor<M, M1>& B) {
		tensor<N, M1> _Res;
		auto a = reinterpret_cast<const double*>(A.data());
		auto b = reinterpret_cast<const double*>(B.data());
		auto d = reinterpret_cast<double*>(_Res.data());
		__dot_transpose_a_b(a, b, d, N, M, M1);
		return _Res;
	}

	template <size_t N, size_t M>
	inline void mul_a_b(tensor<N, M>& A, const tensor<N, M>& B) {
		auto a = reinterpret_cast<double*>(A.data());
		auto b = reinterpret_cast<const double*>(B.data());
		__mul_a_b(a, b, N, M);
	}
#ifdef _STD
}
#include <sstream>
namespace network {
	template <class... Args>
	constexpr _STD string to_string(Args&&... _Args) {
		static_assert(sizeof...(_Args) != 0, "to_string() requires at least one argument");

		_STD ostringstream _Oss;
		(_Oss << ... << _Args);
		return _Oss.str();
	}
#endif
}