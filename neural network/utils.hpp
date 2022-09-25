#pragma once

#include <tuple>
#include <random>
#include <utility.h>

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
		std::mt19937 engine(time(NULL));
		std::normal_distribution<double> normal(0, 1);

		for (auto& _Val : _Rng) {
			_Val = normal(engine) * 0.1;
		}
	};

	template <size_t N, size_t M>
	using tensor = matrix<double, N, M>;

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