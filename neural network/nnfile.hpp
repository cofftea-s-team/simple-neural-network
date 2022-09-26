#pragma once

#include "kernel.cuh"
#include <utility.h>
#include "utils.hpp"
#include "linear.hpp"
#include "network.hpp"
#include <fstream>

namespace cuda_network {
	
	using std::ifstream;
	using std::ofstream;

	class nnfile {
	public:
		template <class... _Layers>
		inline void save(nnetwork<_Layers...>& _Net, const char* _Path) const {
			unsigned int _Layer_number = 0;
			_Net.for_each([&]<size_t N, size_t M>(linear<N, M>& _Layer, auto&&) -> void {
				pair<tensor<N, M>, tensor<1, M>> _Data;
				copy_to_host<N * M>(_Layer._Weights, reinterpret_cast<double*>(&_Data.first));
				copy_to_host<M>(_Layer._Biases, reinterpret_cast<double*>(&_Data.second));

				ofstream _File(_Create_path(_Path, _Layer_number), std::ios::binary);
				_File.write(reinterpret_cast<char*>(&_Data), sizeof(pair<tensor<N, M>, tensor<1, M>>));
				_File.close();

				++_Layer_number;
			});
		}

		template <class... _Layers>
		inline void load(nnetwork<_Layers...>& _Net, const char* _Path) const {
			unsigned int _Layer_number = 0;
			_Net.for_each([&]<size_t N, size_t M>(linear<N, M>&_Layer, auto&&) -> void {
				pair<tensor<N, M>, tensor<1, M>> _Data;
				ifstream _File(_Create_path(_Path, _Layer_number), std::ios::binary);
				_File.read(reinterpret_cast<char*>(&_Data), sizeof(pair<tensor<N, M>, tensor<1, M>>));
				_File.close();

				copy_to_cuda<N * M>(reinterpret_cast<double*>(&_Data.first), _Layer._Weights);
				copy_to_cuda<M>(reinterpret_cast<double*>(&_Data.second), _Layer._Biases);

				++_Layer_number;
			});
		}
	private:
		auto _Create_path(const char* _File_name, size_t _Id) const {
			return to_string(_File_name, "_L", _Id, ".cofftea");
		}
	};
}