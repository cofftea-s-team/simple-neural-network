#pragma once

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
		inline void save(const nnetwork<_Layers...>& _Net, const char* _Path) const {
			ofstream _File(_Path, std::ios::binary);
			_File.write(reinterpret_cast<const char*>(&_Net), sizeof(_Net));
			_File.close();
		}

		template <class... _Layers>
		inline void load(nnetwork<_Layers...>& _Net, const char* _Path) const {
			ifstream _File(_Path, std::ios::binary);
			_File.read(reinterpret_cast<char*>(&_Net), sizeof(_Net));
			_File.close();
			_Net._Reload();
		}
	};
}