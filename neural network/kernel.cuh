
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
namespace network {
	template <class _Ty>
	inline _Ty* cuda_alloc(size_t _Count) {
		_Ty* _Ptr;
		auto cudaStatus = cudaMalloc((void**)&_Ptr, _Count * sizeof(_Ty));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "malloc failed!");
			exit(-1);
		}
		return _Ptr;
	}

	template <size_t _Size, class _Ty>
	inline _Ty* alloc_in_cuda(const _Ty* _Src) {
		_Ty* _Ptr = cuda_alloc<_Ty>(_Size);
		auto cudaStatus = cudaMemcpy(_Ptr, _Src, _Size * sizeof(_Ty), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			exit(-1);
		}
		return _Ptr;
	}

	template <size_t _Size, class _Ty>
	inline void copy_to_cuda(const _Ty* _Src, _Ty* _Data) {
		auto cudaStatus = cudaMemcpy(_Data, _Src, _Size * sizeof(_Ty), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			exit(-1);
		}
	}

	inline double check_cuda_ptr(const double* ptr) {
		cudaPointerAttributes att;
		auto err = cudaPointerGetAttributes(&att, ptr);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudaPointerGetAttributes failed!");
			exit(-1);
		}
	}

	template <size_t _Size, class _Ty>
	inline void copy_to_host(const _Ty* _Src, _Ty* _Data) {

		auto _Tmp = malloc(_Size * sizeof(_Ty));
		auto cudaStatus = cudaMemcpy(_Tmp, _Src, _Size * sizeof(_Ty), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			exit(-1);
		}
		memmove(_Data, _Tmp, _Size * sizeof(_Ty));
		delete _Tmp;
	}

	template <size_t _Size, class _Ty>
	inline _Ty* copy_to_host(const _Ty* _Src) {
		_Ty* _Data = new _Ty[_Size];
		auto cudaStatus = cudaMemcpy(_Data, _Src, _Size * sizeof(_Ty), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			exit(-1);
		}
		return _Data;
	}

	void __dot_a_b_add_c(const double* a, const double* b, const double* c, double* d, size_t N, size_t M, size_t M1);
	void __dot_a_transpose_b(const double* a, const double* b, double* d, size_t N, size_t M, size_t M1);
	void __dot_transpose_a_b(const double* a, const double* b, double* d, size_t N, size_t M, size_t M1);
	void __mul_a_b(double* a, const double* b, size_t N, size_t M);
}