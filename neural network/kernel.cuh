
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

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

void __dot_a_b_add_c(const double* a, const double* b, const double* c, double* d, size_t N, size_t M, size_t M1);
void __dot_a_transpose_b(const double* a, const double* b, double* d, size_t N, size_t M, size_t M1);
void __dot_transpose_a_b(const double* a, const double* b, double* d, size_t N, size_t M, size_t M1);
void __mul_a_b(double* a, const double* b, size_t N, size_t M);
template <class _Ty, size_t N, size_t M>
inline void print_matrix(const _Ty(&_Arr)[N][M]) {
    for (auto&& row : _Arr) {
        for (auto&& e : row) {
            std::cout << e << " ";
        }
        std::cout << std::endl;
    }
}
    