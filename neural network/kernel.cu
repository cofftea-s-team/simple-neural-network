#include "kernel.cuh"

__global__ 
void __dot_a_b_add_c_kernel(double* _A, double* _B, double* _C, double* _D, size_t N, size_t M, size_t M1)
{
    int i = blockIdx.x;
    int j = threadIdx.x;

    *(_D + i * M1 + j) = _C[j];
    for (int k = 0; k < M; ++k) {
        *(_D + i * M1 + j) += *(_A + i * M + k) * *(_B + k * M1 + j);
    }
}

void __dot_a_b_add_c(const double* a, const double* b, const double* c, double* d, size_t N, size_t M, size_t M1) {
    auto _A = cuda_alloc<double>(N * M);
    auto _B = cuda_alloc<double>(M * M1);
    auto _C = cuda_alloc<double>(M1);
    auto _D = cuda_alloc<double>(N * M1);

    cudaMemcpy(_A, a, N * M * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(_B, b, M * M1 * sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(_C, c, M1 * sizeof(double), cudaMemcpyHostToDevice); 

    __dot_a_b_add_c_kernel<<<N, M1>>> (_A, _B, _C, _D, N, M, M1); // applies dot on a, b and adds c
    cudaDeviceSynchronize();

	cudaMemcpy(d, _D, N * M1 * sizeof(double), cudaMemcpyDeviceToHost); // copy result to host
    cudaFree(_A);
    cudaFree(_B);
    cudaFree(_C);
    cudaFree(_D);
}

__global__ 
void __dot_a_transpose_b_kernel(const double* _A, const double* _B, double* _D, size_t N, size_t M, size_t M1) 
{
    int i = blockIdx.x;
    int j = threadIdx.x;

    *(_D + i * M1 + j) = 0;
    for (int k = 0; k < M; ++k) {
        *(_D + i * M1 + j) += *(_A + i * M + k) * *(_B + j * M + k);
    }
}

void __dot_a_transpose_b(const double* a, const double* b, double* d, size_t N, size_t M, size_t M1)
{
	auto _A = cuda_alloc<double>(N * M);
	auto _B = cuda_alloc<double>(M * M1);
	auto _D = cuda_alloc<double>(N * M1);

	cudaMemcpy(_A, a, N * M * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(_B, b, M * M1 * sizeof(double), cudaMemcpyHostToDevice);

    __dot_a_transpose_b_kernel<<<N, M1>>> (_A, _B, _D, N, M, M1); // applies dot on a, transposed b
	cudaDeviceSynchronize();

	cudaMemcpy(d, _D, N * M1 * sizeof(double), cudaMemcpyDeviceToHost); // copy result to host
	cudaFree(_A);
	cudaFree(_B);
	cudaFree(_D);
}

__global__
void __dot_transpose_a_b_kernel(const double* _A, const double* _B, double* _D, size_t N, size_t M, size_t M1)
{
	int i = blockIdx.x;
	int j = threadIdx.x;

	*(_D + i * M1 + j) = 0;
	for (int k = 0; k < M; ++k) {
		*(_D + i * M1 + j) += *(_A + k * N + i) * *(_B + k * M1 + j);
	}
}

void __dot_transpose_a_b(const double* a, const double* b, double* d, size_t N, size_t M, size_t M1)
{
	auto _A = cuda_alloc<double>(N * M);
	auto _B = cuda_alloc<double>(M * M1);
	auto _D = cuda_alloc<double>(N * M1);

	cudaMemcpy(_A, a, N * M * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(_B, b, M * M1 * sizeof(double), cudaMemcpyHostToDevice);

	__dot_transpose_a_b_kernel << <N, M1 >> > (_A, _B, _D, N, M, M1); // applies dot on transposed a, b
	cudaDeviceSynchronize();

	cudaMemcpy(d, _D, N * M1 * sizeof(double), cudaMemcpyDeviceToHost); // copy result to host
	cudaFree(_A);
	cudaFree(_B);
	cudaFree(_D);
}
