#include <complex>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cstring>
#include <iostream>
#include <unistd.h>

#include <cufft.h>
#include <cuda_runtime.h>

#include "common.h"

extern void baseline_fft(int c, int r, std::complex<double> *in, std::complex<double> *out);
extern void accelerate_fft(int c, int r, std::complex<double> *in, std::complex<double> *out);

inline void cuAssert(cudaError_t status, const char *file, int line) {
    if (status != cudaSuccess)
        std::cerr<<"cuda assert: "<<cudaGetErrorString(status)<<", file: "<<file<<", line: "<<line<<std::endl;
}
#define cuErrCheck(res)                                 \
    {                                                   \
        cuAssert((res), __FILE__, __LINE__);            \
    }

void init_edge(std::vector<int> &edge) {
    for(int ii = 0; ii < edge.size(); ii++) {
        if (edge[ii] % 2 == 0) {
            edge[ii] += 1;
        }
    } 
}

void start_case(int c, int r, std::complex<double> **in,
                std::complex<double> **out_base, std::complex<double> **out_acce) {
    int n = c * c * c;

    *in = (std::complex<double> *)malloc(sizeof(std::complex<double>) * n);
    *out_base = (std::complex<double> *)malloc(sizeof(std::complex<double>) * n);
    *out_acce = (std::complex<double> *)malloc(sizeof(std::complex<double>) * n);

    // There will be a better way to init data
    for (int ii = 0; ii < c; ii++) {
        int idx_ii = ii * c * c;
        for (int jj = 0; jj < c; jj++) {
            int idx_jj = idx_ii + jj * c;
            for (int kk = 0; kk < c; kk++) {
                int dis_2 = (ii - c/2) * (ii - c/2) +
                            (jj - c/2) * (jj - c/2) +
                            (kk - c/2) * (kk - c/2);
                if (dis_2 > r * r) {
                    (*in)[idx_jj + kk] = std::complex<double> (0, 0);
                }
                else {
                    (*in)[idx_jj + kk] = randn();
                }
            }
        }
    }
}

void warmup_cufft(int c) {
    cufftHandle plan;
    cufftPlan3d(&plan, c, c, c, CUFFT_Z2Z);
    cufftDestroy(plan);
}

bool check_result(int c, std::complex<double> *out_base,
                  std::complex<double> *out_acce, double tol = 1e-8) {
    for (int ii = 0; ii < c; ii++) {
        int idx_ii = ii * c * c;
        for (int jj = 0; jj < c; jj++) {
            int idx_jj = idx_ii + jj * c;
            for (int kk = 0; kk < c; kk++) {
                int idx_kk = idx_jj + kk;
                if (std::abs(out_base[idx_kk]-out_acce[idx_kk]) > tol) {
                    return false;
                }
            }
        }
    }
    return true;
}

void end_case(std::complex<double> **in, std::complex<double> **out_base,
              std::complex<double> **out_acce) {
    std::free(*in);
    std::free(*out_base);
    std::free(*out_acce);
}

int main()
{
	std::vector<int> edge_config = {32, 32, 64, 64, 128, 128, 256, 256, 512, 512};
	std::vector<int> radius_config = {8, 6, 16, 12, 32, 24, 64, 48, 96, 128};
    const int repeat_time = 10;

    assert(edge_config.size() == radius_config.size());

    std::complex<double> *in_data;
    std::complex<double> *out_baseline;
    std::complex<double> *out_accelerate;
    std::complex<double> *in_data_d;
    std::complex<double> *out_baseline_d;
    std::complex<double> *out_accelerate_d;

    init_edge(edge_config);
    int num_cases = edge_config.size();

    for (int ii = 0; ii < num_cases; ii++) {
        int c = edge_config[ii];
		int n = c * c * c;
        int r = radius_config[ii];
        start_case(c, r, &in_data, &out_baseline, &out_accelerate);

		cudaMalloc((void **)&in_data_d, sizeof(std::complex<double>) * n);
		cudaMalloc((void **)&out_baseline_d, sizeof(std::complex<double>) * n);
		cudaMalloc((void **)&out_accelerate_d, sizeof(std::complex<double>) * n);
		cudaMemcpy(in_data_d, in_data, sizeof(std::complex<double>) * n, cudaMemcpyHostToDevice);
        float sum_baseline_time = 0.0;
        float sum_accelerate_time = 0.0;
        float time_elapsed = 0.0;
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        warmup_cufft(c);
        for (int jj = 0; jj < repeat_time; jj++) {
            cudaEventRecord(start);
            baseline_fft(c, r, in_data_d, out_baseline_d);
            cudaDeviceSynchronize();
            cuErrCheck(cudaGetLastError());
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time_elapsed,start,stop);
            sum_baseline_time += time_elapsed;
        }

		cudaMemcpy(out_baseline, out_baseline_d,
				   sizeof(std::complex<double>) * c * c * c, cudaMemcpyDeviceToHost);
        cudaFree(out_baseline_d);

        for (int jj = 0; jj < repeat_time; jj++) {
            cudaEventRecord(start, 0);
            // need to complete
            // accelerate_fft(c, r, in_data_d, out_accelerate_d);
            cudaDeviceSynchronize();
            cuErrCheck(cudaGetLastError());
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time_elapsed,start,stop);
            sum_accelerate_time += time_elapsed;
        }
		cudaMemcpy(out_accelerate, out_accelerate_d,
				   sizeof(std::complex<double>) * c * c * c, cudaMemcpyDeviceToHost);
	    cudaFree(out_accelerate_d);
		cudaFree(in_data_d);

        if (!check_result(c, out_baseline, out_accelerate)) {
			std::cout << "Error" << std::endl;
			exit(1);
		}
        end_case(&in_data, &out_baseline, &out_accelerate);

        std::cout << "accelerate = " << sum_baseline_time << " ms" << std::endl;
    }
}
