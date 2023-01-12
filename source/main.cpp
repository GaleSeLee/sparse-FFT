#include <complex>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cstring>

#include <cufft.h>
#include <cuda_runtime.h>

#include "common.h"

extern void baseline_fft(int c, int r, std::complex<double> *in, std::complex<double> *out);
extern void accelerate_fft(int c, int r, std::complex<double> *in, std::complex<double> *out);

void init_edge(std::vector<int> &edge) {
    for(int ii = 0; ii < edge.size(); ii++) {
        if (edge[ii] % 2 == 0) {
            edge[ii] += 1;
        }
    } 
}

void start_case(int c, int r, std::complex<double> *in,
                std::complex<double> *out_base, std::complex<double> *out_acce,
                std::complex<double> *in_d, std::complex<double> *out_base_d,
                std::complex<double> *out_acce_d) {
    int n = c * c * c;

    in = (std::complex<double> *)malloc(sizeof(std::complex<double>) * n);
    out_base = (std::complex<double> *)malloc(sizeof(std::complex<double>) * n);
    out_acce = (std::complex<double> *)malloc(sizeof(std::complex<double>) * n);
    cudaMalloc((void **)&in_d, sizeof(std::complex<double>) * n);
    cudaMalloc((void **)&out_base_d, sizeof(std::complex<double>) * n);
    cudaMalloc((void **)&out_acce_d, sizeof(std::complex<double>) * n);

    for (int ii = 0; ii < c; ii++) {
        int idx_ii = ii * c * c;
        for (int jj = 0; jj < c; jj++) {
            int idx_jj = idx_ii + jj * c;
            for (int kk = 0; kk < c; kk++)
                int dis_2 = std::abs(ii - c/2) * std::abs(ii - c/2) +
                            std::abs(jj - c/2) * std::abs(jj - c/2) +
                            std::abs(kk - c/2) * std::abs(kk - c/2);
                if (dis_2 > r * r) {
                    in[idx_jj + kk] = std::complex<double> (0, 0);
                }
                else {
                    in[idx_jj + kk] = randn();
                }
        }
    }
    cudaMemcpy(in_d, in, sizeof(std::complex<double>) * n, cudaMemcpyHostToDevice);
}

void warmup_cufft(int c) {
    cufftHandle plan;
    cufftPlan3d(&plan, c, c, c, CUFFT_Z2Z);
}

bool check_result(int c, std::complex<double> *out_base,
                  std::complex<double> *out_acce, double tol = 1e-8) {
    // TODO
    // 1e-8
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
    return true;
}

void end_case(std::complex<double> *in, std::complex<double> *out_base,
              std::complex<double> *out_acce) {
    std::free(in);
    std::free(out_base);
    std::free(out_acce);
}

int main()
{
    std::vector<int> edge_config{32, 32, 64, 64, 128, 128, 256, 256, 512, 512};
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
         // cpu memory to gpu memory
        start_case(edge_config[ii], radius_config[ii],
                   in_data, out_baseline, out_accelerate,
                   in_data_d, out_baseline_d, out_accelerate_d);

        float sum_baseline_time = 0.0;
        float sum_accelerate_time = 0.0;
        float time_elapsed = 0.0;
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        warmup_cufft(edge_config[ii]);
        for (int jj = 0; jj < repeat_time; jj++) {
            cudaEventRecord(start, 0);
            baseline_fft(c, r, in_data_d, out_baseline_d);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(start);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time_elapsed,start,stop);
            sum_baseline_time += time_elapsed;
        }

        for (int jj = 0; jj < repeat_time; jj++) {
            cudaEventRecord(start, 0);
            // accelerate_fft(c, r, in_data_d, out_accelerate_d);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(start);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time_elapsed,start,stop);
            sum_accelerate_time += time_elapsed;
        }

        check_result(out_baseline, out_accelerate);
        std::cout << "accelerate = " << time_elapsed << " ms" << std::endl;
        end_case(in_data, out_baseline, out_accelerate);
    }
}
