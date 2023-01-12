#include <complex>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cstring>
#include <cuda_runtime.h>

#include "common.h"

extern void baseline_fft(int c, int r, std::complex<double> *, std::complex<double> *);
extern void accelerate_fft(int c, int r, std::complex<double> *, std::complex<double> *);

void init_edge(std::vector<int> &edge) {
    for(int ii = 0; ii < edge.size(); ii++) {
        if (edge[ii] % 2 == 0) {
            edge[ii] += 1;
        }
    } 
}

void start_case(int c, int r, std::complex<double> *in,
                std::complex<double> *out_base, std::complex<double> *out_acce) {
    int n = c * c * c;
    in = (std::complex<double> *)malloc(sizeof(std::complex<double>) * n);
    out_base = (std::complex<double> *)malloc(sizeof(std::complex<double>) * n);
    out_acce = (std::complex<double> *)malloc(sizeof(std::complex<double>) * n);
    std::memset(out_base, 0, sizeof(std::complex<double>) * n);
    std::memset(out_acce, 0, sizeof(std::complex<double>) * n);
    for (int ii = 0; ii < c; ii++) {
        int idx_ii = ii * c * c;
        for (int jj = 0; jj < c; jj++) {
            int idx_jj = idx_ii + jj * c;
            for (int kk = 0; kk < c; kk++)
                in[idx_jj + kk] = randn();
        }
    }
}

bool check_result(int c, std::complex<double> *out_base,
                  std::complex<double> *out_acce) {

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
    assert(edge_config.size() == radius_config.size());

    std::complex<double> *in_data;
    std::complex<double> *out_baseline;
    std::complex<double> *out_accelerate;

    init_edge(edge_config);
    int num_cases = edge_config.size();

    for (int ii = 0; ii < num_cases; ii++) {
        start_case(edge_config[ii], radius_config[ii],
                   in_data, out_baseline, out_accelerate);
        baseline_fft(c, r, in_data, out_baseline);
        accelerate_fft(c, r, in_data, out_accelerate);
        check_result();
        end_case(in_data, out_baseline, out_accelerate);
    }
}
