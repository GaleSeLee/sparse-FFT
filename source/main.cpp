#include <complex>
#include <vector>
#include <stdlib.h>
#include <time.h>

int main()
{
    std::vector<int> edge_config = {};
    std::vector<int> radius_config = {};
    std::vector<std::complex<double>> A_baseline = {}, B_baseline = {};
    std::vector<std::complex<double>> A_accelerate = {}, B_accelerate = {};

    INPUT input();
    input.init(edge_config, radius_config);
    assert(edge_config.size() == radius_config.size());
    int num_cases = edge_config.size();

    for (int ii = 0; ii < num_cases; ii++) {
        init_complex(A_baseline);
        A_accelerate = A_baseline;
        baseline_fft(A_baseline, B_baseline);
        accelerate_fft(A_accelerate, B_accelerate);
        check_result(B_baseline, B_accelerate);

    }
}
