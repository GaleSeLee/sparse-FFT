#include "common.h"

std::complex<double> randn() {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::normal_distribution<double> dist(0, 1);
    return std::complex<double> {dist(mt), dist(mt)};
}