#include "common.h"

std::complex<double> randn() {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::normal_distribution<double> dist(0, 1);
    return {dist(mt), dist(mt)};
}