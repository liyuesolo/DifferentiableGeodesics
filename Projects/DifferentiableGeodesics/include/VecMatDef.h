#ifndef VECMATDEF_H
#define VECMATDEF_H

#include <Eigen/Core>

template <typename T, int dim>
using Vector = Eigen::Matrix<T, dim, 1, 0, dim, 1>;

template <typename T, int n, int m>
using Matrix = Eigen::Matrix<T, n, m, 0, n, m>;


using T = double;

#endif