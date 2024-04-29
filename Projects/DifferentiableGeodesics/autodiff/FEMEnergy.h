#ifndef FEM_ENERGY_H
#define FEM_ENERGY_H

#include "../include/VecMatDef.h"

void computeLinearTet3DNeoHookeanEnergy(double E, double nu, const Eigen::Matrix<double,4,3> & x, const Eigen::Matrix<double,4,3> & X, double& energy);
void computeLinearTet3DNeoHookeanEnergyGradient(double E, double nu, const Eigen::Matrix<double,4,3> & x, const Eigen::Matrix<double,4,3> & X, Eigen::Matrix<double, 12, 1>& energygradient);
void computeLinearTet3DNeoHookeanEnergyHessian(double E, double nu, const Eigen::Matrix<double,4,3> & x, const Eigen::Matrix<double,4,3> & X, Eigen::Matrix<double, 12, 12>& energyhessian);

#endif