#ifndef CST_3D_SHELL_H
#define CST_3D_SHELL_H

#include "../include/VecMatDef.h"


T compute3DCSTShellEnergy(T poissonsRatio, T stiffness, const Matrix<T,3,1> & x1, const Matrix<T,3,1> & x2, const Matrix<T,3,1> & x3, 
	const Matrix<T,3,1> & x1Undef, const Matrix<T,3,1> & x2Undef, const Matrix<T,3,1> & x3Undef);

void compute3DCSTShellEnergyGradient(T poissonsRatio, T stiffness, const Matrix<T,3,1> & x1, const Matrix<T,3,1> & x2, const Matrix<T,3,1> & x3, 
	const Matrix<T,3,1> & x1Undef, const Matrix<T,3,1> & x2Undef, const Matrix<T,3,1> & x3Undef, Matrix<T, 9, 1>& energygradient);

void compute3DCSTShellEnergyHessian(T poissonsRatio, T stiffness, const Matrix<T,3,1> & x1, const Matrix<T,3,1> & x2, const Matrix<T,3,1> & x3, 
	const Matrix<T,3,1> & x1Undef, const Matrix<T,3,1> & x2Undef, const Matrix<T,3,1> & x3Undef, Matrix<T, 9, 9>& energyhessian);


double computeDSBendingEnergy(double stiffness, const Matrix<double,3,1> & x0, const Matrix<double,3,1> & x1, const Matrix<double,3,1> & x2, const Matrix<double,3,1> & x3, 
	const Matrix<double,3,1> & x0Undef, const Matrix<double,3,1> & x1Undef, const Matrix<double,3,1> & x2Undef, const Matrix<double,3,1> & x3Undef);
void computeDSBendingEnergyGradient(double stiffness, const Matrix<double,3,1> & x0, const Matrix<double,3,1> & x1, const Matrix<double,3,1> & x2, const Matrix<double,3,1> & x3, 
	const Matrix<double,3,1> & x0Undef, const Matrix<double,3,1> & x1Undef, const Matrix<double,3,1> & x2Undef, const Matrix<double,3,1> & x3Undef, Matrix<double, 12, 1>& energygradient);
void computeDSBendingEnergyHessian(double stiffness, const Matrix<double,3,1> & x0, const Matrix<double,3,1> & x1, const Matrix<double,3,1> & x2, const Matrix<double,3,1> & x3, 
	const Matrix<double,3,1> & x0Undef, const Matrix<double,3,1> & x1Undef, const Matrix<double,3,1> & x2Undef, const Matrix<double,3,1> & x3Undef, Matrix<double, 12, 12>& energyhessian);

void computeTetVolume(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, double& energy);
void computeTetVolumeGradient(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, Eigen::Matrix<double, 9, 1>& energygradient);
void computeTetVolumeHessian(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, Eigen::Matrix<double, 9, 9>& energyhessian);
#endif