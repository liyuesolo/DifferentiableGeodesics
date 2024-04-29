#ifndef POINT_TO_PLANE_H
#define POINT_TO_PLANE_H

#include "../include/VecMatDef.h"

void compute4PointsPlaneFittingEnergy(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, double& energy);
void compute4PointsPlaneFittingEnergyGradient(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, Eigen::Matrix<double, 12, 1>& energygradient);
void compute4PointsPlaneFittingEnergyHessian(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, Eigen::Matrix<double, 12, 12>& energyhessian);


void compute5PointsPlaneFittingEnergy(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, const Eigen::Matrix<double,3,1> & v4, 
	double& energy);
void compute5PointsPlaneFittingEnergyGradient(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, const Eigen::Matrix<double,3,1> & v4, 
	Eigen::Matrix<double, 15, 1>& energygradient);
void compute5PointsPlaneFittingEnergyHessian(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, const Eigen::Matrix<double,3,1> & v4, 
	Eigen::Matrix<double, 15, 15>& energyhessian);

void compute6PointsPlaneFittingEnergy(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, const Eigen::Matrix<double,3,1> & v4, 
	const Eigen::Matrix<double,3,1> & v5, double& energy);
void compute6PointsPlaneFittingEnergyGradient(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, const Eigen::Matrix<double,3,1> & v4, 
	const Eigen::Matrix<double,3,1> & v5, Eigen::Matrix<double, 18, 1>& energygradient);
void compute6PointsPlaneFittingEnergyHessian(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, const Eigen::Matrix<double,3,1> & v4, 
	const Eigen::Matrix<double,3,1> & v5, Eigen::Matrix<double, 18, 18>& energyhessian);

void compute7PointsPlaneFittingEnergy(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, const Eigen::Matrix<double,3,1> & v4, 
	const Eigen::Matrix<double,3,1> & v5, const Eigen::Matrix<double,3,1> & v6, double& energy);
void compute7PointsPlaneFittingEnergyGradient(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, const Eigen::Matrix<double,3,1> & v4, 
	const Eigen::Matrix<double,3,1> & v5, const Eigen::Matrix<double,3,1> & v6, Eigen::Matrix<double, 21, 1>& energygradient);
void compute7PointsPlaneFittingEnergyHessian(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, const Eigen::Matrix<double,3,1> & v4, 
	const Eigen::Matrix<double,3,1> & v5, const Eigen::Matrix<double,3,1> & v6, Eigen::Matrix<double, 21, 21>& energyhessian);

void compute8PointsPlaneFittingEnergy(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, const Eigen::Matrix<double,3,1> & v4, 
	const Eigen::Matrix<double,3,1> & v5, const Eigen::Matrix<double,3,1> & v6, const Eigen::Matrix<double,3,1> & v7, double& energy);
void compute8PointsPlaneFittingEnergyGradient(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, const Eigen::Matrix<double,3,1> & v4, 
	const Eigen::Matrix<double,3,1> & v5, const Eigen::Matrix<double,3,1> & v6, const Eigen::Matrix<double,3,1> & v7, Eigen::Matrix<double, 24, 1>& energygradient);
void compute8PointsPlaneFittingEnergyHessian(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, const Eigen::Matrix<double,3,1> & v4, 
	const Eigen::Matrix<double,3,1> & v5, const Eigen::Matrix<double,3,1> & v6, const Eigen::Matrix<double,3,1> & v7, Eigen::Matrix<double, 24, 24>& energyhessian);


void compute9PointsPlaneFittingEnergy(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, const Eigen::Matrix<double,3,1> & v4, 
	const Eigen::Matrix<double,3,1> & v5, const Eigen::Matrix<double,3,1> & v6, const Eigen::Matrix<double,3,1> & v7, const Eigen::Matrix<double,3,1> & v8, double& energy);
void compute9PointsPlaneFittingEnergyGradient(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, const Eigen::Matrix<double,3,1> & v4, 
	const Eigen::Matrix<double,3,1> & v5, const Eigen::Matrix<double,3,1> & v6, const Eigen::Matrix<double,3,1> & v7, const Eigen::Matrix<double,3,1> & v8, Eigen::Matrix<double, 27, 1>& energygradient);
void compute9PointsPlaneFittingEnergyHessian(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, const Eigen::Matrix<double,3,1> & v4, 
	const Eigen::Matrix<double,3,1> & v5, const Eigen::Matrix<double,3,1> & v6, const Eigen::Matrix<double,3,1> & v7, const Eigen::Matrix<double,3,1> & v8, Eigen::Matrix<double, 27, 27>& energyhessian);

#endif
