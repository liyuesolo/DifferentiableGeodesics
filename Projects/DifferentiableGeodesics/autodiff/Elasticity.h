#ifndef ELASTICIY_H
#define ELASTICIY_H

#include "../include/VecMatDef.h"

void computeGeodesicNHEnergy(double lambda, double mu, double undeformed_area, const Eigen::Matrix<double,3,1> & l, const Eigen::Matrix<double,2,2> & X_inv, 
	double& energy);
void computeGeodesicNHEnergyGradient(double lambda, double mu, double undeformed_area, const Eigen::Matrix<double,3,1> & l, const Eigen::Matrix<double,2,2> & X_inv, 
	Eigen::Matrix<double, 3, 1>& energygradient);
void computeGeodesicNHEnergyHessian(double lambda, double mu, double undeformed_area, const Eigen::Matrix<double,3,1> & l, const Eigen::Matrix<double,2,2> & X_inv, 
	Eigen::Matrix<double, 3, 3>& energyhessian);


void computeGeodesicNHEnergyWithC(double lambda, double mu, const Eigen::Matrix<double,3,1> & l, const Eigen::Matrix<double,3,3> & A_inv, double rest_area, 
	double& energy);
void computeGeodesicNHEnergyWithCGradient(double lambda, double mu, const Eigen::Matrix<double,3,1> & l, const Eigen::Matrix<double,3,3> & A_inv, double rest_area, 
	Eigen::Matrix<double, 3, 1>& energygradient);
void computeGeodesicNHEnergyWithCHessian(double lambda, double mu, const Eigen::Matrix<double,3,1> & l, const Eigen::Matrix<double,3,3> & A_inv, double rest_area, 
	Eigen::Matrix<double, 3, 3>& energyhessian);


void computeGeodesicEnergyWithReferenceC(double w, const Eigen::Matrix<double,3,1> & l, const Eigen::Matrix<double,3,3> & A_inv, double rest_area, const Eigen::Matrix<double,3,1> & C_ref_entries, 
	double& energy);
void computeGeodesicEnergyWithReferenceCGradient(double w, const Eigen::Matrix<double,3,1> & l, const Eigen::Matrix<double,3,3> & A_inv, double rest_area, const Eigen::Matrix<double,3,1> & C_ref_entries, 
	Eigen::Matrix<double, 3, 1>& energygradient);
void computeGeodesicEnergyWithReferenceCHessian(double w, const Eigen::Matrix<double,3,1> & l, const Eigen::Matrix<double,3,3> & A_inv, double rest_area, const Eigen::Matrix<double,3,1> & C_ref_entries, 
	Eigen::Matrix<double, 3, 3>& energyhessian);
	
#endif