#ifndef MISC_H
#define MISC_H

#include "../include/VecMatDef.h"

void computed2gdxdv(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, double t0, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, 
	double t1, const Eigen::Matrix<double,3,1> & v4, const Eigen::Matrix<double,3,1> & v5, double t2, Eigen::Matrix<double, 9, 18>& d2gdxdv);

void computed2gdxdv1edgeCase(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, double t0, 
	const Eigen::Matrix<double,3,1> & bary0, const Eigen::Matrix<double,3,1> & bary1, Eigen::Matrix<double, 3, 12>& d2gdxdv);

void computed2gdxdv2edgeCase(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, const Eigen::Matrix<double,3,1> & v4, 
	double t0, double t1, const Eigen::Matrix<double,3,1> & bary0, Eigen::Matrix<double, 6, 15>& d2gdxdv);
#endif
