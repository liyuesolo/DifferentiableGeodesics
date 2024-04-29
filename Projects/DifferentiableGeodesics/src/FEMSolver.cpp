#include "../include/IntrinsicSimulation.h"
#include "../autodiff/FEMEnergy.h"

void IntrinsicSimulation::computeDeformationGradient(const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, TM& F)
{
    //N(b) = [1]
    Matrix<T, 4, 3> dNdb;
        dNdb << -1.0, -1.0, -1.0, 
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0;
    TM dXdb = x_undeformed.transpose() * dNdb;
    TM dxdb = x_deformed.transpose() * dNdb;
    F = dxdb * dXdb.inverse();
}

T IntrinsicSimulation::computeVolume(const EleNodes& x_undeformed)
{
    TV a = x_undeformed.row(1) - x_undeformed.row(0);
	TV b = x_undeformed.row(2) - x_undeformed.row(0);
	TV c = x_undeformed.row(3) - x_undeformed.row(0);
	T volumeParallelepiped = a.cross(b).dot(c);
	T tetVolume = 1.0 / 6.0 * volumeParallelepiped;
	return tetVolume;
}


T IntrinsicSimulation::computeInversionFreeStepsize()
{
    if (!use_FEM)
        return 1.0;
    Matrix<T, 4, 3> dNdb;
        dNdb << -1.0, -1.0, -1.0, 
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0;
           
    VectorXT step_sizes = VectorXT::Zero(num_ele);

    iterateElementParallel([&](const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, const VtxList& indices, int tet_idx)
    {
        TM dXdb = x_undeformed.transpose() * dNdb;
        TM dxdb = x_deformed.transpose() * dNdb;
        TM A = dxdb * dXdb.inverse();
        T a, b, c, d;
        a = A.determinant();
        b = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0) + A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0) + A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
        c = A.diagonal().sum();
        d = 0.8;

        T t = getSmallestPositiveRealCubicRoot(a, b, c, d);
        if (t < 0 || t > 1) t = 1;
            step_sizes(tet_idx) = t;
    });
    return step_sizes.minCoeff();
}

void IntrinsicSimulation::polarSVD(TM& F, TM& U, TV& Sigma, TM& VT)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(F, Eigen::ComputeThinU | Eigen::ComputeThinV);
	U = svd.matrixU();
	Sigma = svd.singularValues();
	Eigen::Matrix3d V = svd.matrixV();


	// if det(u) = -1 flip the sign of u3 and sigma3
	if (U.determinant() < 0)
	{
		for (int i = 0; i < 3; i++)
		{
			U(i, 2) *= -1.0;
		}
		Sigma[2] *= -1.0;
	}
	// if det(v) = -1 flip the sign of v3 and sigma3
	if (V.determinant() < 0)
	{
		for (int i = 0; i < 3; i++)
		{
			V(i, 2) *= -1.0;
		}
		Sigma[2] *= -1.0;
	}

	VT = V.transpose();

	F = U * Sigma.asDiagonal() * VT;
}

T IntrinsicSimulation::addElastsicPotential()
{
    
    VectorXT energies_neoHookean(num_ele);
    energies_neoHookean.setZero();
    iterateElementSerial([&](const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, const VtxList& indices, int tet_idx)
    {
        T ei = 0.0;
        computeLinearTet3DNeoHookeanEnergy(E_solids[tet_idx], nu_solid, x_deformed, x_undeformed, ei);
        energies_neoHookean[tet_idx] += ei;
        // if (std::isnan(ei))
        // {
        //     std::cout << "NAN " << ei << std::endl;
        // }
    });
    
    return energies_neoHookean.sum();
}

void IntrinsicSimulation::addElasticForceEntries(VectorXT& residual)
{
    iterateElementSerial([&](const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, const VtxList& indices, int tet_idx)
    {
        // T volume = computeVolume(x_undeformed);
        // if (volume < 1e-8)
        //     return;
        Vector<T, 12> dedx;
        computeLinearTet3DNeoHookeanEnergyGradient(E_solids[tet_idx], nu_solid, x_deformed, x_undeformed, dedx);
        addForceEntry<3>(residual, indices, -dedx, fem_dof_start);
        // std::cout << dedx.transpose() << std::endl;
    });
}


void IntrinsicSimulation::addElasticHessianEntries(std::vector<Entry>& entries)
{
    iterateElementSerial([&](const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, const VtxList& indices, int tet_idx)
    {
        T volume = computeVolume(x_undeformed);
        // if (volume < 1e-8)
        //     return;
        Matrix<T, 12, 12> hessian, hessian_ad;
        computeLinearTet3DNeoHookeanEnergyHessian(E_solids[tet_idx], nu_solid, x_deformed, x_undeformed, hessian);
        addHessianEntry<3, 3>(entries, indices, hessian, fem_dof_start, fem_dof_start);
    });
}