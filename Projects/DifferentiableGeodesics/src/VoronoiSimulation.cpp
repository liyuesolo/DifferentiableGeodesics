#include "geometrycentral/utilities/timing.h"
#include "../include/VoronoiCells.h"
bool VoronoiCells::advanceOneStep(int step)
{
    T tol = 1e-6;
    int ls_max = 12;
    // max_newton_iter = 500;
    int n_dof = samples.size() * 2;
    if (step == 0)
    {
        dirichlet_data[0] = 0.0;
        dirichlet_data[1] = 0.0;
        if (use_MMA)
        {
            mma_solver.updateDoF(n_dof, 0);
            mma_solver.SetAsymptotes(0.2, 0.65, 1.05);
        }
    }
    VectorXT residual(n_dof); residual.setZero();
    // START_TIMING(computeResidual)
    T residual_norm = computeResidual(residual);
    T energy = computeTotalEnergy();    
    if (use_MMA)
    {
        std::cout << "[MMA] iter " << step << "/" 
            << max_mma_iter << " obj: " << energy << " residual_norm " 
            << residual_norm << " tol: " << newton_tol << std::endl;
        
        if (add_length)
        {
            VectorXT targets = VectorXT::Constant(unique_voronoi_edges.size(), target_length);
            std::cout << "mean " << (current_length - targets).cwiseAbs().mean() << " max " 
                << (current_length - targets).cwiseAbs().maxCoeff() << std::endl;
        }
        
        if (residual_norm < newton_tol || step == max_mma_iter)
        {
            return true;
        }
        VectorXT design_parameters(n_dof);
        for (int i = 0; i < samples.size(); i++)
        {
            design_parameters[i * 2 + 0] = samples[i].faceCoords[0];
            design_parameters[i * 2 + 1] = samples[i].faceCoords[1];
        }
        VectorXT min_p = design_parameters.array() - 0.2, 
            max_p = design_parameters.array() + 0.2,
            current = design_parameters;
        mma_solver.UpdateEigen(design_parameters, -residual, VectorXT(), VectorXT(), min_p, max_p);
        for (int i = 0; i < samples.size(); i++)
        {
            TV2 dx = design_parameters.segment<2>(i * 2) - current.segment<2>(i*2);
            updateSurfacePoint(samples[i], dx);
        }
        constructVoronoiDiagram(true);
    }
    else
    {
        // FINISH_TIMING_PRINT(computeResidual)
        std::cout << "[NEWTON] iter " << step << "/" 
            << max_newton_iter << ": residual_norm " 
            << residual_norm << " tol: " << newton_tol << std::endl;
        
        if (residual_norm < newton_tol || step == max_newton_iter)
        {
            return true;
        }
        START_TIMING(lineSearchNewton)
        T du_norm = lineSearchNewton(residual);

        

        FINISH_TIMING_PRINT(lineSearchNewton)
        if(step == max_newton_iter || du_norm > 1e10 || du_norm < 1e-8)
            return true;
    }
    return false;
}

T VoronoiCells::computeTotalEnergy()
{
    T energy = 0.0;
    if (add_peri)
    {
        T perimeter_term = computePerimeterMinimizationEnergy(w_peri);
        energy += perimeter_term;
    }
    if (add_centroid)
    {
        T centroid_term = computeCentroidalVDEnergy(w_centroid);
        energy += centroid_term;
    }
    if (add_coplanar)
    {
        T coplanar_term = addCoplanarVDEnergy(w_co);
        energy += coplanar_term;
    }
    if (add_length)
    {
        T length_term = addSameLengthVDEnergy(w_len);
        energy += length_term;
    }
    if (add_reg)
    {
        T reg_term = addRegEnergy(w_reg);
        energy += reg_term;
    }
    return energy;
}

T VoronoiCells::computeResidual(VectorXT& residual)
{
    if (add_peri)
        addPerimeterMinimizationForceEntries(residual, w_peri);
    if (add_centroid)
        addCentroidalVDForceEntries(residual, w_centroid);
    if (add_reg)
        addRegForceEntries(residual, w_reg);
    if (add_coplanar)
        addCoplanarVDForceEntries(residual, w_co);
    if (add_length)
        addSameLengthVDForceEntries(residual, w_len);
    if (!run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
            {
                residual[offset] = 0;
            });
    return residual.norm();
}

void VoronoiCells::buildSystemMatrix(StiffnessMatrix& K)
{
    std::vector<Entry> entries;

    if (add_peri)
        addPerimeterMinimizationHessianEntries(entries, w_peri);
    if (add_centroid)
        addCentroidalVDHessianEntries(entries, w_centroid);
    if (add_reg)
        addRegHessianEntries(entries, w_reg);   
    if (add_coplanar)
        addCoplanarVDHessianEntries(entries, w_co);
    int n_dof = samples.size() * 2;
    K.resize(n_dof, n_dof);
    
    K.setFromTriplets(entries.begin(), entries.end());
    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    
    K.makeCompressed();
}

T VoronoiCells::lineSearchNewton(const VectorXT& residual)
{
    VectorXT search_direction = residual;

    if (use_Newton)
    {
        StiffnessMatrix K(residual.rows(), residual.rows());
        buildSystemMatrix(K);
        bool success = linearSolve(K, residual, search_direction);
        if (!success)
            std::cout << "linear solve failed " << std::endl;
    }
    else if (use_lbfgs)
    {
        lbfgs_solver.apply(search_direction);
        if (verbose)
            std::cout << "du dot -grad" << search_direction.normalized().dot(residual.normalized()) << std::endl;
    }

    T E0 = computeTotalEnergy();
    T alpha = 1.0;
    std::vector<SurfacePoint> samples_current = samples;
    for (int ls = 0; ls < ls_max; ls++)
    {
        samples = samples_current;
        tbb::parallel_for(0, (int)samples.size(), [&](int i)
        {
            updateSurfacePoint(samples[i], alpha * search_direction.segment<2>(i * 2));
        });
        
        constructVoronoiDiagram(true);
        T E1 = computeTotalEnergy();
        // std::cout << "E0 " << E0 << " E1 " << E1 << std::endl;
        // break;
        
        if (E1 < E0)
            break;
        alpha *= 0.5;
    }

    if (use_lbfgs)
    {
        // sk = xk+1 - xk == delta_u
        // yk = gk+1 - gk == -residual - gi
        VectorXT g1 = VectorXT::Zero(residual.rows());
        computeResidual(g1);
        if (lbfgs_first_step)
        {
            lbfgs_solver.update(alpha * search_direction, VectorXT::Zero(residual.rows()));
            lbfgs_first_step = false;   
        }
        else
            lbfgs_solver.update(alpha * search_direction, -g1 - gi);
        gi = -g1;
    }
    return alpha * search_direction.norm();
}