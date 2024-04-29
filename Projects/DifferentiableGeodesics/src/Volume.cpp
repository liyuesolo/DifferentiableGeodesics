#include "../autodiff/CST3DShell.h"
#include "../include/IntrinsicSimulation.h"

T IntrinsicSimulation::computeVolume(bool use_rest_shape)
{
    auto computeTetVolume = [&](const TV& a, const TV& b, const TV& c, const TV& d)->T
    {
        return 1.0 / 6.0 * (b - a).cross(c - a).dot(d - a);
    };
    
    T volume = 0.0;
    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices;
        if (use_rest_shape)
            vertices = getFaceVtxUndeformed(face_idx);
        else
            vertices = getFaceVtxDeformed(face_idx);
        volume += computeTetVolume(TV::Zero(), vertices.row(0), vertices.row(1), vertices.row(2));

    });
    return volume;
}

void IntrinsicSimulation::addVolumePreservationEnergy(T w, T& energy)
{
    T volume_current = computeVolume();
    energy += w * (volume_current - rest_volume) * (volume_current - rest_volume);
}

void IntrinsicSimulation::addVolumePreservationForceEntries(T w, VectorXT& residual)
{
    T volume_current = computeVolume();
    T coeff = 2.0 * w * (volume_current - rest_volume);
    
    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices = getFaceVtxDeformed(face_idx);
        FaceIdx nodal_indices = faces.segment<3>(face_idx * 3);
        Vector<T, 9> dedx;
        computeTetVolumeGradient(TV::Zero(), 
            vertices.row(0), vertices.row(1), vertices.row(2), dedx);
        addForceEntry<3>(residual, 
            {nodal_indices[0], nodal_indices[1], nodal_indices[2]}, 
            -coeff * dedx, shell_dof_start);
    });
}

void IntrinsicSimulation::addVolumePreservationHessianEntries(T w, 
    std::vector<Entry>& entries, MatrixXT& WoodBuryMatrix)
{
    T volume_current = computeVolume();
    T coeff = 2.0 * w * (volume_current - rest_volume);
    VectorXT dVdx_full = VectorXT::Zero(deformed.rows());

    iterateFaceSerial([&](int face_idx)
    {
        FaceVtx vertices = getFaceVtxDeformed(face_idx);
        FaceIdx nodal_indices = faces.segment<3>(face_idx * 3);

        Vector<T, 9> dedx;
        computeTetVolumeGradient(TV::Zero(), 
            vertices.row(0), vertices.row(1), vertices.row(2), dedx);
        Matrix<T, 9, 9> d2edx2;
        computeTetVolumeHessian(TV::Zero(), 
            vertices.row(0), vertices.row(1), vertices.row(2), d2edx2);

        addForceEntry<3>(dVdx_full, 
            {nodal_indices[0], nodal_indices[1], nodal_indices[2]}, 
            dedx, shell_dof_start);

        addHessianEntry<3, 3>(entries, 
            {nodal_indices[0], nodal_indices[1], nodal_indices[2]}, 
            coeff * d2edx2, shell_dof_start, shell_dof_start);

    });

    if (woodbury)
    {
        dVdx_full *= std::sqrt(2.0*w);
        if (!run_diff_test)
        {
            iterateDirichletDoF([&](int offset, T target)
            {
                dVdx_full[offset] = 0.0;
            });
        }
        int n_row = deformed.rows(), n_col = WoodBuryMatrix.cols();
        WoodBuryMatrix.conservativeResize(n_row, n_col + 1);
        WoodBuryMatrix.col(n_col) = dVdx_full;
    }
    else
    {
        
        int n_vtx = extrinsic_vertices.rows() / 3;
        for (int dof_i = 0; dof_i < n_vtx; dof_i++)
        {
            for (int dof_j = 0; dof_j < n_vtx; dof_j++)
            {
                VectorXT dVdx;
                getSubVector<3>(dVdx_full, {dof_i, dof_j}, dVdx, shell_dof_start);
                
                TV dVdxi = dVdx.segment<3>(0);
                TV dVdxj = dVdx.segment<3>(3);
                MatrixXT hessian_partial = 2.0 * w * dVdxi * dVdxj.transpose();
                
                addHessianBlock<3, 3>(entries, {dof_i, dof_j}, 
                hessian_partial, shell_dof_start, shell_dof_start);
            }
        }
    }
}