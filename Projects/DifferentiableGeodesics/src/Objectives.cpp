#include "../include/VoronoiCells.h"
#include "../autodiff/PointToPlane.h"
#define PARALLEL


T VoronoiCells::addRegEnergy(T w)
{
    T energy = 0.0;
    for (int i = 0; i < samples.size(); i++)
    {
        TV si = toTV(samples[i].interpolate(geometry->vertexPositions));
        TV Si = toTV(samples_rest[i].interpolate(geometry->vertexPositions));
        energy += 0.5 * (si - Si).squaredNorm();
    }
    return energy * w;
}
void VoronoiCells::addRegForceEntries(VectorXT& grad, T w)
{
    
    for (int i = 0; i < samples.size(); i++)
    {
        TV si = toTV(samples[i].interpolate(geometry->vertexPositions));
        TV Si = toTV(samples_rest[i].interpolate(geometry->vertexPositions));
        TV dOdx = w * (si - Si);
        Matrix<T, 3, 2> dxdw;
        computeSurfacePointdxds(samples[i], dxdw);
        TV2 dOds = dOdx.transpose() * dxdw;
        addForceEntry<2>(grad, {i}, -dOds);
    }
    
}

void VoronoiCells::addRegHessianEntries(std::vector<Entry>& entries, T w)
{
    for (int i = 0; i < samples.size(); i++)
    {
        TV si = toTV(samples[i].interpolate(geometry->vertexPositions));
        TV Si = toTV(samples_rest[i].interpolate(geometry->vertexPositions));
        TV dOdx = (si - Si);
        Matrix<T, 3, 2> dxdw;
        computeSurfacePointdxds(samples[i], dxdw);
        TV2 dOds = dOdx.transpose() * dxdw;
        TM2 d2Ods2 = w * dOds * dOds.transpose();
        addHessianEntry<2, 2>(entries, {i}, d2Ods2);
    }
}

void VoronoiCells::diffTestScale()
{
    VectorXT grad(samples.size() * 2); grad.setZero();
    T E0 = computeTotalEnergy();
    T g_norm = computeResidual(grad);    
    std::cout << "|g| " << g_norm << std::endl;
    std::vector<SurfacePoint> samples_current = samples;
    VectorXT dx(samples.size() * 2);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    dx *= 0.01;
    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {
        samples = samples_current;
        tbb::parallel_for(0, (int)samples.size(), [&](int i){
            updateSurfacePoint(samples[i], dx.segment<2>(i * 2));
        });
        constructVoronoiDiagram(true);
        if (add_length)
        {
            retrace = true;
            traceGeodesics();
        }
        T E1 = computeTotalEnergy();
        T dE = E1 - E0;
        dE -= grad.dot(dx);
        std::cout << "dE " << dE << std::endl;
        if (i > 0)
        {
            std::cout << (previous/dE) << std::endl;
        }
        previous = dE;
        dx *= 0.5;
    }
    samples = samples_current;
}

void VoronoiCells::diffTest()
{
    T epsilon = 1e-6;
    std::vector<SurfacePoint> samples_current = samples;
    int n_dof = samples.size();
    VectorXT gradient_FD(n_dof); gradient_FD.setZero();
    VectorXT gradient(n_dof); gradient.setZero();
    T _E0;
    T g_norm = computeResidual(gradient);    
    gradient *= -1.0;

    for (int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        int idx = std::floor(dof_i / 2);
        int dim = dof_i % 2;
        TV2 dx = TV2::Zero();
        dx[dim] += epsilon;
        updateSurfacePoint(samples[idx], dx);
        constructVoronoiDiagram(true);
        if (add_length)
        {
            retrace = true;
            traceGeodesics();
        }
        T E1 = computeTotalEnergy();
        samples = samples_current;

        dx[dim] -= 2.0 * epsilon;
        updateSurfacePoint(samples[idx], dx);
        constructVoronoiDiagram(true);
        if (add_length)
        {
            retrace = true;
            traceGeodesics();
        }
        T E0 = computeTotalEnergy();
        samples = samples_current;

        gradient_FD(dof_i) = (E1 - E0) / (2.0 * epsilon);

        // if( std::abs(gradient_FD(dof_i)) < 1e-8 && std::abs(gradient(dof_i)) < 1e-8)
        //     continue;
        if (std::abs( gradient_FD(dof_i) - gradient(dof_i)) < 1e-3 * std::abs(gradient(dof_i)))
            continue;

        std::cout << " dof " << dof_i << " vtx " << std::floor(dof_i/2.0) << " FD " << gradient_FD(dof_i) << " symbolic " << gradient(dof_i) << std::endl;
        
        std::getchar();
    }
    
    samples = samples_current;
}

void VoronoiCells::checkLengthObjective(T& mean_val, T& max_val)
{
    VectorXT targets = VectorXT::Constant(unique_voronoi_edges.size(), target_length);
    mean_val = (current_length - targets).cwiseAbs().mean() / target_length;
    max_val = (current_length - targets).cwiseAbs().maxCoeff() / target_length;
}

T VoronoiCells::addSameLengthVDEnergy(T w)
{
    T energy = 0.0;
    // VectorXT targets = VectorXT::Constant(unique_voronoi_edges.size(), target_length);
    // energy = w * 0.5 * (current_length - targets).dot(current_length - targets);
    int n_edges = unique_voronoi_edges.size();
    for (int edge_i = 0; edge_i < n_edges; edge_i++)
    {
        T coeff = current_length[edge_i] - target_length;
        energy += 0.5 * coeff * coeff;
    }
    return energy;
}

void VoronoiCells::addSameLengthVDForceEntries(VectorXT& grad, T w)
{
    VectorXT dlds = VectorXT::Zero(samples.size() * 2);
    StiffnessMatrix dxds_all_nodes;
    computeDxDsAllNodes(dxds_all_nodes);

    int n_edges = unique_voronoi_edges.size();
    for (int edge_i = 0; edge_i < n_edges; edge_i++)
    {
        T coeff = current_length[edge_i] - target_length;
        Vector<T, 6> dldx;
        computeGeodesicLengthGradient(unique_voronoi_edges[edge_i], edge_i, dldx);
        for (int dim = 0; dim < 2; dim++)
        {
            int ixn_idx = unique_voronoi_edges[edge_i][dim];
            for (int idx : unique_ixn_points[ixn_idx].second)
            {
                TV2 dldsi = coeff * dldx.segment<3>(dim * 3).transpose() * dxds_all_nodes.block(ixn_idx * 3, idx * 2, 3, 2);
                addForceEntry<2>(dlds, {idx}, dldsi);
            }
        }
        
    }
    grad += -w * dlds;
}

void VoronoiCells::addSameLengthVDHessianEntries(std::vector<Entry>& entries, T w)
{

}

void VoronoiCells::checkCoplanarity(T& mean_val, T& max_val)
{
    std::vector<T> distances;
    iterateVoronoiCells([&](const VoronoiCellData& cell_data, int cell_idx)
    {
        int edge_cnt = 0;
        int n_voro_cell_edge = cell_data.cell_edge_lengths.size();
        int n_cell_node = cell_data.cell_vtx_nodes.size();
        MatrixXT points(n_cell_node, 3);
        int cnt = 0;
        for (int idx : cell_data.cell_vtx_nodes)
        {
            auto ixn = unique_ixn_points[cell_data.cell_nodes[idx]];
            SurfacePoint xi = ixn.first;
            points.row(cnt++) = toTV(xi.interpolate(geometry->vertexPositions));
        }
        MatrixXT A(n_cell_node, 3);
		VectorXT rhs(n_cell_node); 
		A.col(2).setConstant(1.0);
		for (int j = 0; j < n_cell_node; j++)
		{
			for (int i = 0; i < 2; i++)
				A(j, i) = points(j, i);
			rhs[j] = points(j, 2);
		}
		TM ATA = A.transpose() * A;

		TV coeff = ATA.inverse() * (A.transpose() * rhs);

        // Compute the normal equation coefficients
        // MatrixXT ATA = A.transpose() * A;
        // VectorXT ATb = A.transpose() * points.col(2);

        // TV coeff = ATA.ldlt().solve(ATb);

        T a = coeff[0], b = coeff[1], c = coeff[2];
		
		T denom = std::sqrt(a * a + b * b);
		
		for (int i = 0; i < n_cell_node; i++)
		{
            T nom = std::abs(a * points(i, 0) + b * points(i, 1) + c - points(i, 2));
            T dis = nom / denom;
            distances.push_back(dis);
		}
    });
    VectorXT _distance(distances.size());
    for (int i = 0; i < distances.size(); i++)
    {
        _distance[i] = distances[i];
    }
    // std::cout << "mean " << _distance.mean() << " max " << _distance.maxCoeff() << std::endl;
    mean_val = _distance.mean();
    max_val = _distance.maxCoeff();
    
}

T VoronoiCells::addCoplanarVDEnergy(T w)
{
    VectorXT energies(voronoi_cell_data.size());
    energies.setZero();

    iterateVoronoiCells([&](const VoronoiCellData& cell_data, int cell_idx)
    {
        int edge_cnt = 0;
        int n_voro_cell_edge = cell_data.cell_edge_lengths.size();
        // std::cout << cell_data.cell_vtx_nodes.size() << std::endl;
        std::vector<TV> points;
        for (int idx : cell_data.cell_vtx_nodes)
        {
            auto ixn = unique_ixn_points[cell_data.cell_nodes[idx]];
            SurfacePoint xi = ixn.first;
            points.push_back(toTV(xi.interpolate(geometry->vertexPositions)));
        }
        T ei = 0.0;
        if (cell_data.cell_vtx_nodes.size() == 4)
            compute4PointsPlaneFittingEnergy(points[0], points[1], points[2], points[3], ei);
        else if (cell_data.cell_vtx_nodes.size() == 5)
            compute5PointsPlaneFittingEnergy(points[0], points[1], points[2], points[3], points[4], ei);
        else if (cell_data.cell_vtx_nodes.size() == 6)
            compute6PointsPlaneFittingEnergy(points[0], points[1], points[2], points[3], points[4], points[5], ei);
        else if (cell_data.cell_vtx_nodes.size() == 7)
            compute7PointsPlaneFittingEnergy(points[0], points[1], points[2], points[3], points[4], points[5], points[6], ei);
        else if (cell_data.cell_vtx_nodes.size() == 8)
            compute8PointsPlaneFittingEnergy(points[0], points[1], points[2], points[3], points[4], points[5], points[6], points[7], ei);
        else if (cell_data.cell_vtx_nodes.size() == 9)
            compute9PointsPlaneFittingEnergy(points[0], points[1], points[2], points[3], points[4], points[5], points[6], points[7], points[8], ei);
        else
        {
            // std::cout << "" << std::endl;
        }
        energies[cell_idx] += ei / T(cell_data.cell_vtx_nodes.size());
    });
    return energies.sum();
}


void VoronoiCells::addCoplanarVDForceEntries(VectorXT& grad, T w)
{
    
    int n_dof = samples.size() * 2;
    StiffnessMatrix dxds_all_nodes;
    computeDxDsAllNodes(dxds_all_nodes);
    std::vector<VectorXT> grads(voronoi_cell_data.size(), 
            VectorXT::Zero(n_sites * 2));
    iterateVoronoiCells([&](const VoronoiCellData& cell_data, int cell_idx)
    {
        int edge_cnt = 0;
        int n_voro_cell_edge = cell_data.cell_edge_lengths.size();
        // std::cout << cell_data.cell_vtx_nodes.size() << std::endl;
        std::vector<TV> points;
        for (int idx : cell_data.cell_vtx_nodes)
        {
            auto ixn = unique_ixn_points[cell_data.cell_nodes[idx]];
            SurfacePoint xi = ixn.first;
            points.push_back(toTV(xi.interpolate(geometry->vertexPositions)));
        }

        int n_cell_node = cell_data.cell_vtx_nodes.size();
        
        VectorXT dedx(n_cell_node * 3);
        dedx.setZero();
        if (cell_data.cell_vtx_nodes.size() == 4)
        {
            Vector<T, 12> _dedx;
            compute4PointsPlaneFittingEnergyGradient(points[0], points[1], points[2], points[3], _dedx);            
            dedx.head<12>() = _dedx;
        }
        else if (cell_data.cell_vtx_nodes.size() == 5)
        {
            Vector<T, 15> _dedx;
            compute5PointsPlaneFittingEnergyGradient(points[0], points[1], points[2], points[3], points[4], _dedx);            
            dedx.head<15>() = _dedx;
        }
        else if (cell_data.cell_vtx_nodes.size() == 6)
        {
            Vector<T, 18> _dedx;
            compute6PointsPlaneFittingEnergyGradient(points[0], points[1], points[2], points[3], points[4], points[5], _dedx);            
            dedx.head<18>() = _dedx;
        }
        else if (cell_data.cell_vtx_nodes.size() == 7)
        {
            Vector<T, 21> _dedx;
            compute7PointsPlaneFittingEnergyGradient(points[0], points[1], points[2], points[3], points[4], points[5], points[6], _dedx);            
            dedx.head<21>() = _dedx;
        }
        else if (cell_data.cell_vtx_nodes.size() == 8)
        {
            Vector<T, 24> _dedx;
            compute8PointsPlaneFittingEnergyGradient(points[0], points[1], points[2], points[3], points[4], points[5], points[6], points[7], _dedx);            
            dedx.head<24>() = _dedx;
        }
        else if (cell_data.cell_vtx_nodes.size() == 9)
        {
            Vector<T, 27> _dedx;
            compute9PointsPlaneFittingEnergyGradient(points[0], points[1], points[2], points[3], points[4], points[5], points[6], points[7], points[8], _dedx);            
            dedx.head<27>() = _dedx;
        }
        else 
        {
            std::cout << "n_cell_node " << n_cell_node  << std::endl;
        }
        dedx /= T(cell_data.cell_vtx_nodes.size());
        for (int i = 0; i < n_cell_node; i++)
        {
            auto site_indices = unique_ixn_points[cell_data.cell_nodes[cell_data.cell_vtx_nodes[i]]].second;
            for (int site_idx : site_indices)
            {
                TV2 dedw = dedx.segment<3>(i * 3).transpose() * 
                                    dxds_all_nodes.block(cell_data.cell_nodes[cell_data.cell_vtx_nodes[i]] * 3,
                                    site_idx * 2, 3, 2);
                addForceEntry<2>(grads[cell_idx], {site_idx}, dedw);
                // std::cout << dedw.transpose() << std::endl;
            }
        }
    });
    
    for (int i = 0; i < voronoi_cell_data.size(); i++)
    {
        grad += -w * grads[i];
    }
    // std::cout << grad.norm() << std::endl;
    // std::getchar();
}

void VoronoiCells::addCoplanarVDHessianEntries(std::vector<Entry>& entries, T w)
{
    

}

T VoronoiCells::computeCentroidalVDEnergy(T w)
{
    if (edge_weighting)
    {
        VectorXT energies(voronoi_cell_data.size());
        energies.setZero();
        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% iterate over cells %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        iterateVoronoiCellsParallel([&](const VoronoiCellData& cell_data, int cell_idx)
        {
            int edge_cnt = 0;
            int n_voro_cell_edge = cell_data.cell_edge_lengths.size();

            for (int idx : cell_data.cell_vtx_nodes)
            {
                auto ixn = unique_ixn_points[cell_data.cell_nodes[idx]];
                SurfacePoint xi = ixn.first;
                SurfacePoint si = samples[cell_idx];
                std::vector<IxnData> ixn_data_site;
                std::vector<SurfacePoint> path;
                T dis;
                computeGeodesicDistance(xi, si, dis, path, ixn_data_site, false);
                
                int left_idx = (edge_cnt - 1 + n_voro_cell_edge) % n_voro_cell_edge;
                int right_idx = edge_cnt;
                // T left_seg_len = cell_data.cell_edge_lengths[left_idx];
                // T right_seg_len = cell_data.cell_edge_lengths[right_idx];

                int left_node_idx = cell_data.cell_nodes[cell_data.cell_vtx_nodes[left_idx]];
                int right_node_idx = cell_data.cell_nodes[cell_data.cell_vtx_nodes[(edge_cnt+1)%cell_data.cell_vtx_nodes.size()]];

                TV left_node = toTV(unique_ixn_points[left_node_idx].first.interpolate(geometry->vertexPositions));
                TV right_node = toTV(unique_ixn_points[right_node_idx].first.interpolate(geometry->vertexPositions));
                TV current_node = toTV(xi.interpolate(geometry->vertexPositions));
                T left_seg_len = (left_node - current_node).norm();
                T right_seg_len = (right_node - current_node).norm();
                T avg = 0.5 * (left_seg_len + right_seg_len);
                // avg = 1.0;
                energies[cell_idx] += 0.5 * w * avg * dis * dis * cell_weights[cell_idx];
                
                edge_cnt ++;
            }
            
        });
        return energies.sum();
    }
    else
    {
        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% iterate over intersection points %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        VectorXT energies(unique_ixn_points.size());
        energies.setZero();
    #ifdef PARALLEL
        tbb::parallel_for(0, (int)unique_ixn_points.size(),[&](int i)
    #else
        for (int i = 0; i < unique_ixn_points.size(); i++)
    #endif
        {
            auto ixn = unique_ixn_points[i];
            SurfacePoint xi = ixn.first;
            std::vector<int> site_indices = ixn.second;
            if (site_indices.size() == 2)
    #ifdef PARALLEL
                return;
    #else
                continue;
    #endif
            for (int idx : site_indices)
            {
                SurfacePoint si = samples[idx];
                std::vector<IxnData> ixn_data_site;
                std::vector<SurfacePoint> path;
                T dis;
                computeGeodesicDistance(xi, si, dis, path, ixn_data_site, false);
                energies[i] += 0.5 * w * dis * dis * cell_weights[idx];
            }   
            
        }
    #ifdef PARALLEL
        );
    #endif
        return energies.sum();

    }
}

void VoronoiCells::computeDxDsAllNodes(StiffnessMatrix& dxds)
{
    dxds.resize(unique_ixn_points.size() * 3, samples.size() * 2);
    std::vector<Entry> dxds_entries;
    std::vector<std::vector<Entry>> dxds_entry_threads(unique_ixn_points.size(), std::vector<Entry>());
#ifdef PARALLEL
    tbb::parallel_for(0, (int)unique_ixn_points.size(),[&](int i)
#else
    for (int i = 0; i < unique_ixn_points.size(); i++)
#endif
    {
        auto ixn = unique_ixn_points[i];
        SurfacePoint xi = ixn.first;
        std::vector<int> site_indices = ixn.second;
        MatrixXT dx_ds;
        bool valid = computeDxDs(xi, site_indices, dx_ds, /*reduced = */false);
        if (valid)
            addJacobianEntry<3, 2>(dxds_entry_threads[i], {i}, site_indices, dx_ds);
    }
#ifdef PARALLEL
    );
#endif
    for (auto data_thread : dxds_entry_threads)
    {
        dxds_entries.insert(dxds_entries.end(), data_thread.begin(), data_thread.end());
    }
    dxds.setFromTriplets(dxds_entries.begin(), dxds_entries.end());
}

T VoronoiCells::computeCentroidalVDGradient(VectorXT& grad, T& energy, T w)
{
    if (edge_weighting)
    {
        VectorXT energies(voronoi_cell_data.size());
        energies.setZero();
        
        grad.resize(n_sites * 2); grad.setZero();
        std::vector<VectorXT> grads(
            voronoi_cell_data.size(), 
            VectorXT::Zero(n_sites * 2));
        // x 3d intersection points compute jacobian afterwards
        // s is 2d barycentric coordinates
        // compute dxds for all nodes first
        
        StiffnessMatrix dxds_all_nodes;
        computeDxDsAllNodes(dxds_all_nodes);
        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% iterate over cells %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        iterateVoronoiCellsParallel([&](const VoronoiCellData& cell_data, int cell_idx)
        {
            int edge_cnt = 0;
            int n_voro_cell_edge = cell_data.cell_vtx_nodes.size();
            VtxList vtx_list = voronoi_cell_vertices[cell_idx];
            // std::cout << "----- cell " << cell_idx << " -----" << std::endl;
            // for (int idx : cell_data.cell_vtx_nodes)
            //     std::cout << cell_data.cell_nodes[idx] << " ";
            // std::cout << std::endl;
            for (int idx : cell_data.cell_vtx_nodes)
            {
                auto ixn = unique_ixn_points[cell_data.cell_nodes[idx]];
                SurfacePoint xi = ixn.first;
                std::vector<int> site_indices = ixn.second;
                SurfacePoint si = samples[cell_idx];
                Vector<T, 5> dldw;
                T dis = computeGeodesicLengthAndGradient(xi, si, dldw);

                int left_idx = (edge_cnt - 1 + n_voro_cell_edge) % n_voro_cell_edge;
                int right_idx = edge_cnt;
                // T left_seg_len = cell_data.cell_edge_lengths[left_idx];
                // T right_seg_len = cell_data.cell_edge_lengths[right_idx];
                int left_node_idx = cell_data.cell_nodes[cell_data.cell_vtx_nodes[left_idx]];
                int right_node_idx = cell_data.cell_nodes[cell_data.cell_vtx_nodes[(edge_cnt+1)%cell_data.cell_vtx_nodes.size()]];
                
                TV left_node = toTV(unique_ixn_points[left_node_idx].first.interpolate(geometry->vertexPositions));
                TV right_node = toTV(unique_ixn_points[right_node_idx].first.interpolate(geometry->vertexPositions));
                TV current_node = toTV(xi.interpolate(geometry->vertexPositions));

                T left_seg_len = (left_node - current_node).norm();
                T right_seg_len = (right_node - current_node).norm();
                
                T avg = 0.5 * (left_seg_len + right_seg_len);
                // avg = 1.0;
                energies[cell_idx] += 0.5 * avg * dis * dis * cell_weights[cell_idx];

                T weights_term = 0.5 * cell_weights[cell_idx];

                TV pOpx = dldw.segment<3>(0);
                TV2 pOps = dldw.segment<2>(3);
                
                for (int j = 0; j < site_indices.size(); j++)
                {
                    if (site_indices[j] == cell_idx)
                    {
                        TV2 first_term = weights_term * avg * 2.0 * dis * (
                            (
                                pOpx.transpose() * 
                                dxds_all_nodes.block(cell_data.cell_nodes[idx] * 3, 
                                cell_idx * 2, 3, 2)
                            ).transpose()
                            + pOps);
                        // std::cout << "first term " << first_term.transpose() << std::endl;
                        addForceEntry<2>(grads[cell_idx], {cell_idx}, first_term);
                    }
                    else
                    {
                        TV2 first_term = weights_term * avg * 2.0 * dis * (pOpx.transpose() * 
                                        dxds_all_nodes.block(cell_data.cell_nodes[idx] * 3, site_indices[j] * 2, 3, 2)
                                        ).transpose();
                        addForceEntry<2>(grads[cell_idx], {site_indices[j]}, first_term);
                    }

                }

                // iterate over voronoi edges in a single cell
                // compute the derivative of voronoi edge length w.r.t site positions
                Vector<T, 6> dldxleft; 
                dldxleft.segment<3>(0) = -(current_node - left_node).normalized();
                dldxleft.segment<3>(3) = (current_node - left_node).normalized();
                int n_node_left = unique_ixn_points[left_node_idx].second.size();
                int n_node_right = site_indices.size();
                MatrixXT dxds(6, (n_node_left + n_node_right) * 2); dxds.setZero();
                std::vector<int> idx_list;
                int cnt = 0;
                for (int site_idx : unique_ixn_points[left_node_idx].second)
                {
                    dxds.block(0, cnt * 2, 3, 2) += dxds_all_nodes.block(left_node_idx * 3, site_idx * 2, 3, 2);
                    cnt++;
                    idx_list.push_back(site_idx);
                }
                for (int site_idx : site_indices)
                {
                    dxds.block(3, cnt * 2, 3, 2) += dxds_all_nodes.block(cell_data.cell_nodes[idx] * 3, site_idx * 2, 3, 2);
                    cnt++;
                    idx_list.push_back(site_idx);
                }
                VectorXT dlds = weights_term * dis * dis * 0.5 * dldxleft.transpose() * dxds;
                addForceEntry<2>(grads[cell_idx], idx_list, dlds);

                Vector<T, 6> dldxright; 
                dldxright.segment<3>(0) = -(right_node - current_node).normalized();
                dldxright.segment<3>(3) = (right_node - current_node).normalized();
                n_node_left = site_indices.size();
                n_node_right = unique_ixn_points[right_node_idx].second.size();
                dxds.resize(6, (n_node_left + n_node_right) * 2); dxds.setZero();
                idx_list.resize(0);
                cnt = 0;
                for (int site_idx : site_indices)
                {
                    dxds.block(0, cnt * 2, 3, 2) += dxds_all_nodes.block(cell_data.cell_nodes[idx] * 3, site_idx * 2, 3, 2);
                    cnt++;
                    idx_list.push_back(site_idx);
                }
                for (int site_idx : unique_ixn_points[right_node_idx].second)
                {
                    dxds.block(3, cnt * 2, 3, 2) += dxds_all_nodes.block(right_node_idx * 3, site_idx * 2, 3, 2);
                    cnt++;
                    idx_list.push_back(site_idx);
                }
                dlds = weights_term * dis * dis * 0.5 * dldxright.transpose() * dxds;
                addForceEntry<2>(grads[cell_idx], idx_list, dlds);

                // int left_node_location = cell_data.cell_vtx_nodes[left_idx];
                // int right_node_location = left_idx < right_idx ? 
                //     // last node
                //     cell_data.cell_vtx_nodes[(right_idx+1)%cell_data.cell_vtx_nodes.size()] 
                //     : cell_data.cell_vtx_nodes[1] + cell_data.cell_nodes.size();
                // if (right_node_location == 0)
                //     right_node_location = cell_data.cell_nodes.size();
                
                // for (int j = left_node_location; j < right_node_location; j++)
                // {
                //     int left_idx_in_range = cell_data.cell_nodes[j % cell_data.cell_nodes.size()];
                //     int right_idx_in_range = cell_data.cell_nodes[(j + 1) % cell_data.cell_nodes.size()];
                //     TV xj = toTV(unique_ixn_points[left_idx_in_range].first.interpolate(geometry->vertexPositions));
                //     TV xk = toTV(unique_ixn_points[right_idx_in_range].first.interpolate(geometry->vertexPositions));
                //     TV dldxj = -(xk - xj).normalized();
                //     TV dldxk = (xk - xj).normalized();
                //     Vector<T, 6> dldx; dldx.segment<3>(0) = dldxj; dldx.segment<3>(3) = dldxk;
                //     if ((xj-xk).norm() < 1e-10)
                //         continue;
                //     int n_node_left = unique_ixn_points[left_idx_in_range].second.size();
                //     int n_node_right = unique_ixn_points[right_idx_in_range].second.size();
                //     MatrixXT dxds(6, (n_node_left + n_node_right) * 2); dxds.setZero();
                //     std::vector<int> idx_list;
                //     int cnt = 0;
                //     for (int site_idx : unique_ixn_points[left_idx_in_range].second)
                //     {
                //         dxds.block(0, cnt * 2, 3, 2) += dxds_all_nodes.block(left_idx_in_range * 3, site_idx * 2, 3, 2);
                //         cnt++;
                //         idx_list.push_back(site_idx);
                //     }
                //     for (int site_idx : unique_ixn_points[right_idx_in_range].second)
                //     {
                //         dxds.block(3, cnt * 2, 3, 2) += dxds_all_nodes.block(right_idx_in_range * 3, site_idx * 2, 3, 2);
                //         cnt++;
                //         idx_list.push_back(site_idx);
                //     }
                //     VectorXT dlds = weights_term * dis * dis * 0.5 * dldx.transpose() * dxds;
                //     addForceEntry<2>(grads[cell_idx], idx_list, dlds);
                //     // for (int site_idx : unique_ixn_points[left_idx_in_range].second)
                //     // {
                //     //     TV2 second_term = weights_term * dis * dis * 0.5 * 
                //     //         dldxj.transpose() * dxds_all_nodes.block(left_idx_in_range * 3, site_idx * 2, 3, 2);
                //     //     addForceEntry<2>(grads[cell_idx], {site_idx}, second_term);
                //     // }
                //     // for (int site_idx : unique_ixn_points[right_idx_in_range].second)
                //     // {
                //     //     TV2 second_term = weights_term * dis * dis * 0.5 * 
                //     //         dldxk.transpose() * dxds_all_nodes.block(right_idx_in_range * 3, site_idx * 2, 3, 2);
                //     //     addForceEntry<2>(grads[cell_idx], {site_idx}, second_term);
                //     // }
                // }
                edge_cnt ++;
            }
            // std::getchar();
        });
        energy = w * energies.sum();
        for (int i = 0; i < voronoi_cell_data.size(); i++)
        {
            grad += w * grads[i];
        }
        return grad.norm();
    }
    else
    {
        grad.resize(n_sites * 2); grad.setZero();
        VectorXT energies(unique_ixn_points.size());
        energies.setZero();
        std::vector<VectorXT> grads((int)unique_ixn_points.size(), 
            VectorXT::Zero(n_sites * 2));

        StiffnessMatrix dxds_all_nodes;
        computeDxDsAllNodes(dxds_all_nodes);
        
    #ifdef PARALLEL
        tbb::parallel_for(0, (int)unique_ixn_points.size(),[&](int ixn_idx)
    #else
        for (int ixn_idx = 0; ixn_idx < unique_ixn_points.size(); ixn_idx++)
    #endif
        {
            auto ixn = unique_ixn_points[ixn_idx];
            SurfacePoint xi = ixn.first;
            std::vector<int> site_indices = ixn.second;
            if (site_indices.size() == 2)
    #ifdef PARALLEL
                return;
    #else
                continue;
    #endif
            
            for (int i = 0; i < site_indices.size(); i++)
            {
                int idx = site_indices[i];
                SurfacePoint si = samples[idx];
                
                Vector<T, 5> dldw;
                T dis = computeGeodesicLengthAndGradient(xi, si, dldw);

                energies[ixn_idx] += 0.5 * dis * dis * cell_weights[idx];
                T dOdl = dis * cell_weights[idx];

                TV pOpx = dldw.segment<3>(0);
                TV2 pOps = dldw.segment<2>(3);

                TV2 dOds = dOdl * ((pOpx.transpose() * 
                                dxds_all_nodes.block(ixn_idx * 3, idx * 2, 3, 2)
                                ).transpose()
                            + pOps);
                
                addForceEntry<2>(grads[ixn_idx], {idx}, dOds);
                // x is function of all s
                for (int j = 0; j < site_indices.size(); j++)
                {
                    if (j==i) continue;
                    dOds = dOdl * (pOpx.transpose() * 
                                    dxds_all_nodes.block(ixn_idx * 3, site_indices[j] * 2, 3, 2)
                                    ).transpose();
                    addForceEntry<2>(grads[ixn_idx], {site_indices[j]}, dOds);
                }
            }
        }
    #ifdef PARALLEL
        );
    #endif
        for (int i = 0; i < unique_ixn_points.size(); i++)
        {
            grad += w * grads[i];
        }
        energy = w * energies.sum();
        return grad.norm();

    }
}

void VoronoiCells::addCentroidalVDForceEntries(VectorXT& grad, T w)
{
    int n_dof = samples.size() * 2;
    

    if (edge_weighting)
    {
        std::vector<VectorXT> grads(
            voronoi_cell_data.size(), 
            VectorXT::Zero(n_sites * 2));
        // x 3d intersection points compute jacobian afterwards
        // s is 2d barycentric coordinates
        // compute dxds for all nodes first
        
        StiffnessMatrix dxds_all_nodes;
        computeDxDsAllNodes(dxds_all_nodes);
        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% iterate over cells %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        iterateVoronoiCellsParallel([&](const VoronoiCellData& cell_data, int cell_idx)
        {
            int edge_cnt = 0;
            int n_voro_cell_edge = cell_data.cell_edge_lengths.size();
            VtxList vtx_list = voronoi_cell_vertices[cell_idx];
            // std::cout << "----- cell " << cell_idx << " -----" << std::endl;
            for (int idx : cell_data.cell_vtx_nodes)
            {
                auto ixn = unique_ixn_points[cell_data.cell_nodes[idx]];
                SurfacePoint xi = ixn.first;
                std::vector<int> site_indices = ixn.second;
                SurfacePoint si = samples[cell_idx];
                Vector<T, 5> dldw;
                T dis = computeGeodesicLengthAndGradient(xi, si, dldw);

                int left_idx = (edge_cnt - 1 + n_voro_cell_edge) % n_voro_cell_edge;
                int right_idx = edge_cnt;
                // std::cout << "left idx " << left_idx << " right idx " << right_idx << std::endl;

                int left_node_idx = cell_data.cell_nodes[cell_data.cell_vtx_nodes[left_idx]];
                int right_node_idx = cell_data.cell_nodes[cell_data.cell_vtx_nodes[(edge_cnt+1)%cell_data.cell_vtx_nodes.size()]];
                
                TV left_node = toTV(unique_ixn_points[left_node_idx].first.interpolate(geometry->vertexPositions));
                TV right_node = toTV(unique_ixn_points[right_node_idx].first.interpolate(geometry->vertexPositions));
                TV current_node = toTV(xi.interpolate(geometry->vertexPositions));

                T left_seg_len = (left_node - current_node).norm();
                T right_seg_len = (right_node - current_node).norm();

                // T left_seg_len = cell_data.cell_edge_lengths[left_idx];
                // T right_seg_len = cell_data.cell_edge_lengths[right_idx];
                T avg = 0.5 * (left_seg_len + right_seg_len);
                

                T weights_term = 0.5 * cell_weights[cell_idx];

                TV pOpx = dldw.segment<3>(0);
                TV2 pOps = dldw.segment<2>(3);

                TV2 first_term = weights_term * avg * 2.0 * dis * (
                    (
                        pOpx.transpose() * 
                        dxds_all_nodes.block(cell_data.cell_nodes[idx] * 3, 
                        cell_idx * 2, 3, 2)
                    ).transpose()
                    + pOps);

                addForceEntry<2>(grads[cell_idx], {cell_idx}, first_term);
                for (int j = 0; j < site_indices.size(); j++)
                {
                    if (site_indices[j] == cell_idx) continue;
                    first_term = weights_term * avg * 2.0 * dis * (pOpx.transpose() * 
                                    dxds_all_nodes.block(cell_data.cell_nodes[idx] * 3, site_indices[j] * 2, 3, 2)
                                    ).transpose();
                    addForceEntry<2>(grads[cell_idx], {site_indices[j]}, first_term);
                }

                // iterate over voronoi edges in a single cell
                // compute the derivative of voronoi edge length w.r.t site positions
                Vector<T, 6> dldxleft; 
                dldxleft.segment<3>(0) = -(current_node - left_node).normalized();
                dldxleft.segment<3>(3) = (current_node - left_node).normalized();
                int n_node_left = unique_ixn_points[left_node_idx].second.size();
                int n_node_right = site_indices.size();
                MatrixXT dxds(6, (n_node_left + n_node_right) * 2); dxds.setZero();
                std::vector<int> idx_list;
                int cnt = 0;
                for (int site_idx : unique_ixn_points[left_node_idx].second)
                {
                    dxds.block(0, cnt * 2, 3, 2) += dxds_all_nodes.block(left_node_idx * 3, site_idx * 2, 3, 2);
                    cnt++;
                    idx_list.push_back(site_idx);
                }
                for (int site_idx : site_indices)
                {
                    dxds.block(3, cnt * 2, 3, 2) += dxds_all_nodes.block(cell_data.cell_nodes[idx] * 3, site_idx * 2, 3, 2);
                    cnt++;
                    idx_list.push_back(site_idx);
                }
                VectorXT dlds = weights_term * dis * dis * 0.5 * dldxleft.transpose() * dxds;
                addForceEntry<2>(grads[cell_idx], idx_list, dlds);

                Vector<T, 6> dldxright; 
                dldxright.segment<3>(0) = -(right_node - current_node).normalized();
                dldxright.segment<3>(3) = (right_node - current_node).normalized();
                n_node_left = site_indices.size();
                n_node_right = unique_ixn_points[right_node_idx].second.size();
                dxds.resize(6, (n_node_left + n_node_right) * 2); dxds.setZero();
                idx_list.resize(0);
                cnt = 0;
                for (int site_idx : site_indices)
                {
                    dxds.block(0, cnt * 2, 3, 2) += dxds_all_nodes.block(cell_data.cell_nodes[idx] * 3, site_idx * 2, 3, 2);
                    cnt++;
                    idx_list.push_back(site_idx);
                }
                for (int site_idx : unique_ixn_points[right_node_idx].second)
                {
                    dxds.block(3, cnt * 2, 3, 2) += dxds_all_nodes.block(right_node_idx * 3, site_idx * 2, 3, 2);
                    cnt++;
                    idx_list.push_back(site_idx);
                }
                dlds = weights_term * dis * dis * 0.5 * dldxright.transpose() * dxds;
                addForceEntry<2>(grads[cell_idx], idx_list, dlds);
                // int left_node_location = cell_data.cell_vtx_nodes[left_idx];
                // int right_node_location = left_idx < right_idx ? 
                //     // last node
                //     cell_data.cell_vtx_nodes[(right_idx+1)%cell_data.cell_vtx_nodes.size()] 
                //     : cell_data.cell_vtx_nodes[1] + cell_data.cell_nodes.size();
                // if (right_node_location == 0)
                //     right_node_location = cell_data.cell_nodes.size();
                
                // for (int j = left_node_location; j < right_node_location; j++)
                // {
                //     int left_idx_in_range = cell_data.cell_nodes[j % cell_data.cell_nodes.size()];
                //     int right_idx_in_range = cell_data.cell_nodes[(j + 1) % cell_data.cell_nodes.size()];
                //     TV xj = toTV(unique_ixn_points[left_idx_in_range].first.interpolate(geometry->vertexPositions));
                //     TV xk = toTV(unique_ixn_points[right_idx_in_range].first.interpolate(geometry->vertexPositions));
                //     TV dldxj = -(xk - xj).normalized();
                //     TV dldxk = (xk - xj).normalized();
                //     if ((xj-xk).norm() < 1e-10)
                //         continue;
                    
                //     for (int site_idx : unique_ixn_points[left_idx_in_range].second)
                //     {
                //         TV2 second_term = weights_term * dis * dis * 0.5 * 
                //             dldxj.transpose() * dxds_all_nodes.block(left_idx_in_range * 3, site_idx * 2, 3, 2);
                //         addForceEntry<2>(grads[cell_idx], {site_idx}, second_term);
                //         // std::cout << dldxj.transpose() << std::endl;
                //         // std::cout << dxds_all_nodes.block(left_idx_in_range * 3, site_idx * 2, 3, 2) << std::endl;
                //         // std::cout << "second term " << second_term.transpose() << std::endl;
                //         // std::getchar();
                //     }
                //     for (int site_idx : unique_ixn_points[right_idx_in_range].second)
                //     {
                //         TV2 second_term = weights_term * dis * dis * 0.5 * 
                //             dldxk.transpose() * dxds_all_nodes.block(right_idx_in_range * 3, site_idx * 2, 3, 2);
                //         addForceEntry<2>(grads[cell_idx], {site_idx}, second_term);
                //         // std::cout << "second term " << second_term.transpose() << std::endl;
                //     }
                // }
                edge_cnt ++;
            }
            // std::getchar();
        });
    
        for (int i = 0; i < voronoi_cell_data.size(); i++)
        {
            grad += -w * grads[i];
        }
        
    }
    else
    {
        std::vector<VectorXT> grads((int)unique_ixn_points.size(), 
        VectorXT::Zero(n_dof));
        
        StiffnessMatrix dxds_all_nodes;
        computeDxDsAllNodes(dxds_all_nodes);

    #ifdef PARALLEL
        tbb::parallel_for(0, (int)unique_ixn_points.size(),[&](int ixn_idx)
    #else
        for (int ixn_idx = 0; ixn_idx < unique_ixn_points.size(); ixn_idx++)
    #endif
        {
            auto ixn = unique_ixn_points[ixn_idx];
            SurfacePoint xi = ixn.first;
            std::vector<int> site_indices = ixn.second;
            if (site_indices.size() == 2)
    #ifdef PARALLEL
                return;
    #else
                continue;
    #endif
            // MatrixXT dx_ds;
            // computeDxDs(xi, site_indices, dx_ds, true);
            
            for (int i = 0; i < site_indices.size(); i++)
            {
                int idx = site_indices[i];
                SurfacePoint si = samples[idx];
                // Vector<T, 4> dldw;
                // T dis = computeGeodesicLengthAndGradient(xi, si, dldw);
                Vector<T, 5> dldw;
                T dis = computeGeodesicLengthAndGradient(xi, si, dldw);
                
                T dOdl = dis * cell_weights[idx];
                // TV2 pOpx = dldw.segment<2>(0);
                // TV2 pOps = dldw.segment<2>(2);

                TV pOpx = dldw.segment<3>(0);
                TV2 pOps = dldw.segment<2>(3);

                
                // TV2 dOds = dOdl * ((pOpx.transpose() * 
                //                 dx_ds.block(0, i * 2, 2, 2)).transpose()
                //             + pOps);

                TV2 dOds = dOdl * ((pOpx.transpose() * 
                                    dxds_all_nodes.block(ixn_idx * 3, idx * 2, 3, 2)
                                    ).transpose()
                                + pOps);
                
                addForceEntry<2>(grads[ixn_idx], {idx}, dOds);
                // x is function of all s
                for (int j = 0; j < site_indices.size(); j++)
                {
                    if (j==i) continue;
                    // dOds = dOdl * (pOpx.transpose() * dx_ds.block(0, j * 2, 2, 2)).transpose();
                    dOds = dOdl * (pOpx.transpose() * 
                                        dxds_all_nodes.block(ixn_idx * 3, site_indices[j] * 2, 3, 2)
                                        ).transpose();
                    addForceEntry<2>(grads[ixn_idx], {site_indices[j]}, dOds);
                }
            }
        }
    #ifdef PARALLEL
        );
    #endif
        for (int i = 0; i < unique_ixn_points.size(); i++)
        {
            grad += -w * grads[i];
        }
    }

}


void VoronoiCells::addCentroidalVDHessianEntries(std::vector<Entry>& entries, T w)
{
    if (edge_weighting)
    {
        std::vector<std::vector<Entry>> entry_thread(voronoi_cell_data.size(), std::vector<Entry>());
        StiffnessMatrix dxds_all_nodes;
        computeDxDsAllNodes(dxds_all_nodes);
        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% iterate over cells %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        iterateVoronoiCellsParallel([&](const VoronoiCellData& cell_data, int cell_idx)
        {
            int edge_cnt = 0;
            int n_voro_cell_edge = cell_data.cell_edge_lengths.size();
            VtxList vtx_list = voronoi_cell_vertices[cell_idx];
            // std::cout << "----- cell " << cell_idx << " -----" << std::endl;
            for (int idx : cell_data.cell_vtx_nodes)
            {
                auto ixn = unique_ixn_points[cell_data.cell_nodes[idx]];
                SurfacePoint xi = ixn.first;
                std::vector<int> site_indices = ixn.second;
                SurfacePoint si = samples[cell_idx];
                Vector<T, 5> dldw;
                Matrix<T, 5, 5> d2ldw2;
                T dis = computeGeodesicLengthAndGradientAndHessian(xi, si, dldw, d2ldw2);
                int left_idx = (edge_cnt - 1 + n_voro_cell_edge) % n_voro_cell_edge;
                int right_idx = edge_cnt;
                // std::cout << "left idx " << left_idx << " right idx " << right_idx << std::endl;
                // T left_seg_len = cell_data.cell_edge_lengths[left_idx];
                // T right_seg_len = cell_data.cell_edge_lengths[right_idx];

                int left_node_idx = cell_data.cell_nodes[cell_data.cell_vtx_nodes[left_idx]];
                int right_node_idx = cell_data.cell_nodes[cell_data.cell_vtx_nodes[(edge_cnt+1)%cell_data.cell_vtx_nodes.size()]];
                
                TV left_node = toTV(unique_ixn_points[left_node_idx].first.interpolate(geometry->vertexPositions));
                TV right_node = toTV(unique_ixn_points[right_node_idx].first.interpolate(geometry->vertexPositions));
                TV current_node = toTV(xi.interpolate(geometry->vertexPositions));

                T left_seg_len = (left_node - current_node).norm();
                T right_seg_len = (right_node - current_node).norm();

                T avg = 0.5 * (left_seg_len + right_seg_len);

                TV pOpx = dldw.segment<3>(0);
                TV2 pOps = dldw.segment<2>(3);

                VectorXT dOds_total(site_indices.size() * 2);
                dOds_total.setZero();

                // these are the partials when consider rho of x fixed
                for (int j = 0; j < site_indices.size(); j++)
                {
                    TV2 first_term = TV2::Zero();
                    if (site_indices[j] == cell_idx)
                        first_term = cell_weights[cell_idx] * (
                            (
                                pOpx.transpose() * 
                                dxds_all_nodes.block(cell_data.cell_nodes[idx] * 3, 
                                cell_idx * 2, 3, 2)
                            ).transpose()
                            + pOps);
                    else
                        first_term = cell_weights[cell_idx] * (pOpx.transpose() * 
                                    dxds_all_nodes.block(cell_data.cell_nodes[idx] * 3, site_indices[j] * 2, 3, 2)
                                    ).transpose();
                    addForceEntry<2>(dOds_total, {j}, first_term);
                }

                TM2 d2Ods2 = avg * dis * d2ldw2.block(3, 3, 2, 2);
                addHessianEntry<2, 2>(entry_thread[cell_idx], {cell_idx}, w * d2Ods2);


                MatrixXT d2Ods2_total(site_indices.size() * 2,
                    site_indices.size() * 2);
                d2Ods2_total.setZero();

                d2Ods2_total += avg * dOds_total * dOds_total.transpose();
                
                TM d2Odx2 = d2ldw2.block(0, 0, 3, 3);
                MatrixXT d2ldxds(3, site_indices.size() * 2); d2ldxds.setZero();
                for (int j = 0; j < site_indices.size(); j++)
                {
                    d2ldxds.block(0, j * 2, 3, 2) = dxds_all_nodes.block(
                                                cell_data.cell_nodes[idx]* 3, 
                                                site_indices[j] * 2, 3, 2);
                }
                d2Ods2_total += avg * dis * d2ldxds.transpose() * d2Odx2 * d2ldxds;
                addHessianEntry<2, 2>(entry_thread[cell_idx], site_indices, w * d2Ods2_total);

                Vector<T, 6> dldxleft; 
                dldxleft.segment<3>(0) = -(current_node - left_node).normalized();
                dldxleft.segment<3>(3) = (current_node - left_node).normalized();
                Matrix<T, 6, 6> d2ldleft2;
                edgeLengthHessian(left_node, current_node, d2ldleft2);
                int n_node_left = unique_ixn_points[left_node_idx].second.size();
                int n_node_right = site_indices.size();
                MatrixXT dxds(6, (n_node_left + n_node_right) * 2); dxds.setZero();
                std::vector<int> idx_list;
                int cnt = 0;
                for (int site_idx : unique_ixn_points[left_node_idx].second)
                {
                    dxds.block(0, cnt * 2, 3, 2) += dxds_all_nodes.block(left_node_idx * 3, site_idx * 2, 3, 2);
                    cnt++;
                    idx_list.push_back(site_idx);
                }
                for (int site_idx : site_indices)
                {
                    dxds.block(3, cnt * 2, 3, 2) += dxds_all_nodes.block(cell_data.cell_nodes[idx] * 3, site_idx * 2, 3, 2);
                    cnt++;
                    idx_list.push_back(site_idx);
                }

                

                MatrixXT d2Ods2_total_second = 
                    0.5 * dis * dis * dxds.transpose() * d2ldleft2 * dxds;
                // std::cout << d2Ods2_total_second << std::endl;
                addHessianEntry<2, 2>(entry_thread[cell_idx], idx_list, w * d2Ods2_total_second);

                // VectorXT dlds = weights_term * dis * dis * 0.5 * dldxleft.transpose() * dxds;
                // addForceEntry<2>(grads[cell_idx], idx_list, dlds);

                Vector<T, 6> dldxright; 
                dldxright.segment<3>(0) = -(right_node - current_node).normalized();
                dldxright.segment<3>(3) = (right_node - current_node).normalized();
                Matrix<T, 6, 6> d2ldright2;
                edgeLengthHessian(current_node, right_node, d2ldright2);
                n_node_left = site_indices.size();
                n_node_right = unique_ixn_points[right_node_idx].second.size();
                dxds.resize(6, (n_node_left + n_node_right) * 2); dxds.setZero();
                idx_list.resize(0);
                cnt = 0;
                for (int site_idx : site_indices)
                {
                    dxds.block(0, cnt * 2, 3, 2) += dxds_all_nodes.block(cell_data.cell_nodes[idx] * 3, site_idx * 2, 3, 2);
                    cnt++;
                    idx_list.push_back(site_idx);
                }
                for (int site_idx : unique_ixn_points[right_node_idx].second)
                {
                    dxds.block(3, cnt * 2, 3, 2) += dxds_all_nodes.block(right_node_idx * 3, site_idx * 2, 3, 2);
                    cnt++;
                    idx_list.push_back(site_idx);
                }
                d2Ods2_total_second = 
                    0.5 * dis * dis * dxds.transpose() * d2ldright2 * dxds;
                // std::cout << d2Ods2_total_second << std::endl;
                addHessianEntry<2, 2>(entry_thread[cell_idx], idx_list, w * d2Ods2_total_second);
                // dlds = weights_term * dis * dis * 0.5 * dldxright.transpose() * dxds;
                // addForceEntry<2>(grads[cell_idx], idx_list, dlds);

                MatrixXT partial = 2.0 * (dldxleft.segment<3>(3) + dldxright.segment<3>(0)).transpose() * dxds.block(
                    3, unique_ixn_points[left_node_idx].second.size()*3,
                    3, site_indices.size() * 3) * dis * dOds_total;
                // addHessianEntry<2, 2>(entry_thread[cell_idx], site_indices, w * partial);
                
                // iterate over voronoi edges in a single cell
                // compute the derivative of voronoi edge length w.r.t site positions
               
                // int left_node_location = cell_data.cell_vtx_nodes[left_idx];
                // int right_node_location = left_idx < right_idx ? 
                //     // last node
                //     cell_data.cell_vtx_nodes[(right_idx+1)%cell_data.cell_vtx_nodes.size()] 
                //     : cell_data.cell_vtx_nodes[1] + cell_data.cell_nodes.size();
                // if (right_node_location == 0)
                //     right_node_location = cell_data.cell_nodes.size();
                
                // for (int j = left_node_location; j < right_node_location; j++)
                // {
                //     int left_idx_in_range = cell_data.cell_nodes[j % cell_data.cell_nodes.size()];
                //     int right_idx_in_range = cell_data.cell_nodes[(j + 1) % cell_data.cell_nodes.size()];
                //     TV xj = toTV(unique_ixn_points[left_idx_in_range].first.interpolate(geometry->vertexPositions));
                //     TV xk = toTV(unique_ixn_points[right_idx_in_range].first.interpolate(geometry->vertexPositions));
                //     Vector<T, 6> dldxjxk;
                //     dldxjxk.segment<3>(0) = -(xk - xj).normalized();
                //     dldxjxk.segment<3>(3) = (xk - xj).normalized();
        
                //     Matrix<T, 6, 6> d2ldxjxk;
                //     edgeLengthHessian(xj, xk, d2ldxjxk);

                //     int n_node_left = unique_ixn_points[left_idx_in_range].second.size();
                //     int n_node_right = unique_ixn_points[right_idx_in_range].second.size();

                //     MatrixXT dxds(6, (n_node_left + n_node_right) * 2); dxds.setZero();
                //     std::vector<int> idx_list;
                //     int cnt = 0;
                //     for (int site_idx : unique_ixn_points[left_idx_in_range].second)
                //     {
                //         dxds.block(0, cnt * 2, 3, 2) += dxds_all_nodes.block(left_idx_in_range * 3, site_idx * 2, 3, 2);
                //         cnt++;
                //         idx_list.push_back(site_idx);
                //         // TV2 second_term = 0.5 * dis * dis * 0.5 * 
                //         //     dldxj.transpose() * dxds_all_nodes.block(left_idx_in_range * 3, site_idx * 2, 3, 2);
                //         // addForceEntry<2>(grads[cell_idx], {site_idx}, second_term);
                //     }
                //     for (int site_idx : unique_ixn_points[right_idx_in_range].second)
                //     {
                //         dxds.block(3, cnt * 2, 3, 2) += dxds_all_nodes.block(right_idx_in_range * 3, site_idx * 2, 3, 2);
                //         cnt++;
                //         idx_list.push_back(site_idx);
                //         // TV2 second_term = 0.5 * dis * dis * 0.5 * 
                //             // dldxk.transpose() * dxds_all_nodes.block(right_idx_in_range * 3, site_idx * 2, 3, 2);
                //         // addForceEntry<2>(grads[cell_idx], {site_idx}, second_term);
                //     }
                //     MatrixXT d2Ods2_total_second = 
                //         0.5 * dis * dis * dxds.transpose() * d2ldxjxk * dxds;
                //     // std::cout << d2Ods2_total_second << std::endl;
                //     addHessianEntry<2, 2>(entry_thread[cell_idx], idx_list, w * d2Ods2_total_second);
                // }
                
                edge_cnt ++;
            }
            
        });
        for (int i = 0; i < voronoi_cell_data.size(); i++)
        {
            entries.insert(entries.end(), entry_thread[i].begin(), entry_thread[i].end());   
        }
    }
    else
    {
        std::vector<std::vector<Entry>> entry_thread(unique_ixn_points.size(), std::vector<Entry>());
        #ifdef PARALLEL
            tbb::parallel_for(0, (int)unique_ixn_points.size(),[&](int ixn_idx)
        #else
            for (int ixn_idx = 0; ixn_idx < unique_ixn_points.size(); ixn_idx++)
        #endif
            {
                auto ixn = unique_ixn_points[ixn_idx];
                SurfacePoint xi = ixn.first;
                std::vector<int> site_indices = ixn.second;
                if (site_indices.size() == 2)
        #ifdef PARALLEL
                    return;
        #else
                    continue;
        #endif
                MatrixXT dx_ds;
                computeDxDs(xi, site_indices, dx_ds, true);
                VectorXT dOds_total(site_indices.size() * 2);
                dOds_total.setZero();

                MatrixXT d2Ods2_total(site_indices.size() * 2,
                    site_indices.size() * 2);
                d2Ods2_total.setZero(); 

                TM2 d2Odx2 = TM2::Zero();

                MatrixXT d2Ods2 = d2Ods2_total;

                MatrixXT d2Odxds(2, site_indices.size() * 2);
                d2Odxds.setZero();

                for (int i = 0; i < site_indices.size(); i++)
                {
                    int idx = site_indices[i];
                    SurfacePoint si = samples[idx];
                    Vector<T, 4> dldw;
                    Matrix<T, 4, 4> d2ldw2;
                    T dis = computeGeodesicLengthAndGradientAndHessian(xi, si, dldw, d2ldw2);
                    
                    T dOdl = dis * cell_weights[idx];
                    TV2 pOpx = dldw.segment<2>(0);
                    TV2 pOps = dldw.segment<2>(2);

                    TV2 dOds = dOdl * ((pOpx.transpose() * 
                                    dx_ds.block(0, i * 2, 2, 2)).transpose()
                                + pOps);
                    
                    addForceEntry<2>(dOds_total, {i}, dOds);

                    d2Odx2 += dis * d2ldw2.block(2, 2, 2, 2);
                    d2Ods2.block(i * 2, i * 2, 2, 2) += dis * d2ldw2.block(0, 0, 2, 2) * cell_weights[idx];
                    d2Odxds.block(0, i * 2, 2, 2) += dis * d2ldw2.block(0, 2, 2, 2) * cell_weights[idx];
                    
                    // x is function of all s
                    for (int j = 0; j < site_indices.size(); j++)
                    {
                        if (j==i) continue;
                        dOds = dOdl * (pOpx.transpose() * dx_ds.block(0, j * 2, 2, 2)).transpose();
                        addForceEntry<2>(dOds_total, {j}, dOds);
                    }
                }
                

                d2Ods2_total += dOds_total * dOds_total.transpose();
                d2Ods2_total += dx_ds.transpose() * d2Odx2 * dx_ds;
                d2Ods2_total += d2Ods2;
                
                addHessianEntry<2, 2>(entry_thread[ixn_idx], site_indices, w * d2Ods2_total);
            }
        #ifdef PARALLEL
            );
        #endif
            for (int i = 0; i < unique_ixn_points.size(); i++)
            {
                entries.insert(entries.end(), entry_thread[i].begin(), entry_thread[i].end());
                
            }
    }
    
}

T VoronoiCells::computeCentroidalVDHessian(StiffnessMatrix& hess, VectorXT& grad, T& energy, T w)
{
    grad.resize(n_sites * 2); grad.setZero();

    energy = 0.0;
    std::vector<Entry> entries;
    for (int ixn_idx = 0; ixn_idx < unique_ixn_points.size(); ixn_idx++)
    {
        auto ixn = unique_ixn_points[ixn_idx];
        SurfacePoint xi = ixn.first;
        std::vector<int> site_indices = ixn.second;
        if (site_indices.size() == 2)
            continue;
        MatrixXT dx_ds;
        computeDxDs(xi, site_indices, dx_ds, true);
        VectorXT dOds_total(site_indices.size() * 2);
        dOds_total.setZero();

        MatrixXT d2Ods2_total(site_indices.size() * 2,
            site_indices.size() * 2);
        d2Ods2_total.setZero();

        TM2 d2Odx2 = TM2::Zero();

        MatrixXT d2Ods2 = d2Ods2_total;

        MatrixXT d2Odxds(2, site_indices.size() * 2);
        d2Odxds.setZero();

        for (int i = 0; i < site_indices.size(); i++)
        {
            int idx = site_indices[i];
            SurfacePoint si = samples[idx];
            Vector<T, 4> dldw;
            Matrix<T, 4, 4> d2ldw2;
            T dis = computeGeodesicLengthAndGradientAndHessian(xi, si, dldw, d2ldw2);
            energy += 0.5 * dis * dis;
            T dOdl = dis;
            TV2 pOpx = dldw.segment<2>(0);
            TV2 pOps = dldw.segment<2>(2);

            TV2 dOds = dOdl * ((pOpx.transpose() * 
                            dx_ds.block(0, i * 2, 2, 2)).transpose()
                        + pOps);
            
            addForceEntry<2>(dOds_total, {i}, dOds);


            d2Odx2 += dis * d2ldw2.block(2, 2, 2, 2);
            d2Ods2.block(i * 2, i * 2, 2, 2) += dis * d2ldw2.block(0, 0, 2, 2);
            d2Odxds.block(0, i * 2, 2, 2) += dis * d2ldw2.block(0, 2, 2, 2);
            
            // x is function of all s
            for (int j = 0; j < site_indices.size(); j++)
            {
                if (j==i) continue;
                dOds = dOdl * (pOpx.transpose() * dx_ds.block(0, j * 2, 2, 2)).transpose();
                addForceEntry<2>(dOds_total, {j}, dOds);
            }
        }
        addForceEntry<2>(grad, site_indices, dOds_total);

        d2Ods2_total += dOds_total * dOds_total.transpose();
        d2Ods2_total += dx_ds.transpose() * d2Odx2 * dx_ds;
        d2Ods2_total += d2Ods2;
        // d2Ods2_total += d2Odxds.transpose() * dx_ds;
        // d2Ods2_total += dx_ds.transpose() * d2Odxds;

        addHessianEntry<2, 2>(entries, site_indices, d2Ods2_total);
    }
    hess.resize(n_sites * 2, n_sites * 2);
    hess.setFromTriplets(entries.begin(), entries.end());
    projectDirichletDoFMatrix(hess, dirichlet_data);
    hess.makeCompressed();
    return grad.norm();
}

T VoronoiCells::computePerimeterMinimizationEnergy(T w)
{
    T energy = 0.0;
    int n_edges = valid_VD_edges.size();
    for (int i = 0; i < n_edges; i++)
    {   
        int idx0 = valid_VD_edges[i][0];
        int idx1 = valid_VD_edges[i][1];
        TV v0 = toTV(unique_ixn_points[idx0].first.interpolate(geometry->vertexPositions));
        TV v1 = toTV(unique_ixn_points[idx1].first.interpolate(geometry->vertexPositions));
        energy += 0.5 * w * (v1 - v0).dot(v1 - v0);
    }
    return energy;
}

T VoronoiCells::computePerimeterMinimizationGradient(VectorXT& grad, T& energy, T w)
{
    grad.resize(n_sites * 2); grad.setZero();

    int n_edges = valid_VD_edges.size();
    VectorXT energies(n_edges);
    energies.setZero();
    std::vector<VectorXT> grads(n_edges, 
        VectorXT::Zero(n_sites * 2));


#ifdef PARALLEL
    tbb::parallel_for(0, n_edges,[&](int i)
#else
    for (int i = 0; i < n_edges; i++)
#endif
    {   
        int idx0 = valid_VD_edges[i][0];
        int idx1 = valid_VD_edges[i][1];
        SurfacePoint x0 = unique_ixn_points[idx0].first.inSomeFace();
        SurfacePoint x1 = unique_ixn_points[idx1].first.inSomeFace();
        
        TV v0 = toTV(x0.interpolate(geometry->vertexPositions));
        TV v1 = toTV(x1.interpolate(geometry->vertexPositions));
        energies[i] += 0.5 * (v1 - v0).dot(v1 - v0);
        // energy += (v1 - v0).norm();
        
        std::vector<int> x0_sites = unique_ixn_points[idx0].second;
        std::vector<int> x1_sites = unique_ixn_points[idx1].second;

        TV dOdx0, dOdx1;
        dOdx0 = -(v1 - v0);
        dOdx1 = (v1 - v0);
        // dOdx0 = -(v1 - v0).normalized();
        // dOdx1 = (v1 - v0).normalized();
        MatrixXT dx0_ds, dx1_ds;

        bool valid = computeDxDs(x0, x0_sites, dx0_ds);
        valid &= computeDxDs(x1, x1_sites, dx1_ds);

        // if (!valid)
        //     return 1e12;

        VectorXT dOds_site0 = dOdx0.transpose() * dx0_ds;
        VectorXT dOds_site1 = dOdx1.transpose() * dx1_ds;

        addForceEntry<2>(grads[i], x0_sites, dOds_site0);
        addForceEntry<2>(grads[i], x1_sites, dOds_site1);
    }
#ifdef PARALLEL
    );
#endif
    for (int i = 0; i < n_edges; i++)
    {
        grad += w * grads[i];
    }
    energy = w * energies.sum();
    return grad.norm();
}

void VoronoiCells::addPerimeterMinimizationForceEntries(VectorXT& grad, T w)
{
    int n_dof = samples.size() * 2;
    int n_edges = valid_VD_edges.size();
    
    std::vector<VectorXT> grads(n_edges, 
        VectorXT::Zero(n_dof));


#ifdef PARALLEL
    tbb::parallel_for(0, n_edges,[&](int i)
#else
    for (int i = 0; i < n_edges; i++)
#endif
    {   
        int idx0 = valid_VD_edges[i][0];
        int idx1 = valid_VD_edges[i][1];
        SurfacePoint x0 = unique_ixn_points[idx0].first.inSomeFace();
        SurfacePoint x1 = unique_ixn_points[idx1].first.inSomeFace();
        
        TV v0 = toTV(x0.interpolate(geometry->vertexPositions));
        TV v1 = toTV(x1.interpolate(geometry->vertexPositions));
        
        std::vector<int> x0_sites = unique_ixn_points[idx0].second;
        std::vector<int> x1_sites = unique_ixn_points[idx1].second;

        TV dOdx0, dOdx1;
        dOdx0 = -(v1 - v0);
        dOdx1 = (v1 - v0);
        // dOdx0 = -(v1 - v0).normalized();
        // dOdx1 = (v1 - v0).normalized();
        MatrixXT dx0_ds, dx1_ds;

        bool valid = computeDxDs(x0, x0_sites, dx0_ds);
        valid &= computeDxDs(x1, x1_sites, dx1_ds);

        // if (!valid)
        //     return 1e12;

        VectorXT dOds_site0 = dOdx0.transpose() * dx0_ds;
        VectorXT dOds_site1 = dOdx1.transpose() * dx1_ds;

        addForceEntry<2>(grads[i], x0_sites, dOds_site0);
        addForceEntry<2>(grads[i], x1_sites, dOds_site1);
    }
#ifdef PARALLEL
    );
#endif
    for (int i = 0; i < n_edges; i++)
    {
        grad += -w * grads[i];
    }
}
void VoronoiCells::addPerimeterMinimizationHessianEntries(std::vector<Entry>& entries, T w)
{
    int n_edges = valid_VD_edges.size();
    for (int i = 0; i < n_edges; i++)
    {   
        int idx0 = valid_VD_edges[i][0];
        int idx1 = valid_VD_edges[i][1];
        SurfacePoint x0 = unique_ixn_points[idx0].first.inSomeFace();
        SurfacePoint x1 = unique_ixn_points[idx1].first.inSomeFace();
        
        TV v0 = toTV(x0.interpolate(geometry->vertexPositions));
        TV v1 = toTV(x1.interpolate(geometry->vertexPositions));
        

        // energy += (v1 - v0).norm();
        std::vector<int> x0_sites = unique_ixn_points[idx0].second;
        std::vector<int> x1_sites = unique_ixn_points[idx1].second;

        TV dOdx0, dOdx1;
        dOdx0 = -(v1 - v0);
        dOdx1 = (v1 - v0);

        // dOdx0 = -(v1 - v0).normalized();
        // dOdx1 = (v1 - v0).normalized();
        MatrixXT dx0_ds, dx1_ds;

        bool valid = computeDxDs(x0, x0_sites, dx0_ds);
        valid &= computeDxDs(x1, x1_sites, dx1_ds);


        VectorXT dOds_site0 = dOdx0.transpose() * dx0_ds;
        VectorXT dOds_site1 = dOdx1.transpose() * dx1_ds;
        
        
        TM d2Odx2 = (TM::Identity() - (v1 - v0).normalized() * (v1 - v0).normalized().transpose()) / (v1 - v0).norm();


        MatrixXT d2x0ds2 = dx0_ds.transpose() * (d2Odx2 + TM::Identity()) * dx0_ds;
        MatrixXT d2x1ds2 = dx1_ds.transpose() * (d2Odx2 + TM::Identity()) * dx1_ds;

        

        addHessianEntry<2, 2>(entries, x0_sites, w * d2x0ds2);
        addHessianEntry<2, 2>(entries, x1_sites, w * d2x1ds2);

    }
}

T VoronoiCells::computePerimeterMinimizationHessian(StiffnessMatrix& hess, 
    VectorXT& grad, T& energy, T w)
{
    grad.resize(n_sites * 2); grad.setZero();

    energy = 0.0;
    std::vector<Entry> entries;
    int n_edges = valid_VD_edges.size();
    for (int i = 0; i < n_edges; i++)
    {   
        int idx0 = valid_VD_edges[i][0];
        int idx1 = valid_VD_edges[i][1];
        SurfacePoint x0 = unique_ixn_points[idx0].first.inSomeFace();
        SurfacePoint x1 = unique_ixn_points[idx1].first.inSomeFace();
        
        TV v0 = toTV(x0.interpolate(geometry->vertexPositions));
        TV v1 = toTV(x1.interpolate(geometry->vertexPositions));
        energy += 0.5 * (v1 - v0).dot(v1 - v0);

        // energy += (v1 - v0).norm();
        std::vector<int> x0_sites = unique_ixn_points[idx0].second;
        std::vector<int> x1_sites = unique_ixn_points[idx1].second;

        TV dOdx0, dOdx1;
        dOdx0 = -(v1 - v0);
        dOdx1 = (v1 - v0);

        // dOdx0 = -(v1 - v0).normalized();
        // dOdx1 = (v1 - v0).normalized();
        MatrixXT dx0_ds, dx1_ds;

        bool valid = computeDxDs(x0, x0_sites, dx0_ds);
        valid &= computeDxDs(x1, x1_sites, dx1_ds);

        if (!valid)
            return 1e12;


        VectorXT dOds_site0 = dOdx0.transpose() * dx0_ds;
        VectorXT dOds_site1 = dOdx1.transpose() * dx1_ds;
        addForceEntry<2>(grad, x0_sites, dOds_site0);
        addForceEntry<2>(grad, x1_sites, dOds_site1);

        
        // TM d2Odx2 = (TM::Identity() - (v1 - v0).normalized() * (v1 - v0).normalized().transpose()) / (v1 - v0).norm();
        MatrixXT d2x0ds2 = dx0_ds.transpose() * dx0_ds;
        MatrixXT d2x1ds2 = dx1_ds.transpose() * dx1_ds;

        MatrixXT d2Odxds = -dx1_ds.transpose() * dx0_ds;

        addHessianEntry<2, 2>(entries, x0_sites, d2x0ds2);
        addHessianEntry<2, 2>(entries, x1_sites, d2x1ds2);

    }

    hess.resize(n_sites * 2, n_sites * 2);
    hess.setFromTriplets(entries.begin(), entries.end());
    hess *= w;
    projectDirichletDoFMatrix(hess, dirichlet_data);
    hess.makeCompressed();
    return grad.norm();
}

T VoronoiCells::computeDistanceMatchingEnergy(const std::vector<int>& site_indices, 
    SurfacePoint& xi_current)
{
    T energy = 0.0;
    TV current = toTV(xi_current.interpolate(geometry->vertexPositions));
    TV site0_location = toTV(samples[site_indices[0]].interpolate(geometry->vertexPositions));
    T dis_to_site0;
    std::vector<IxnData> ixn_data_site;
    std::vector<SurfacePoint> path; 
    if (metric == Euclidean)
        dis_to_site0 = (current - site0_location).norm();
    else
        computeGeodesicDistance(samples[site_indices[0]], xi_current, 
            dis_to_site0, path, ixn_data_site, false);
    
    for (int j = 1; j < site_indices.size(); j++)
    {
        int site_idx = site_indices[j];
        
        TV site_location = toTV(samples[site_indices[j]].interpolate(geometry->vertexPositions));
        T dis_to_site;
        if (metric == Euclidean)
            dis_to_site = (current - site_location).norm();
        else
            computeGeodesicDistance(samples[site_idx], xi_current, dis_to_site,
                path, ixn_data_site, false);
        energy += 0.5 * std::pow(dis_to_site - dis_to_site0, 2);
    }
    return energy;
}

T VoronoiCells::computeDistanceMatchingGradient(const std::vector<int>& site_indices, 
    SurfacePoint& xi_current, TV2& grad, T& energy)
{
    grad = TV2::Zero();
    energy = 0.0;
    TV current = toTV(xi_current.interpolate(geometry->vertexPositions));
    TV site0_location = toTV(samples[site_indices[0]].interpolate(geometry->vertexPositions));
    TV v0 = toTV(geometry->vertexPositions[xi_current.face.halfedge().vertex()]);
    TV v1 = toTV(geometry->vertexPositions[xi_current.face.halfedge().next().vertex()]);
    TV v2 = toTV(geometry->vertexPositions[xi_current.face.halfedge().next().next().vertex()]);

    // std::cout << "face# " << fi.getIndex() << " xi " << current.transpose() << std::endl;
    TV bary = toTV(xi_current.faceCoords);
    TV pt = v0 * bary[0] + v1 * bary[1] + v2 * bary[2];
    // std::cout << pt.transpose() << " " << current.transpose() << std::endl;
    // std::getchar();

    // std::cout << v0.transpose() << " " << v1.transpose() << " " << v2.transpose() << std::endl;
    T dis_to_site0;
    Matrix<T, 3, 2> dxdw; 
    dxdw.col(0) = v0 - v2;
    dxdw.col(1) = v1 - v2;
    
    TV dldx0;
    if (metric == Euclidean)
    {
        dis_to_site0 = (current - site0_location).norm();
        dldx0 = (current - site0_location).normalized();
    }
    else
    {
        std::vector<IxnData> ixn_data_site;
        std::vector<SurfacePoint> path;
        computeGeodesicDistance(samples[site_indices[0]], xi_current, dis_to_site0, path, 
            ixn_data_site, true);
        int length = path.size();
        
        TV vtx0 = toTV(path[0].interpolate(geometry->vertexPositions));
        TV vtx1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));
        if (length == 2)
        {
            dldx0 = (vtx1 - vtx0).normalized();
        }
        else
        {
            TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
            dldx0 = -(ixn1 - vtx1).normalized();
        }
    }
    
    for (int j = 1; j < site_indices.size(); j++)
    {
        int site_idx = site_indices[j];
        TV site_location = toTV(samples[site_indices[j]].interpolate(geometry->vertexPositions));
        T dis_to_site;
        TV dldxj;
        if (metric == Euclidean)
        {
            dis_to_site = (current - site_location).norm();
            dldxj = (current - site_location).normalized();
        }
        else
        {
            std::vector<IxnData> ixn_data_site;
            std::vector<SurfacePoint> path; 

            computeGeodesicDistance(samples[site_idx], xi_current, dis_to_site,
                path, ixn_data_site, true);

            int length = path.size();
        
            TV vtx0 = toTV(path[0].interpolate(geometry->vertexPositions));
            TV vtx1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));
            if (length == 2)
            {
                dldxj = (vtx1 - vtx0).normalized();
            }
            else
            {
                TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
                dldxj = -(ixn1 - vtx1).normalized();
            }
        }
        
        // std::cout << site0_location.transpose() << " " << site_location.transpose() << std::endl;
        // std::cout << "dis_to_site0 " << dis_to_site0 << " dis_to_site " << dis_to_site << std::endl;
        energy += 0.5 * std::pow(dis_to_site - dis_to_site0, 2); 

        T dOdl = (dis_to_site - dis_to_site0); 
        
        grad += dOdl * -dxdw.transpose() * dldx0;
        grad += dOdl * dxdw.transpose() * dldxj;
    }
    return grad.norm();
}

void VoronoiCells::computeDistanceMatchingHessian(const std::vector<int>& site_indices, 
        SurfacePoint& xi_current, TM2& hess)
{
    hess = TM2::Zero();
    
    TV current = toTV(xi_current.interpolate(geometry->vertexPositions));
    TV site0_location = toTV(samples[site_indices[0]].interpolate(geometry->vertexPositions));
    TV v0 = toTV(geometry->vertexPositions[xi_current.face.halfedge().vertex()]);
    TV v1 = toTV(geometry->vertexPositions[xi_current.face.halfedge().next().vertex()]);
    TV v2 = toTV(geometry->vertexPositions[xi_current.face.halfedge().next().next().vertex()]);

    T dis_to_site0;
    Matrix<T, 3, 2> dxdw; 
    dxdw.col(0) = v0 - v2;
    dxdw.col(1) = v1 - v2;
    
    TV dldx0;
    TM d2ldx02;
    if (metric == Euclidean)
    {
        dis_to_site0 = (current - site0_location).norm();
        dldx0 = (current - site0_location).normalized();
        d2ldx02 = (TM::Identity() - dldx0 * dldx0.transpose()) / dis_to_site0;
    }
    else
    {
        std::vector<IxnData> ixn_data_site;
        std::vector<SurfacePoint> path;
        computeGeodesicDistance(samples[site_indices[0]], xi_current, dis_to_site0, path, 
            ixn_data_site, true);
        int length = path.size();
        
        TV vtx0 = toTV(path[0].interpolate(geometry->vertexPositions));
        TV vtx1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));
        if (length == 2)
        {
            dldx0 = (vtx1 - vtx0).normalized();
            d2ldx02 = (TM::Identity() - dldx0 * dldx0.transpose()) / dis_to_site0;
        }
        else
        {
            TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
            dldx0 = -(ixn1 - vtx1).normalized();
        }
    }
    
    for (int j = 1; j < site_indices.size(); j++)
    {
        int site_idx = site_indices[j];
        TV site_location = toTV(samples[site_indices[j]].interpolate(geometry->vertexPositions));
        T dis_to_site;
        TV dldxj;
        TM d2ldxj2;
        if (metric == Euclidean)
        {
            dis_to_site = (current - site_location).norm();
            dldxj = (current - site_location).normalized();
            d2ldxj2 = (TM::Identity() - dldxj * dldxj.transpose()) / dis_to_site;
        }
        else
        {
            std::vector<IxnData> ixn_data_site;
            std::vector<SurfacePoint> path; 

            computeGeodesicDistance(samples[site_idx], xi_current, dis_to_site,
                path, ixn_data_site, true);

            int length = path.size();
        
            TV vtx0 = toTV(path[0].interpolate(geometry->vertexPositions));
            TV vtx1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));
            if (length == 2)
            {
                dldxj = (vtx1 - vtx0).normalized();
            }
            else
            {
                TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
                dldxj = -(ixn1 - vtx1).normalized();

                int ixn_dof = (length - 2) * 3;
                MatrixXT dfdc(ixn_dof, 3); dfdc.setZero();
                MatrixXT dfdx(ixn_dof, ixn_dof); dfdx.setZero();
                MatrixXT dxdt(ixn_dof, length-2); dxdt.setZero();
                MatrixXT d2gdcdx(ixn_dof, 3); d2gdcdx.setZero();

                for (int ixn_id = 0; ixn_id < length - 3; ixn_id++)
                {
                    // std::cout << "inside" << std::endl;
                    Matrix<T, 6, 6> hess;
                    TV ixn_i = toTV(path[1 + ixn_id].interpolate(geometry->vertexPositions));
                    TV ixn_j = toTV(path[1 + ixn_id+1].interpolate(geometry->vertexPositions));

                    edgeLengthHessian(ixn_i, ixn_j, hess);
                    dfdx.block(ixn_id*3, ixn_id * 3, 6, 6) += hess;
                }
                for (int ixn_id = 0; ixn_id < length - 2; ixn_id++)
                {
                    TV x_start = ixn_data_site[1+ixn_id].start;
                    TV x_end = ixn_data_site[1+ixn_id].end;
                    dxdt.block(ixn_id * 3, ixn_id, 3, 1) = (x_end - x_start);
                }
                TM dlndxn = (TM::Identity() - dldxj * dldxj.transpose()) / (ixn1 - vtx1).norm();
                dfdx.block(ixn_dof-3, ixn_dof-3, 3, 3) += dlndxn;
                dfdc.block(ixn_dof-3, 0, 3, 3) += -dlndxn;
                d2gdcdx.block(ixn_dof-3, 0, 3, 3) += -dlndxn;

                MatrixXT dxdtd2gdxdc = dxdt.transpose() * d2gdcdx;
                MatrixXT dxdtd2gdx2dxdt = dxdt.transpose() * dfdx * dxdt;
                MatrixXT dtdc = dxdtd2gdx2dxdt.colPivHouseholderQr().solve(-dxdtd2gdxdc);


                TM d2gdc2 = dlndxn;

                d2ldxj2 = d2gdc2 + d2gdcdx.transpose() * dxdt * dtdc;
            }
        }
        
        // std::cout << site0_location.transpose() << " " << site_location.transpose() << std::endl;
        // std::cout << "dis_to_site0 " << dis_to_site0 << " dis_to_site " << dis_to_site << std::endl;

        T dOdl = (dis_to_site - dis_to_site0); 
        TV2 dldw = dxdw.transpose() * (dldxj - dldx0);

        // grad += dOdl * dldw
        // hess += dOdl * d2ldw2 + dldw^T d2Odl2 * dldw 
        hess += dldw * dldw.transpose();
    }
    
}

T VoronoiCells::computeDistanceMatchingEnergyGradientHessian(const std::vector<int>& site_indices, 
        SurfacePoint& xi_current, TM2& hess, TV2& grad, T& energy)
{
    hess = TM2::Zero();
    grad = TV2::Zero();
    energy = 0.0;

    TV current = toTV(xi_current.interpolate(geometry->vertexPositions));
    TV site0_location = toTV(samples[site_indices[0]].interpolate(geometry->vertexPositions));
    TV v0 = toTV(geometry->vertexPositions[xi_current.face.halfedge().vertex()]);
    TV v1 = toTV(geometry->vertexPositions[xi_current.face.halfedge().next().vertex()]);
    TV v2 = toTV(geometry->vertexPositions[xi_current.face.halfedge().next().next().vertex()]);

    T dis_to_site0;
    Matrix<T, 3, 2> dxdw; 
    dxdw.col(0) = v0 - v2;
    dxdw.col(1) = v1 - v2;
    
    TV dldx0;
    TM d2ldx02;
    if (metric == Euclidean)
    {
        dis_to_site0 = (current - site0_location).norm();
        dldx0 = (current - site0_location).normalized();
        d2ldx02 = (TM::Identity() - dldx0 * dldx0.transpose()) / dis_to_site0;
    }
    else
    {
        std::vector<IxnData> ixn_data_site;
        std::vector<SurfacePoint> path;
        computeGeodesicDistance(samples[site_indices[0]], xi_current, dis_to_site0, path, 
            ixn_data_site, true);
        int length = path.size();
        
        TV vtx0 = toTV(path[0].interpolate(geometry->vertexPositions));
        TV vtx1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));
        if (length == 2)
        {
            dldx0 = (vtx1 - vtx0).normalized();
            d2ldx02 = (TM::Identity() - dldx0 * dldx0.transpose()) / dis_to_site0;
        }
        else
        {
            TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
            dldx0 = -(ixn1 - vtx1).normalized();
        }
    }
    
    for (int j = 1; j < site_indices.size(); j++)
    {
        int site_idx = site_indices[j];
        TV site_location = toTV(samples[site_indices[j]].interpolate(geometry->vertexPositions));
        T dis_to_site;
        TV dldxj;
        TM d2ldxj2;
        if (metric == Euclidean)
        {
            dis_to_site = (current - site_location).norm();
            dldxj = (current - site_location).normalized();
            d2ldxj2 = (TM::Identity() - dldxj * dldxj.transpose()) / dis_to_site;
        }
        else
        {
            std::vector<IxnData> ixn_data_site;
            std::vector<SurfacePoint> path; 

            computeGeodesicDistance(samples[site_idx], xi_current, dis_to_site,
                path, ixn_data_site, true);

            int length = path.size();
        
            TV vtx0 = toTV(path[0].interpolate(geometry->vertexPositions));
            TV vtx1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));
            if (length == 2)
            {
                dldxj = (vtx1 - vtx0).normalized();
            }
            else
            {
                TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
                dldxj = -(ixn1 - vtx1).normalized();

                int ixn_dof = (length - 2) * 3;
                MatrixXT dfdc(ixn_dof, 3); dfdc.setZero();
                MatrixXT dfdx(ixn_dof, ixn_dof); dfdx.setZero();
                MatrixXT dxdt(ixn_dof, length-2); dxdt.setZero();
                MatrixXT d2gdcdx(ixn_dof, 3); d2gdcdx.setZero();

                for (int ixn_id = 0; ixn_id < length - 3; ixn_id++)
                {
                    // std::cout << "inside" << std::endl;
                    Matrix<T, 6, 6> hess;
                    TV ixn_i = toTV(path[1 + ixn_id].interpolate(geometry->vertexPositions));
                    TV ixn_j = toTV(path[1 + ixn_id+1].interpolate(geometry->vertexPositions));

                    edgeLengthHessian(ixn_i, ixn_j, hess);
                    dfdx.block(ixn_id*3, ixn_id * 3, 6, 6) += hess;
                }
                for (int ixn_id = 0; ixn_id < length - 2; ixn_id++)
                {
                    TV x_start = ixn_data_site[1+ixn_id].start;
                    TV x_end = ixn_data_site[1+ixn_id].end;
                    dxdt.block(ixn_id * 3, ixn_id, 3, 1) = (x_end - x_start);
                }
                TM dlndxn = (TM::Identity() - dldxj * dldxj.transpose()) / (ixn1 - vtx1).norm();
                dfdx.block(ixn_dof-3, ixn_dof-3, 3, 3) += dlndxn;
                dfdc.block(ixn_dof-3, 0, 3, 3) += -dlndxn;
                d2gdcdx.block(ixn_dof-3, 0, 3, 3) += -dlndxn;

                MatrixXT dxdtd2gdxdc = dxdt.transpose() * d2gdcdx;
                MatrixXT dxdtd2gdx2dxdt = dxdt.transpose() * dfdx * dxdt;
                MatrixXT dtdc = dxdtd2gdx2dxdt.colPivHouseholderQr().solve(-dxdtd2gdxdc);


                TM d2gdc2 = dlndxn;

                d2ldxj2 = d2gdc2 + d2gdcdx.transpose() * dxdt * dtdc;
            }
        }
        
        // std::cout << site0_location.transpose() << " " << site_location.transpose() << std::endl;
        // std::cout << "dis_to_site0 " << dis_to_site0 << " dis_to_site " << dis_to_site << std::endl;

        T dOdl = (dis_to_site - dis_to_site0); 
        TV2 dldw = dxdw.transpose() * (dldxj - dldx0);

        energy += 0.5 * std::pow(dis_to_site - dis_to_site0, 2); 
        
        grad += dOdl * -dxdw.transpose() * dldx0;
        grad += dOdl * dxdw.transpose() * dldxj;
        hess += dldw * dldw.transpose();
    }
    return grad.norm();
}