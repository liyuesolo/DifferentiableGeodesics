#include "../include/IntrinsicSimulation.h"
#include "../autodiff/Elasticity.h"


T IntrinsicSimulation::layout2D(const TV& l0, TM& A_inv)
{
    T s = l0.sum() / 2.0;
    T Ai = std::sqrt(s * (s - l0[0]) * (s - l0[1]) * (s - l0[2]));
    T theta = std::acos((l0[0] * l0[0] - l0[1] * l0[1] + l0[2] * l0[2])/(2.0 * l0[2] * l0[0]));
    TV2 ei(l0[0], 0);
    TV2 ej(l0[2] * std::cos(theta), l0[2] * std::sin(theta));
    // T qyk = 2.0 * Ai / l0[0];
    // T qxk = std::sqrt(l0[2]*l0[2] - qyk * qyk);
    // // if (theta > 0.5 * M_PI)
    // {   
    //     // if (ej[0]-(-qxk) > 1e-8)
    //     //     std::cout <<" dd0" << std::endl;
    //     // ej[0] = -qxk;
    // }

    TV2 ek = ej - ei;
    Matrix<T, 3, 3> A; 
    A(0, 0) = ei[0] * ei[0]; A(0, 1) = 2.0 * ei[0] * ei[1]; A(0, 2) = ei[1] * ei[1]; 
    A(1, 0) = ek[0] * ek[0]; A(1, 1) = 2.0 * ek[0] * ek[1]; A(1, 2) = ek[1] * ek[1]; 
    A(2, 0) = ej[0] * ej[0]; A(2, 1) = 2.0 * ej[0] * ej[1]; A(2, 2) = ej[1] * ej[1]; 
    A_inv = A.inverse();

    return Ai;
}

void IntrinsicSimulation::computeGeodesicTriangleRestShape()
{
    X_inv_geodesic.resize(triangles.size());
    iterateTriangleSerial([&](Triangle tri, int tri_idx){
        Edge e0, e1, e2;
		getTriangleEdges(tri, e0, e1, e2);
        // here we use current length because it's undeformed
        T l0 = undeformed_length[edge_map[e0]];
        T l1 = undeformed_length[edge_map[e1]];
        T l2 = undeformed_length[edge_map[e2]];

        T theta = std::acos((l0 * l0 - l1 * l1 + l2 * l2)/(2.0 * l2 * l0));
        TV2 ei(l0, 0);
        TV2 ej(l2 * std::cos(theta), l2 * std::sin(theta));
        TM2 delta_rest; delta_rest.col(0) = ei; delta_rest.col(1) = ej;
        X_inv_geodesic[tri_idx] = delta_rest.inverse();
        
    });
}

void IntrinsicSimulation::addGeodesicNHEnergy(T& energy)
{
    iterateTriangleSerial([&](Triangle tri, int tri_idx){
        Edge e0, e1, e2;
		getTriangleEdges(tri, e0, e1, e2);
        TV l0; l0 << undeformed_length[edge_map[e0]], undeformed_length[edge_map[e1]], undeformed_length[edge_map[e2]];
        TV l; l << current_length[edge_map[e0]], current_length[edge_map[e1]], current_length[edge_map[e2]];
        TM A_inv; T Ai = layout2D(l0, A_inv);
        T element_energy;
        if (use_reference_C_tensor)
            computeGeodesicEnergyWithReferenceC(E, l, A_inv, Ai, reference_C_entries, element_energy);
        else
            computeGeodesicNHEnergyWithC(lambda, mu, l, A_inv, Ai, element_energy);
        // // viF^TFvi = l*2
        // Vector<T, 3> b; b << l0*l0, l1*l1, l2*l2;
        // Vector<T, 3> C_entries = A.inverse() * b;

        energy += element_energy;
    });
}

void IntrinsicSimulation::addGeodesicNHForceEntry(VectorXT& residual)
{
    iterateTriangleSerial([&](Triangle tri, int tri_idx){
        Edge e0, e1, e2;
		getTriangleEdges(tri, e0, e1, e2);
        TV l0; l0 << undeformed_length[edge_map[e0]], undeformed_length[edge_map[e1]], undeformed_length[edge_map[e2]];
        TV l; l << current_length[edge_map[e0]], current_length[edge_map[e1]], current_length[edge_map[e2]];
        TM A_inv; T Ai = layout2D(l0, A_inv);
        
        Vector<T, 3> dedl;
        if (use_reference_C_tensor)
            computeGeodesicEnergyWithReferenceCGradient(E, l, A_inv, Ai, reference_C_entries, dedl);
        else
            computeGeodesicNHEnergyWithCGradient(lambda, mu, l, A_inv, Ai, dedl);
        
        // TV b; b << l[0]*l[0], l[1]*l[1], l[2]*l[2];
        // TV C_entries = A_inv * b;
        
        // std::cout << C_entries.transpose() << " " << Ai << std::endl;
        // std::cout << dedl.transpose() << std::endl;
        // std::getchar();
        // computeGeodesicNHEnergyGradient(lambda, mu, undeformed_area[tri_idx], TV(l0, l1, l2), X_inv_geodesic[tri_idx], dedl);

        if (two_way_coupling)
        {
			
            VectorXT dldq0, dldq1, dldq2;
            std::vector<int> dof_indices0, dof_indices1, dof_indices2;
            computeGeodesicLengthGradientCoupled(e0, dldq0, dof_indices0);
			computeGeodesicLengthGradientCoupled(e1, dldq1, dof_indices1);
			computeGeodesicLengthGradientCoupled(e2, dldq2, dof_indices2);

			addForceEntry(residual, {e0[0], e0[1]}, -dedl[0] * dldq0.segment<4>(0));
			addForceEntry(residual, {e1[0], e1[1]}, -dedl[1] * dldq1.segment<4>(0));
			addForceEntry(residual, {e2[0], e2[1]}, -dedl[2] * dldq2.segment<4>(0));
            
            addForceEntry<3>(residual, dof_indices0, 
                -dedl[0] * dldq0.segment(4, dof_indices0.size() * 3), /*shift = */shell_dof_start);
			addForceEntry<3>(residual, dof_indices1, 
                -dedl[1] * dldq1.segment(4, dof_indices1.size() * 3), /*shift = */shell_dof_start);
			addForceEntry<3>(residual, dof_indices2, 
                -dedl[2] * dldq2.segment(4, dof_indices2.size() * 3), /*shift = */shell_dof_start);
                }
        else
        {
            Vector<T, 4> dl0dw0, dl1dw1, dl2dw2;
            computeGeodesicLengthGradient(e0, dl0dw0);
            computeGeodesicLengthGradient(e1, dl1dw1);
            computeGeodesicLengthGradient(e2, dl2dw2);

            addForceEntry(residual, {e0[0], e0[1]}, -dedl[0] * dl0dw0);
            addForceEntry(residual, {e1[0], e1[1]}, -dedl[1] * dl1dw1);
            addForceEntry(residual, {e2[0], e2[1]}, -dedl[2] * dl2dw2);
        }
    });
}
    
void IntrinsicSimulation::addGeodesicNHHessianEntry(std::vector<Entry>& entries)
{
    iterateTriangleSerial([&](Triangle tri, int tri_idx){
        Edge e0, e1, e2;
		getTriangleEdges(tri, e0, e1, e2);
        TV l0; l0 << undeformed_length[edge_map[e0]], undeformed_length[edge_map[e1]], undeformed_length[edge_map[e2]];
        TV l; l << current_length[edge_map[e0]], current_length[edge_map[e1]], current_length[edge_map[e2]];
        TM A_inv; T Ai = layout2D(l0, A_inv);
        
        Vector<T, 3> dedl;
        if (use_reference_C_tensor)
            computeGeodesicEnergyWithReferenceCGradient(E, l, A_inv, Ai, reference_C_entries, dedl);
        else
            computeGeodesicNHEnergyWithCGradient(lambda, mu, l, A_inv, Ai, dedl);
        Matrix<T, 3, 3> d2edl2;
        if (use_reference_C_tensor)
            computeGeodesicEnergyWithReferenceCHessian(E, l, A_inv, Ai, reference_C_entries, d2edl2);
        else
            computeGeodesicNHEnergyWithCHessian(lambda, mu, l, A_inv, Ai, d2edl2);
        // min e(l(w))
        // d2edw2 = dldw^T d2edl2 dldw + dedl d2ldw2

        
        // for (int i = 0; i < 2; i++)
        // {
        //     for (int j = 0; j < 2; j++)
        //     {
        //         if (std::isnan(d2edl2(i, j)))
        //         {
        //             TV b; b << l[0]*l[0], l[1]*l[1], l[2]*l[2];
        //             TV C_entries = A_inv * b;
        //             std::cout << d2edl2 << std::endl;
        //             std::cout << C_entries.transpose() << std::endl;
        //             std::getchar();
        //         }
        //     }
        // }
        // std::cout<< d2edl2 << std::endl;
        // std::getchar();

        if (two_way_coupling)
        {
			VectorXT dldq0, dldq1, dldq2;
			MatrixXT d2ldq02, d2ldq12, d2ldq22;
            std::vector<int> dof_indices0, dof_indices1, dof_indices2;
            computeGeodesicLengthGradientAndHessianCoupled(e0, dldq0, d2ldq02, dof_indices0);
			computeGeodesicLengthGradientAndHessianCoupled(e1, dldq1, d2ldq12, dof_indices1);
			computeGeodesicLengthGradientAndHessianCoupled(e2, dldq2, d2ldq22, dof_indices2);

			std::unordered_map<int, int> index_map;
			index_map[tri[0]] = 0; index_map[tri[1]] = 1; index_map[tri[2]] = 2;

			std::unordered_set<int> unique_dof;
			std::vector<int> idx_v_full;
			for (int idx : dof_indices0)
				unique_dof.insert(idx);
			for (int idx : dof_indices1)
				unique_dof.insert(idx);
			for (int idx : dof_indices2)
				unique_dof.insert(idx);

			std::unordered_map<int, int> index_map_v;
			int cnt = 0;
			for (int idx : unique_dof)
			{
				idx_v_full.push_back(idx);
				index_map_v[idx] = cnt++;
			}

			int nv_total = index_map_v.size();

			MatrixXT dldq(3, 6 + nv_total * 3); 
			dldq.setZero();
			VectorXT row0(6 + nv_total * 3); row0.setZero();
			VectorXT row1(6 + nv_total * 3); row1.setZero();
			VectorXT row2(6 + nv_total * 3); row2.setZero();

			addForceEntry<2>(row0, {index_map[e0[0]], index_map[e0[1]]}, dldq0.segment<4>(0));
			addForceEntry<2>(row1, {index_map[e1[0]], index_map[e1[1]]}, dldq1.segment<4>(0));
			addForceEntry<2>(row2, {index_map[e2[0]], index_map[e2[1]]}, dldq2.segment<4>(0));
			

			std::vector<int> dof0_remapped, dof1_remapped, dof2_remapped;
			for (int idx : dof_indices0)
				dof0_remapped.push_back(index_map_v[idx]);
			for (int idx : dof_indices1)
				dof1_remapped.push_back(index_map_v[idx]);
			for (int idx : dof_indices2)
				dof2_remapped.push_back(index_map_v[idx]);

			addForceEntry<3>(row0, dof0_remapped, dldq0.segment(4, dof_indices0.size() * 3), 6);
			addForceEntry<3>(row1, dof1_remapped, dldq1.segment(4, dof_indices1.size() * 3), 6);
			addForceEntry<3>(row2, dof2_remapped, dldq2.segment(4, dof_indices2.size() * 3), 6);
			
			dldq.row(0) = row0; dldq.row(1) = row1; dldq.row(2) = row2;

			VectorXT dedq = dedl.transpose() * dldq; // correct

			MatrixXT tensor_term(6 + nv_total * 3, 6 + nv_total *3);
			tensor_term.setZero();

			addHessianMatrixEntry(tensor_term, {index_map[e0[0]], index_map[e0[1]]}, dedl[0] * d2ldq02.block(0, 0, 4, 4));
			addHessianMatrixEntry(tensor_term, {index_map[e1[0]], index_map[e1[1]]}, dedl[1] * d2ldq12.block(0, 0, 4, 4));
			addHessianMatrixEntry(tensor_term, {index_map[e2[0]], index_map[e2[1]]}, dedl[2] * d2ldq22.block(0, 0, 4, 4));

			addHessianMatrixEntry<3, 3>(tensor_term, dof0_remapped, dedl[0] * d2ldq02.block(4, 4, dof_indices0.size() * 3, dof_indices0.size() * 3), 6, 6);
			addHessianMatrixEntry<3, 3>(tensor_term, dof1_remapped, dedl[1] * d2ldq12.block(4, 4, dof_indices1.size() * 3, dof_indices1.size() * 3), 6, 6);
			addHessianMatrixEntry<3, 3>(tensor_term, dof2_remapped, dedl[2] * d2ldq22.block(4, 4, dof_indices2.size() * 3, dof_indices2.size() * 3), 6, 6);

			addJacobianMatrixEntry<2, 3>(tensor_term, {index_map[e0[0]], index_map[e0[1]]}, dof0_remapped, dedl[0] * d2ldq02.block(0, 4, 4, dof_indices0.size() * 3), 0, 6);
			addJacobianMatrixEntry<3, 2>(tensor_term, dof0_remapped, {index_map[e0[0]], index_map[e0[1]]}, dedl[0] * d2ldq02.block(4, 0, dof_indices0.size() * 3, 4), 6, 0);

			addJacobianMatrixEntry<2, 3>(tensor_term, {index_map[e1[0]], index_map[e1[1]]}, dof1_remapped, dedl[1] * d2ldq12.block(0, 4, 4, dof_indices1.size() * 3), 0, 6);
			addJacobianMatrixEntry<3, 2>(tensor_term, dof1_remapped, {index_map[e1[0]], index_map[e1[1]]}, dedl[1] * d2ldq12.block(4, 0, dof_indices1.size() * 3, 4), 6, 0);

			addJacobianMatrixEntry<2, 3>(tensor_term, {index_map[e2[0]], index_map[e2[1]]}, dof2_remapped, dedl[2] * d2ldq22.block(0, 4, 4, dof_indices2.size() * 3), 0, 6);
			addJacobianMatrixEntry<3, 2>(tensor_term, dof2_remapped, {index_map[e2[0]], index_map[e2[1]]}, dedl[2] * d2ldq22.block(4, 0, dof_indices2.size() * 3, 4), 6, 0);

		
			
			MatrixXT hessian = (dedq * dedq.transpose());
			hessian += (dldq.transpose() * d2edl2 * dldq);
			hessian += tensor_term;


			addHessianEntry<2, 2>(entries, {tri[0], tri[1], tri[2]}, hessian.block(0, 0, 6, 6));
			addHessianEntry<3, 3>(entries, idx_v_full, hessian.block(6, 6, nv_total * 3, nv_total * 3), shell_dof_start, shell_dof_start);

			addJacobianEntry<2, 3>(entries, {tri[0], tri[1], tri[2]}, idx_v_full, hessian.block(0, 6, 6, nv_total * 3), 0, shell_dof_start);
			addJacobianEntry<3, 2>(entries, idx_v_full, {tri[0], tri[1], tri[2]}, hessian.block(6, 0, nv_total * 3, 6), shell_dof_start, 0);
			
		}
		else
        {
            Vector<T, 4> dl0dw0, dl1dw1, dl2dw2;
            Matrix<T, 4, 4> d2l0dw02, d2l1dw12, d2l2dw22;
            computeGeodesicLengthGradientAndHessian(e0, dl0dw0, d2l0dw02);
            computeGeodesicLengthGradientAndHessian(e1, dl1dw1, d2l1dw12);
            computeGeodesicLengthGradientAndHessian(e2, dl2dw2, d2l2dw22);
            std::unordered_map<int, int> index_map;
            index_map[tri[0]] = 0; index_map[tri[1]] = 1; index_map[tri[2]] = 2;
            MatrixXT dldw(3, 6); dldw.setZero();
            VectorXT row0(6); row0.setZero();
            VectorXT row1(6); row1.setZero();
            VectorXT row2(6); row2.setZero();
            addForceEntry(row0, {index_map[e0[0]], index_map[e0[1]]}, dl0dw0);
            addForceEntry(row1, {index_map[e1[0]], index_map[e1[1]]}, dl1dw1);
            addForceEntry(row2, {index_map[e2[0]], index_map[e2[1]]}, dl2dw2);
            dldw.row(0) = row0; dldw.row(1) = row1; dldw.row(2) = row2;

            MatrixXT tensor_term(6, 6); tensor_term.setZero();
            addHessianMatrixEntry(tensor_term, {index_map[e0[0]], index_map[e0[1]]}, dedl[0] * d2l0dw02);
            addHessianMatrixEntry(tensor_term, {index_map[e1[0]], index_map[e1[1]]}, dedl[1] * d2l1dw12);
            addHessianMatrixEntry(tensor_term, {index_map[e2[0]], index_map[e2[1]]}, dedl[2] * d2l2dw22);

            
            Matrix<T, 6, 6> hessian; hessian.setZero();
            hessian += dldw.transpose() * d2edl2 * dldw;
            hessian += tensor_term;
            addHessianEntry(entries, {tri[0], tri[1], tri[2]}, hessian);

        }
        
    });
}