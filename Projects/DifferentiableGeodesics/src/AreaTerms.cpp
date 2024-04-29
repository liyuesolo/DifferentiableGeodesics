#include "../include/IntrinsicSimulation.h"

void IntrinsicSimulation::areaLengthFormula(const Eigen::Matrix<double,3,1> & l, double& energy){
	double _i_var[11];
	_i_var[0] = (l(0,0))+(l(1,0));
	_i_var[1] = 2;
	_i_var[2] = (_i_var[0])+(l(2,0));
	_i_var[3] = (_i_var[2])/(_i_var[1]);
	_i_var[4] = (_i_var[3])-(l(0,0));
	_i_var[5] = (_i_var[3])-(l(1,0));
	_i_var[6] = (_i_var[3])*(_i_var[4]);
	_i_var[7] = (_i_var[3])-(l(2,0));
	_i_var[8] = (_i_var[6])*(_i_var[5]);
	_i_var[9] = (_i_var[8])*(_i_var[7]);
	_i_var[10] = std::sqrt(_i_var[9]);
	energy = _i_var[10];
}
void IntrinsicSimulation::areaLengthFormulaGradient(const Eigen::Matrix<double,3,1> & l, Eigen::Matrix<double, 3, 1>& energygradient){
	double _i_var[32];
	_i_var[0] = (l(0,0))+(l(1,0));
	_i_var[1] = 2;
	_i_var[2] = (_i_var[0])+(l(2,0));
	_i_var[3] = (_i_var[2])/(_i_var[1]);
	_i_var[4] = (_i_var[3])-(l(0,0));
	_i_var[5] = (_i_var[3])-(l(1,0));
	_i_var[6] = (_i_var[3])*(_i_var[4]);
	_i_var[7] = (_i_var[3])-(l(2,0));
	_i_var[8] = (_i_var[6])*(_i_var[5]);
	_i_var[9] = (_i_var[8])*(_i_var[7]);
	_i_var[10] = std::sqrt(_i_var[9]);
	_i_var[11] = (_i_var[1])*(_i_var[10]);
	_i_var[12] = 1;
	_i_var[13] = (_i_var[12])/(_i_var[11]);
	_i_var[14] = (_i_var[13])*(_i_var[7]);
	_i_var[15] = (_i_var[14])*(_i_var[5]);
	_i_var[16] = (_i_var[15])*(_i_var[4]);
	_i_var[17] = (_i_var[13])*(_i_var[8]);
	_i_var[18] = (_i_var[14])*(_i_var[6]);
	_i_var[19] = (_i_var[17])+(_i_var[16]);
	_i_var[20] = (_i_var[15])*(_i_var[3]);
	_i_var[21] = (_i_var[19])+(_i_var[18]);
	_i_var[22] = 0.5;
	_i_var[23] = (_i_var[21])+(_i_var[20]);
	_i_var[24] = -1;
	_i_var[25] = (_i_var[23])*(_i_var[22]);
	_i_var[26] = (_i_var[20])*(_i_var[24]);
	_i_var[27] = (_i_var[18])*(_i_var[24]);
	_i_var[28] = (_i_var[17])*(_i_var[24]);
	_i_var[29] = (_i_var[26])+(_i_var[25]);
	_i_var[30] = (_i_var[27])+(_i_var[25]);
	_i_var[31] = (_i_var[28])+(_i_var[25]);
	energygradient(0,0) = _i_var[29];
	energygradient(1,0) = _i_var[30];
	energygradient(2,0) = _i_var[31];
}
void IntrinsicSimulation::areaLengthFormulaHessian(const Eigen::Matrix<double,3,1> & l, Eigen::Matrix<double, 3, 3>& energyhessian){
	double _i_var[102];
	_i_var[0] = (l(0,0))+(l(1,0));
	_i_var[1] = 2;
	_i_var[2] = (_i_var[0])+(l(2,0));
	_i_var[3] = (_i_var[2])/(_i_var[1]);
	_i_var[4] = (_i_var[3])-(l(0,0));
	_i_var[5] = (_i_var[3])-(l(1,0));
	_i_var[6] = (_i_var[3])*(_i_var[4]);
	_i_var[7] = (_i_var[3])-(l(2,0));
	_i_var[8] = (_i_var[6])*(_i_var[5]);
	_i_var[9] = (_i_var[8])*(_i_var[7]);
	_i_var[10] = std::sqrt(_i_var[9]);
	_i_var[11] = (_i_var[1])*(_i_var[10]);
	_i_var[12] = (_i_var[11])*(_i_var[11]);
	_i_var[13] = 1;
	_i_var[14] = (_i_var[13])/(_i_var[12]);
	_i_var[15] = -(_i_var[14]);
	_i_var[16] = (_i_var[13])/(_i_var[11]);
	_i_var[17] = (_i_var[15])*(_i_var[1]);
	_i_var[18] = (_i_var[17])*(_i_var[16]);
	_i_var[19] = (_i_var[7])*(_i_var[8]);
	_i_var[20] = (_i_var[7])*(_i_var[7]);
	_i_var[21] = (_i_var[19])*(_i_var[18]);
	_i_var[22] = (_i_var[20])*(_i_var[18]);
	_i_var[23] = (_i_var[5])*(_i_var[6]);
	_i_var[24] = (_i_var[5])*(_i_var[5]);
	_i_var[25] = (_i_var[21])+(_i_var[16]);
	_i_var[26] = (_i_var[16])*(_i_var[7]);
	_i_var[27] = (_i_var[23])*(_i_var[22]);
	_i_var[28] = (_i_var[24])*(_i_var[22]);
	_i_var[29] = (_i_var[5])*(_i_var[25]);
	_i_var[30] = (_i_var[27])+(_i_var[26]);
	_i_var[31] = (_i_var[1])*(_i_var[4]);
	_i_var[32] = (_i_var[8])*(_i_var[8]);
	_i_var[33] = (_i_var[6])*(_i_var[28]);
	_i_var[34] = (_i_var[3])*(_i_var[29]);
	_i_var[35] = (_i_var[4])*(_i_var[30]);
	_i_var[36] = (_i_var[6])*(_i_var[25]);
	_i_var[37] = (_i_var[4])*(_i_var[4]);
	_i_var[38] = (_i_var[31])*(_i_var[29]);
	_i_var[39] = (_i_var[32])*(_i_var[18]);
	_i_var[40] = -1;
	_i_var[41] = (_i_var[26])*(_i_var[5]);
	_i_var[42] = (_i_var[34])+(_i_var[33]);
	_i_var[43] = (_i_var[36])+(_i_var[35]);
	_i_var[44] = (_i_var[37])*(_i_var[28]);
	_i_var[45] = (_i_var[39])+(_i_var[38]);
	_i_var[46] = (_i_var[6])*(_i_var[6]);
	_i_var[47] = (_i_var[40])*(_i_var[29]);
	_i_var[48] = (_i_var[3])*(_i_var[3]);
	_i_var[49] = (_i_var[3])*(_i_var[30]);
	_i_var[50] = (_i_var[42])+(_i_var[41]);
	_i_var[51] = (_i_var[1])*(_i_var[43]);
	_i_var[52] = (_i_var[45])+(_i_var[44]);
	_i_var[53] = (_i_var[46])*(_i_var[22]);
	_i_var[54] = (_i_var[4])*(_i_var[47]);
	_i_var[55] = (_i_var[40])*(_i_var[39]);
	_i_var[56] = (_i_var[48])*(_i_var[28]);
	_i_var[57] = (_i_var[50])+(_i_var[49]);
	_i_var[58] = (_i_var[52])+(_i_var[51]);
	_i_var[59] = (_i_var[40])*(_i_var[53]);
	_i_var[60] = (_i_var[40])*(_i_var[43]);
	_i_var[61] = (_i_var[40])*(_i_var[36]);
	_i_var[62] = (_i_var[55])+(_i_var[54]);
	_i_var[63] = (_i_var[40])*(_i_var[56]);
	_i_var[64] = (_i_var[40])*(_i_var[57]);
	_i_var[65] = (_i_var[1])*(_i_var[57]);
	_i_var[66] = (_i_var[58])+(_i_var[53]);
	_i_var[67] = (_i_var[40])*(_i_var[49]);
	_i_var[68] = (_i_var[60])+(_i_var[59]);
	_i_var[69] = (_i_var[3])*(_i_var[47]);
	_i_var[70] = (_i_var[62])+(_i_var[61]);
	_i_var[71] = (_i_var[64])+(_i_var[63]);
	_i_var[72] = 0.5;
	_i_var[73] = (_i_var[66])+(_i_var[65]);
	_i_var[74] = (_i_var[68])+(_i_var[67]);
	_i_var[75] = (_i_var[70])+(_i_var[69]);
	_i_var[76] = (_i_var[72])*(_i_var[71]);
	_i_var[77] = (_i_var[40])*(_i_var[67]);
	_i_var[78] = (_i_var[73])+(_i_var[56]);
	_i_var[79] = 0.25;
	_i_var[80] = (_i_var[72])*(_i_var[74]);
	_i_var[81] = (_i_var[72])*(_i_var[75]);
	_i_var[82] = (_i_var[1])*(_i_var[76]);
	_i_var[83] = (_i_var[77])+(_i_var[76]);
	_i_var[84] = (_i_var[79])*(_i_var[78]);
	_i_var[85] = (_i_var[40])*(_i_var[69]);
	_i_var[86] = (_i_var[1])*(_i_var[80]);
	_i_var[87] = (_i_var[40])*(_i_var[61]);
	_i_var[88] = (_i_var[1])*(_i_var[81]);
	_i_var[89] = (_i_var[56])+(_i_var[82]);
	_i_var[90] = (_i_var[83])+(_i_var[80]);
	_i_var[91] = (_i_var[81])+(_i_var[84]);
	_i_var[92] = (_i_var[85])+(_i_var[76]);
	_i_var[93] = (_i_var[53])+(_i_var[86]);
	_i_var[94] = (_i_var[87])+(_i_var[80]);
	_i_var[95] = (_i_var[39])+(_i_var[88]);
	_i_var[96] = (_i_var[89])+(_i_var[84]);
	_i_var[97] = (_i_var[90])+(_i_var[84]);
	_i_var[98] = (_i_var[92])+(_i_var[91]);
	_i_var[99] = (_i_var[93])+(_i_var[84]);
	_i_var[100] = (_i_var[94])+(_i_var[91]);
	_i_var[101] = (_i_var[95])+(_i_var[84]);
	energyhessian(0,0) = _i_var[96];
	energyhessian(1,0) = _i_var[97];
	energyhessian(2,0) = _i_var[98];
	energyhessian(0,1) = _i_var[97];
	energyhessian(1,1) = _i_var[99];
	energyhessian(2,1) = _i_var[100];
	energyhessian(0,2) = _i_var[98];
	energyhessian(1,2) = _i_var[100];
	energyhessian(2,2) = _i_var[101];
}


void IntrinsicSimulation::computeAllTriangleArea(VectorXT& area)
{
    area.resize(triangles.size());
    int cnt = 0;
    for (const Triangle& tri : triangles)
    {
        Edge e0(tri[0], tri[1]), e1(tri[1], tri[2]), e2(tri[2], tri[0]);
        T l0 = current_length[edge_map[e0]];
        T l1 = current_length[edge_map[e1]];
        T l2 = current_length[edge_map[e2]];

        T s = (l0 + l1 + l2) / 2.0;
        area[cnt] = std::sqrt(s * (s - l0) * (s - l1) * (s - l2));
        // areaLengthFormula(TV(l0, l1, l2), area[cnt]);
        
        cnt++;
    }
}

void IntrinsicSimulation::addTriangleAreaEnergy(T w, T& energy)
{
    VectorXT current_area;
    computeAllTriangleArea(current_area);
    energy += w * (current_area - rest_area).dot(current_area - rest_area); 
}

void IntrinsicSimulation::addTriangleAreaForceEntries(T w, VectorXT& residual)
{
    VectorXT current_area;
    computeAllTriangleArea(current_area);
    int cnt = 0;
    for (const Triangle& tri : triangles)
    {
        T coeff = 2.0 * w * (current_area[cnt] - rest_area[cnt]);
        Edge e0, e1, e2;
		getTriangleEdges(tri, e0, e1, e2);
        T l0 = current_length[edge_map[e0]];
        T l1 = current_length[edge_map[e1]];
        T l2 = current_length[edge_map[e2]];
        TV l(l0, l1, l2);
        TV dadl;
        areaLengthFormulaGradient(l, dadl);
        

		if (two_way_coupling)
        {
			
            VectorXT dldq0, dldq1, dldq2;
            std::vector<int> dof_indices0, dof_indices1, dof_indices2;
            computeGeodesicLengthGradientCoupled(e0, dldq0, dof_indices0);
			computeGeodesicLengthGradientCoupled(e1, dldq1, dof_indices1);
			computeGeodesicLengthGradientCoupled(e2, dldq2, dof_indices2);

			addForceEntry(residual, {e0[0], e0[1]}, -dadl[0] * dldq0.segment<4>(0) * coeff);
			addForceEntry(residual, {e1[0], e1[1]}, -dadl[1] * dldq1.segment<4>(0) * coeff);
			addForceEntry(residual, {e2[0], e2[1]}, -dadl[2] * dldq2.segment<4>(0) * coeff);
            
            addForceEntry<3>(residual, dof_indices0, 
                -dadl[0] * dldq0.segment(4, dof_indices0.size() * 3) * coeff, /*shift = */shell_dof_start);
			addForceEntry<3>(residual, dof_indices1, 
                -dadl[1] * dldq1.segment(4, dof_indices1.size() * 3) * coeff, /*shift = */shell_dof_start);
			addForceEntry<3>(residual, dof_indices2, 
                -dadl[2] * dldq2.segment(4, dof_indices2.size() * 3) * coeff, /*shift = */shell_dof_start);
		
        }
        else
        {
            Vector<T, 4> dl0dw0, dl1dw1, dl2dw2;
			computeGeodesicLengthGradient(e0, dl0dw0);
			computeGeodesicLengthGradient(e1, dl1dw1);
			computeGeodesicLengthGradient(e2, dl2dw2);

			addForceEntry(residual, {e0[0], e0[1]}, -dadl[0] * dl0dw0 * coeff);
			addForceEntry(residual, {e1[0], e1[1]}, -dadl[1] * dl1dw1 * coeff);
			addForceEntry(residual, {e2[0], e2[1]}, -dadl[2] * dl2dw2 * coeff);
        }
        cnt++;
    }
}

void IntrinsicSimulation::addTriangleAreaHessianEntries(T w, std::vector<Entry>& entries)
{
    VectorXT current_area;
    computeAllTriangleArea(current_area);
	int cnt = 0;
    for (const Triangle& tri : triangles)
    {
        Edge e0, e1, e2;
		T da = current_area[cnt] - rest_area[cnt];
		getTriangleEdges(tri, e0, e1, e2);
        T l0 = current_length[edge_map[e0]];
        T l1 = current_length[edge_map[e1]];
        T l2 = current_length[edge_map[e2]];
        TV l(l0, l1, l2);
        TV dadl;
        areaLengthFormulaGradient(l, dadl);
		TM d2adl2;
		areaLengthFormulaHessian(l, d2adl2);

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

			VectorXT dadq = dadl.transpose() * dldq; // correct

			MatrixXT tensor_term(6 + nv_total * 3, 6 + nv_total *3);
			tensor_term.setZero();

			addHessianMatrixEntry(tensor_term, {index_map[e0[0]], index_map[e0[1]]}, dadl[0] * d2ldq02.block(0, 0, 4, 4));
			addHessianMatrixEntry(tensor_term, {index_map[e1[0]], index_map[e1[1]]}, dadl[1] * d2ldq12.block(0, 0, 4, 4));
			addHessianMatrixEntry(tensor_term, {index_map[e2[0]], index_map[e2[1]]}, dadl[2] * d2ldq22.block(0, 0, 4, 4));

			addHessianMatrixEntry<3, 3>(tensor_term, dof0_remapped, dadl[0] * d2ldq02.block(4, 4, dof_indices0.size() * 3, dof_indices0.size() * 3), 6, 6);
			addHessianMatrixEntry<3, 3>(tensor_term, dof1_remapped, dadl[1] * d2ldq12.block(4, 4, dof_indices1.size() * 3, dof_indices1.size() * 3), 6, 6);
			addHessianMatrixEntry<3, 3>(tensor_term, dof2_remapped, dadl[2] * d2ldq22.block(4, 4, dof_indices2.size() * 3, dof_indices2.size() * 3), 6, 6);

			addJacobianMatrixEntry<2, 3>(tensor_term, {index_map[e0[0]], index_map[e0[1]]}, dof0_remapped, dadl[0] * d2ldq02.block(0, 4, 4, dof_indices0.size() * 3), 0, 6);
			addJacobianMatrixEntry<3, 2>(tensor_term, dof0_remapped, {index_map[e0[0]], index_map[e0[1]]}, dadl[0] * d2ldq02.block(4, 0, dof_indices0.size() * 3, 4), 6, 0);

			addJacobianMatrixEntry<2, 3>(tensor_term, {index_map[e1[0]], index_map[e1[1]]}, dof1_remapped, dadl[1] * d2ldq12.block(0, 4, 4, dof_indices1.size() * 3), 0, 6);
			addJacobianMatrixEntry<3, 2>(tensor_term, dof1_remapped, {index_map[e1[0]], index_map[e1[1]]}, dadl[1] * d2ldq12.block(4, 0, dof_indices1.size() * 3, 4), 6, 0);

			addJacobianMatrixEntry<2, 3>(tensor_term, {index_map[e2[0]], index_map[e2[1]]}, dof2_remapped, dadl[2] * d2ldq22.block(0, 4, 4, dof_indices2.size() * 3), 0, 6);
			addJacobianMatrixEntry<3, 2>(tensor_term, dof2_remapped, {index_map[e2[0]], index_map[e2[1]]}, dadl[2] * d2ldq22.block(4, 0, dof_indices2.size() * 3, 4), 6, 0);

			tensor_term *=  2.0 * w * da;

			
			MatrixXT hessian = 2.0 * w * (dadq * dadq.transpose());
			hessian += 2.0 * w * da * (dldq.transpose() * d2adl2 * dldq);
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

			MatrixXT dadld2ldw2(6, 6); dadld2ldw2.setZero();
			addHessianMatrixEntry(dadld2ldw2, {index_map[e0[0]], index_map[e0[1]]}, dadl[0] * d2l0dw02);
			addHessianMatrixEntry(dadld2ldw2, {index_map[e1[0]], index_map[e1[1]]}, dadl[1] * d2l1dw12);
			addHessianMatrixEntry(dadld2ldw2, {index_map[e2[0]], index_map[e2[1]]}, dadl[2] * d2l2dw22);

			Matrix<T, 6, 6> tensor_term; tensor_term.setZero();
			tensor_term += 2.0 * w * da * dadld2ldw2;

			Vector<T, 6> dadw = dadl.transpose() * dldw;
			Matrix<T, 6, 6> hessian; hessian.setZero();
			hessian += 2.0 * w * (dadw * dadw.transpose());
			hessian += 2.0 * w * da * (dldw.transpose() * d2adl2 * dldw);
			hessian += tensor_term;
			addHessianEntry(entries, {tri[0], tri[1], tri[2]}, hessian);
		}
		cnt++;
    }
}