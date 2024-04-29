#include "../autodiff/Misc.h"
#include "../include/IntrinsicSimulation.h"

// feeling lazy doing this term by hand
void computedgdc_d2cdwdv(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, const Eigen::Matrix<double,3,1> & v4, 
	const Eigen::Matrix<double,3,1> & v5, const Eigen::Matrix<double,4,1> & w, const Eigen::Matrix<double,6,1> & dgdc, Eigen::Matrix<double, 4, 18>& dgdcd2gdvdw){
	double _i_var[8];
	_i_var[0] = -1;
	_i_var[1] = 0;
	_i_var[2] = (dgdc(0,0))*(_i_var[0]);
	_i_var[3] = (dgdc(1,0))*(_i_var[0]);
	_i_var[4] = (dgdc(2,0))*(_i_var[0]);
	_i_var[5] = (dgdc(3,0))*(_i_var[0]);
	_i_var[6] = (dgdc(4,0))*(_i_var[0]);
	_i_var[7] = (dgdc(5,0))*(_i_var[0]);
	dgdcd2gdvdw(0,0) = dgdc(0,0);
	dgdcd2gdvdw(1,0) = _i_var[1];
	dgdcd2gdvdw(2,0) = _i_var[1];
	dgdcd2gdvdw(3,0) = _i_var[1];
	dgdcd2gdvdw(0,1) = dgdc(1,0);
	dgdcd2gdvdw(1,1) = _i_var[1];
	dgdcd2gdvdw(2,1) = _i_var[1];
	dgdcd2gdvdw(3,1) = _i_var[1];
	dgdcd2gdvdw(0,2) = dgdc(2,0);
	dgdcd2gdvdw(1,2) = _i_var[1];
	dgdcd2gdvdw(2,2) = _i_var[1];
	dgdcd2gdvdw(3,2) = _i_var[1];
	dgdcd2gdvdw(0,3) = _i_var[1];
	dgdcd2gdvdw(1,3) = dgdc(0,0);
	dgdcd2gdvdw(2,3) = _i_var[1];
	dgdcd2gdvdw(3,3) = _i_var[1];
	dgdcd2gdvdw(0,4) = _i_var[1];
	dgdcd2gdvdw(1,4) = dgdc(1,0);
	dgdcd2gdvdw(2,4) = _i_var[1];
	dgdcd2gdvdw(3,4) = _i_var[1];
	dgdcd2gdvdw(0,5) = _i_var[1];
	dgdcd2gdvdw(1,5) = dgdc(2,0);
	dgdcd2gdvdw(2,5) = _i_var[1];
	dgdcd2gdvdw(3,5) = _i_var[1];
	dgdcd2gdvdw(0,6) = _i_var[2];
	dgdcd2gdvdw(1,6) = _i_var[2];
	dgdcd2gdvdw(2,6) = _i_var[1];
	dgdcd2gdvdw(3,6) = _i_var[1];
	dgdcd2gdvdw(0,7) = _i_var[3];
	dgdcd2gdvdw(1,7) = _i_var[3];
	dgdcd2gdvdw(2,7) = _i_var[1];
	dgdcd2gdvdw(3,7) = _i_var[1];
	dgdcd2gdvdw(0,8) = _i_var[4];
	dgdcd2gdvdw(1,8) = _i_var[4];
	dgdcd2gdvdw(2,8) = _i_var[1];
	dgdcd2gdvdw(3,8) = _i_var[1];
	dgdcd2gdvdw(0,9) = _i_var[1];
	dgdcd2gdvdw(1,9) = _i_var[1];
	dgdcd2gdvdw(2,9) = dgdc(3,0);
	dgdcd2gdvdw(3,9) = _i_var[1];
	dgdcd2gdvdw(0,10) = _i_var[1];
	dgdcd2gdvdw(1,10) = _i_var[1];
	dgdcd2gdvdw(2,10) = dgdc(4,0);
	dgdcd2gdvdw(3,10) = _i_var[1];
	dgdcd2gdvdw(0,11) = _i_var[1];
	dgdcd2gdvdw(1,11) = _i_var[1];
	dgdcd2gdvdw(2,11) = dgdc(5,0);
	dgdcd2gdvdw(3,11) = _i_var[1];
	dgdcd2gdvdw(0,12) = _i_var[1];
	dgdcd2gdvdw(1,12) = _i_var[1];
	dgdcd2gdvdw(2,12) = _i_var[1];
	dgdcd2gdvdw(3,12) = dgdc(3,0);
	dgdcd2gdvdw(0,13) = _i_var[1];
	dgdcd2gdvdw(1,13) = _i_var[1];
	dgdcd2gdvdw(2,13) = _i_var[1];
	dgdcd2gdvdw(3,13) = dgdc(4,0);
	dgdcd2gdvdw(0,14) = _i_var[1];
	dgdcd2gdvdw(1,14) = _i_var[1];
	dgdcd2gdvdw(2,14) = _i_var[1];
	dgdcd2gdvdw(3,14) = dgdc(5,0);
	dgdcd2gdvdw(0,15) = _i_var[1];
	dgdcd2gdvdw(1,15) = _i_var[1];
	dgdcd2gdvdw(2,15) = _i_var[5];
	dgdcd2gdvdw(3,15) = _i_var[5];
	dgdcd2gdvdw(0,16) = _i_var[1];
	dgdcd2gdvdw(1,16) = _i_var[1];
	dgdcd2gdvdw(2,16) = _i_var[6];
	dgdcd2gdvdw(3,16) = _i_var[6];
	dgdcd2gdvdw(0,17) = _i_var[1];
	dgdcd2gdvdw(1,17) = _i_var[1];
	dgdcd2gdvdw(2,17) = _i_var[7];
	dgdcd2gdvdw(3,17) = _i_var[7];
}


// compute the gradient for two way coupling
// computing eqn (33), (34)
// naming convention follows what's in the paper
void IntrinsicSimulation::computeGeodesicLengthGradientCoupled(const Edge& edge, VectorXT& dldq,
    std::vector<int>& dof_indices)
{
    int edge_idx = edge_map[edge];
    SurfacePoint vA = surface_points[edge[0]];
    SurfacePoint vB = surface_points[edge[1]];

    std::vector<SurfacePoint> path = paths[edge_idx];
    std::vector<IxnData> ixn_data = ixn_data_list[edge_idx];
    
    int length = path.size();

    IV tri0, tri1;
    getTriangleIndex(surface_points[edge[0]].face, tri0);
    getTriangleIndex(surface_points[edge[1]].face, tri1);
    std::unordered_set<int> unique_ids;
    
    for (int i = 0; i < 3; i++)
    {
        unique_ids.insert(tri0[i]);
        unique_ids.insert(tri1[i]);
    }
    for (int i = 0; i < length - 2; i++)
    {
        unique_ids.insert(ixn_data[1+i].start_vtx_idx);
        unique_ids.insert(ixn_data[1+i].end_vtx_idx);
    }

    dof_indices.resize(unique_ids.size());

    std::unordered_map<int, int> dof_id_map;
    int _cnt = 0;
    for (const int &id : unique_ids)
    {
        dof_indices[_cnt] = id;
        dof_id_map[id] = _cnt++;
    }
    int nv = unique_ids.size() * 3;
    int nx = (length - 2) * 3;
    dldq.resize(4 + nv); dldq.setZero();
    
    TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
    TV v1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));

    TV v10 = toTV(geometry->vertexPositions[surface_points[edge[0]].face.halfedge().vertex()]);
    TV v11 = toTV(geometry->vertexPositions[surface_points[edge[0]].face.halfedge().next().vertex()]);
    TV v12 = toTV(geometry->vertexPositions[surface_points[edge[0]].face.halfedge().next().next().vertex()]);

    TV v20 = toTV(geometry->vertexPositions[surface_points[edge[1]].face.halfedge().vertex()]);
    TV v21 = toTV(geometry->vertexPositions[surface_points[edge[1]].face.halfedge().next().vertex()]);
    TV v22 = toTV(geometry->vertexPositions[surface_points[edge[1]].face.halfedge().next().next().vertex()]);

    // two end points first, then edge vertices for the intersections
    MatrixXT dxdv(nx, nv); dxdv.setZero();
    VectorXT dldx(nx); dldx.setZero();
    TV dldc0, dldc1;
    if (length == 2 || Euclidean)
    {
        dldc0 = -(v1 - v0).normalized();   
        dldc1 = -dldc0;
    }
    else
    {
        TV ixn0 = toTV(path[1].interpolate(geometry->vertexPositions));
        dldc0 = -(ixn0 - v0).normalized();

        TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
        dldc1 = -(ixn1 - v1).normalized();

        for (int ixn_id = 0; ixn_id < length - 2; ixn_id++)
        {
            T t = ixn_data_list[edge_idx][1+ixn_id].t;
            T t_hat = t;
            if (use_t_wrapper)
                t_hat = wrapper_t<0>(t);

            int start_vtx_idx_local = dof_id_map[ixn_data_list[edge_idx][1+ixn_id].start_vtx_idx];
            int end_vtx_idx_local = dof_id_map[ixn_data_list[edge_idx][1+ixn_id].end_vtx_idx];
            
            dxdv.block(ixn_id*3, start_vtx_idx_local * 3, 3, 3) += Matrix<T, 3, 3>::Identity() * (1.0 - t_hat);
            dxdv.block(ixn_id*3, end_vtx_idx_local * 3, 3, 3) += Matrix<T, 3, 3>::Identity() * t_hat;

            
            // j is the current point
            TV ixn_i = toTV(path[1 + ixn_id-1].interpolate(geometry->vertexPositions));
            TV ixn_j = toTV(path[1 + ixn_id].interpolate(geometry->vertexPositions));
            TV ixn_k = toTV(path[1 + ixn_id+1].interpolate(geometry->vertexPositions));

            // if this is the start and end point of the geodsic curve
            if (ixn_data[1 + ixn_id-1].t != -1.0) 
            {
                ixn_i = ixn_data[1 + ixn_id-1].end * 
                    wrapper_t<0>(ixn_data[1 + ixn_id-1].t) 
                + ixn_data[1 + ixn_id-1].start * 
                (1.0 - wrapper_t<0>(ixn_data[1 + ixn_id-1].t));    
            }
            if (ixn_data[1 + ixn_id].t != -1.0)
            {
                ixn_j = ixn_data[1 + ixn_id].end * wrapper_t<0>(ixn_data[1 + ixn_id].t) 
                + ixn_data[1 + ixn_id].start * (1.0 - wrapper_t<0>(ixn_data[1 + ixn_id].t));    
            }
            if (ixn_data[1 + ixn_id+1].t != -1.0)
            {
                ixn_k = ixn_data[1 + ixn_id+1].end * wrapper_t<0>(ixn_data[1 + ixn_id+1].t) 
                + ixn_data[1 + ixn_id+1].start * (1.0 - wrapper_t<0>(ixn_data[1 + ixn_id+1].t));    
            }

            dldx.segment<3>(ixn_id*3) += -(ixn_i-ixn_j).normalized() + -(ixn_k-ixn_j).normalized();
        }
    }

    Vector<T, 6> dldc; dldc.setZero();
    dldc.segment<3>(0) = dldc0;
    dldc.segment<3>(3) = dldc1;

    Matrix<T, 6, 4> dcdw; dcdw.setZero();
    dcdw.block(0, 0, 3, 1) = v10 - v12;
    dcdw.block(0, 1, 3, 1) = v11 - v12;

    dcdw.block(3, 2, 3, 1) = v20 - v22;
    dcdw.block(3, 3, 3, 1) = v21 - v22;

    dldq.segment<4>(0) = dldc.transpose() * dcdw;

    for (int i = 0; i < 3; i++)
    {
        dldq.segment<3>(4+dof_id_map[tri0[i]]*3) += dldc.segment<3>(0) * vA.faceCoords[i];
        dldq.segment<3>(4+dof_id_map[tri1[i]]*3) += dldc.segment<3>(3) * vB.faceCoords[i];
    }

    if (length > 2)
    {
        dldq.segment(4, nv) += dldx.transpose() *  dxdv;
    }
}


// compute the gradient and hessian together 
// two way coupling
// computing eqn (33), (34), (40-42)
// naming convention follows what's in the paper
void IntrinsicSimulation::computeGeodesicLengthGradientAndHessianCoupled(const Edge& edge, 
        VectorXT& dldq, MatrixXT& d2ldq2, std::vector<int>& dof_indices)
{
    int edge_idx = edge_map[edge];
    SurfacePoint vA = surface_points[edge[0]].inSomeFace();
    SurfacePoint vB = surface_points[edge[1]].inSomeFace();

    
    T l = current_length[edge_idx];
    std::vector<SurfacePoint> path = paths[edge_idx];
    std::vector<IxnData> ixn_data = ixn_data_list[edge_idx];
    
    int length = path.size();
    int ixn_dof = (length - 2) * 3;

    IV tri0, tri1;
    getTriangleIndex(surface_points[edge[0]].face, tri0);
    getTriangleIndex(surface_points[edge[1]].face, tri1);

    std::unordered_set<int> unique_ids;
    
    for (int i = 0; i < 3; i++)
    {
        unique_ids.insert(tri0[i]);
        unique_ids.insert(tri1[i]);
    }
    for (int i = 0; i < length - 2; i++)
    {
        unique_ids.insert(ixn_data[1+i].start_vtx_idx);
        unique_ids.insert(ixn_data[1+i].end_vtx_idx);
        // std::cout << ixn_data[1+i].start_vtx_idx << " " << ixn_data[1+i].end_vtx_idx << std::endl;
    }


    dof_indices.resize(unique_ids.size());

    std::unordered_map<int, int> dof_id_map;
    int _cnt = 0;
    for (const int &id : unique_ids)
    {
        dof_indices[_cnt] = id;
        // std::cout << "dof " << id << std::endl;
        dof_id_map[id] = _cnt++;
    }
    int nv = unique_ids.size() * 3;
    int nt = (length - 2);
    int nx = ixn_dof;

    dldq.resize(4 + nv); dldq.setZero();
    d2ldq2.resize(4 + nv, 4 + nv);
    d2ldq2.setZero();

    
    TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
    TV v1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));

    TV v10 = toTV(geometry->vertexPositions[surface_points[edge[0]].face.halfedge().vertex()]);
    TV v11 = toTV(geometry->vertexPositions[surface_points[edge[0]].face.halfedge().next().vertex()]);
    TV v12 = toTV(geometry->vertexPositions[surface_points[edge[0]].face.halfedge().next().next().vertex()]);

    TV v20 = toTV(geometry->vertexPositions[surface_points[edge[1]].face.halfedge().vertex()]);
    TV v21 = toTV(geometry->vertexPositions[surface_points[edge[1]].face.halfedge().next().vertex()]);
    TV v22 = toTV(geometry->vertexPositions[surface_points[edge[1]].face.halfedge().next().next().vertex()]);

    // two end points first, then edge vertices for the intersections

    MatrixXT dxdv(nx, nv); dxdv.setZero();
    VectorXT dgdx(nx); dgdx.setZero();

    Vector<T, 6> dgdc; dgdc.setZero();
    Matrix<T, 6, 6> d2gdc2; d2gdc2.setZero();

    Matrix<T, 6, 4> dcdw; dcdw.setZero();
    dcdw.block(0, 0, 3, 1) = v10 - v12;
    dcdw.block(0, 1, 3, 1) = v11 - v12;

    dcdw.block(3, 2, 3, 1) = v20 - v22;
    dcdw.block(3, 3, 3, 1) = v21 - v22;

    
    TV dgdc0, dgdc1;
    MatrixXT d2gdxdc(ixn_dof, 6); d2gdxdc.setZero();
    MatrixXT d2gdx2(ixn_dof, ixn_dof); d2gdx2.setZero();
    MatrixXT dxdt(ixn_dof, nt); dxdt.setZero();

    Matrix<T, 6, 6> p2gpc2; p2gpc2.setZero();
    MatrixXT dtdc(nt, 6), dtdv(nt, nv); 
    dtdc.setZero(); dtdv.setZero();
    MatrixXT dgdx_d2xdtdv(nt, nv); dgdx_d2xdtdv.setZero();
    
    MatrixXT dxdtd2gdx2dxdt;

    if (hasSmallSegment(path))
    {
        // std::cout << "has small segment" << std::endl;
        // std::getchar();
    }

    if (length == 2 || Euclidean)
    {
        dgdc0 = -(v1 - v0).normalized();   
        dgdc1 = -dgdc0;
        dgdc.segment<3>(0) = dgdc0;
        dgdc.segment<3>(3) = dgdc1;
        d2gdc2.block(0, 0, 3, 3) = (TM::Identity() - dgdc0 * dgdc0.transpose()) / l;
        d2gdc2.block(3, 3, 3, 3) = d2gdc2.block(0, 0, 3, 3);
        d2gdc2.block(3, 0, 3, 3) = -d2gdc2.block(0, 0, 3, 3);
        d2gdc2.block(0, 3, 3, 3) = -d2gdc2.block(0, 0, 3, 3);
    }
    else
    {
        
        TV ixn0 = toTV(path[1].interpolate(geometry->vertexPositions));
        dgdc0 = -(ixn0 - v0).normalized();

        TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
        dgdc1 = -(ixn1 - v1).normalized();

        dgdc.segment<3>(0) = dgdc0;
        dgdc.segment<3>(3) = dgdc1;

        TM dl0dx0 = (TM::Identity() - dgdc0 * dgdc0.transpose()) / (ixn0 - v0).norm();

        d2gdx2.block(0, 0, 3, 3) += dl0dx0;
        d2gdxdc.block(0, 0, 3, 3) += -dl0dx0;

        for (int ixn_id = 0; ixn_id < length - 2; ixn_id++)
        {
            T t = ixn_data[1+ixn_id].t;
            
            T t_hat = t;
            T d_hat_dt = 1.0;
            if (use_t_wrapper)
            {
                t_hat = wrapper_t<0>(t);
                d_hat_dt = wrapper_t<1>(t);
            }

            int start_vtx_idx_local = dof_id_map[ixn_data[1+ixn_id].start_vtx_idx];
            int end_vtx_idx_local = dof_id_map[ixn_data[1+ixn_id].end_vtx_idx];

            dxdv.block(ixn_id*3, start_vtx_idx_local * 3, 3, 3) 
                += Matrix<T, 3, 3>::Identity() * (1.0 - t_hat);
            dxdv.block(ixn_id*3, end_vtx_idx_local * 3, 3, 3) 
                += Matrix<T, 3, 3>::Identity() * t_hat;
            
            // j is the current point
            TV ixn_i = toTV(path[1 + ixn_id-1].interpolate(geometry->vertexPositions));
            TV ixn_j = toTV(path[1 + ixn_id].interpolate(geometry->vertexPositions));
            TV ixn_k = toTV(path[1 + ixn_id+1].interpolate(geometry->vertexPositions));
            
            // if this is the start and end point of the geodsic curve
            if (ixn_data[1 + ixn_id-1].t != -1.0) 
            {
                ixn_i = ixn_data[1 + ixn_id-1].end * 
                    wrapper_t<0>(ixn_data[1 + ixn_id-1].t) 
                + ixn_data[1 + ixn_id-1].start * 
                (1.0 - wrapper_t<0>(ixn_data[1 + ixn_id-1].t));    
            }
            if (ixn_data[1 + ixn_id].t != -1.0)
            {
                ixn_j = ixn_data[1 + ixn_id].end * wrapper_t<0>(ixn_data[1 + ixn_id].t) 
                + ixn_data[1 + ixn_id].start * (1.0 - wrapper_t<0>(ixn_data[1 + ixn_id].t));    
            }
            if (ixn_data[1 + ixn_id+1].t != -1.0)
            {
                ixn_k = ixn_data[1 + ixn_id+1].end * wrapper_t<0>(ixn_data[1 + ixn_id+1].t) 
                + ixn_data[1 + ixn_id+1].start * (1.0 - wrapper_t<0>(ixn_data[1 + ixn_id+1].t));    
            }
            

            dgdx.segment<3>(ixn_id*3) += -(ixn_i-ixn_j).normalized() + -(ixn_k-ixn_j).normalized();

            TV x_start = ixn_data[1+ixn_id].start;
            TV x_end = ixn_data[1+ixn_id].end;
            dxdt.block(ixn_id * 3, ixn_id, 3, 1) = (x_end - x_start) * d_hat_dt;

            
            TV dgdx_local = dgdx.segment<3>(ixn_id*3);
            // dxi/dti = v1 - v0
            // d2xi/dtidv1 = I
            // d2xi/dtidv0 = -I
    
            dgdx_d2xdtdv.block(ixn_id, start_vtx_idx_local * 3, 1, 3) -= dgdx_local.transpose();
            dgdx_d2xdtdv.block(ixn_id, end_vtx_idx_local * 3, 1, 3) += dgdx_local.transpose();
            
        }

        for (int ixn_id = 0; ixn_id < length - 3; ixn_id++)
        {
            // std::cout << "inside" << std::endl;
            Matrix<T, 6, 6> hess;
            TV ixn_i = toTV(path[1 + ixn_id].interpolate(geometry->vertexPositions));
            TV ixn_j = toTV(path[1 + ixn_id+1].interpolate(geometry->vertexPositions));

            if (ixn_data[1 + ixn_id].t != -1.0)
            {
                ixn_i = ixn_data[1 + ixn_id].end * wrapper_t<0>(ixn_data[1 + ixn_id].t) 
                + ixn_data[1 + ixn_id].start * (1.0 - wrapper_t<0>(ixn_data[1 + ixn_id].t));    
            }
            if (ixn_data[1 + ixn_id+1].t != -1.0)
            {
                ixn_j = ixn_data[1 + ixn_id+1].end * wrapper_t<0>(ixn_data[1 + ixn_id+1].t) 
                + ixn_data[1 + ixn_id+1].start * (1.0 - wrapper_t<0>(ixn_data[1 + ixn_id+1].t));    
            }

            edgeLengthHessian(ixn_i, ixn_j, hess);
            d2gdx2.block(ixn_id*3, ixn_id * 3, 6, 6) += hess;
        }

        TM dlndxn = (TM::Identity() - dgdc1 * dgdc1.transpose()) / (ixn1 - v1).norm();
        d2gdx2.block(ixn_dof-3, ixn_dof-3, 3, 3) += dlndxn;
        d2gdxdc.block(ixn_dof-3, 3, 3, 3) += -dlndxn;

        MatrixXT dxdtd2gdxdc = dxdt.transpose() * d2gdxdc;
        dxdtd2gdx2dxdt = dxdt.transpose() * d2gdx2 * dxdt;
        
        if (dxdt.norm() < 1e-8)
            dtdc.setZero();
        else 
            dtdc = dxdtd2gdx2dxdt.colPivHouseholderQr().solve(-dxdtd2gdxdc);
              
        
        p2gpc2.block(0, 0, 3, 3) += dl0dx0;
        p2gpc2.block(3, 3, 3, 3) += dlndxn;

        d2gdc2 = p2gpc2 + d2gdxdc.transpose() * dxdt * dtdc;   // Eqn 20 in supplementary
    }
    
    // std::cout << "dtdv " << dtdv << std::endl;
    dldq.segment<4>(0) += dgdc.transpose() * dcdw;

    d2ldq2.block(0, 0, 4, 4) += dcdw.transpose() * d2gdc2 * dcdw; // Eqn 20 in supplementary

    for (int i = 0; i < 3; i++)
    {
        dldq.segment<3>(4+dof_id_map[tri0[i]]*3) += dgdc.segment<3>(0) * vA.faceCoords[i];
        dldq.segment<3>(4+dof_id_map[tri1[i]]*3) += dgdc.segment<3>(3) * vB.faceCoords[i];
    }

    if (length > 2)
    {
        dldq.segment(4, nv) += dgdx.transpose() *  dxdv;
        
        MatrixXT dcdv(6, nv); dcdv.setZero();
        for (int i = 0; i < 3; i++)
        {
            dcdv.block(0, dof_id_map[tri0[i]] * 3, 3, 3) += Matrix<T, 3, 3>::Identity() * vA.faceCoords[i];
            dcdv.block(3, dof_id_map[tri1[i]] * 3, 3, 3) += Matrix<T, 3, 3>::Identity() * vB.faceCoords[i];
        }
        
        if (dxdt.norm() < 1e-8)
            dtdv.setZero();
        else 
        {
            MatrixXT b = -dxdt.transpose() * d2gdxdc * dcdv 
                - dgdx_d2xdtdv 
                - dxdt.transpose() * d2gdx2 * dxdt * dtdc * dcdv 
                - dxdt.transpose() * d2gdx2 * dxdv;
            dtdv = dxdtd2gdx2dxdt.colPivHouseholderQr().solve(b);
        }
        

        // std::cout << "-------------" << std::endl;
        d2ldq2.block(4, 4, nv, nv) += dcdv.transpose() * p2gpc2 * dcdv;

        MatrixXT dxdvT_d2gdxdc_dcdv = dxdv.transpose() * d2gdxdc * dcdv;
        
        d2ldq2.block(4, 4, nv, nv) += dxdvT_d2gdxdc_dcdv + dxdvT_d2gdxdc_dcdv.transpose();
        
        MatrixXT dxdt_dtdc_dcdv = dxdt * dtdc * dcdv;
        MatrixXT dxdt_dtdv = dxdt * dtdv;
        MatrixXT dcdvT_d2gdxdc_dxdt_dt_dc_dc_dv = dcdv.transpose() * d2gdxdc.transpose() * dxdt_dtdc_dcdv;

        d2ldq2.block(4, 4, nv, nv) += dcdvT_d2gdxdc_dxdt_dt_dc_dc_dv + dcdvT_d2gdxdc_dxdt_dt_dc_dc_dv.transpose();

        MatrixXT dcdvT_d2gdxdc_dxdt_dt_dv = dcdv.transpose() * d2gdxdc.transpose() * dxdt_dtdv;
        d2ldq2.block(4, 4, nv, nv) += dcdvT_d2gdxdc_dxdt_dt_dv + dcdvT_d2gdxdc_dxdt_dt_dv.transpose();
        
        MatrixXT dxdt_dtdc_dcdv_plus_dxdt_dtdv_plus_dxdv
            = dxdt_dtdc_dcdv + dxdt*dtdv + dxdv;

        d2ldq2.block(4, 4, nv, nv) += dxdt_dtdc_dcdv_plus_dxdt_dtdv_plus_dxdv.transpose() *
            d2gdx2 * dxdt_dtdc_dcdv_plus_dxdt_dtdv_plus_dxdv;

        MatrixXT dgdx_d2xdtdv_dtdc_dcdv = dgdx_d2xdtdv.transpose() * dtdc * dcdv;
        MatrixXT dgdx_d2xdtdv_dtdv = dgdx_d2xdtdv.transpose() * dtdv;

        d2ldq2.block(4, 4, nv, nv) += dgdx_d2xdtdv_dtdc_dcdv + dgdx_d2xdtdv_dtdc_dcdv.transpose();
        d2ldq2.block(4, 4, nv, nv) += dgdx_d2xdtdv_dtdv + dgdx_d2xdtdv_dtdv.transpose();


        MatrixXT dcdwT_d2gdc2_dcdv = dcdw.transpose() * p2gpc2 * dcdv;
        d2ldq2.block(0, 4, 4, nv) += dcdwT_d2gdc2_dcdv;
        d2ldq2.block(4, 0, nv, 4) += dcdwT_d2gdc2_dcdv.transpose();

        MatrixXT long_term = dcdw.transpose() * d2gdxdc.transpose() * (dxdt_dtdc_dcdv + dxdt_dtdv + dxdv);
        d2ldq2.block(0, 4, 4, nv) += long_term;
        d2ldq2.block(4, 0, nv, 4) += long_term.transpose();

        MatrixXT dxdt_dtdc_dcdw = dxdt * dtdc * dcdw;
        MatrixXT dxdt_dtdc_dcdwT_d2gdxdc_dcdv = 
            dxdt_dtdc_dcdw.transpose() * d2gdxdc * dcdv;
        d2ldq2.block(0, 4, 4, nv) += dxdt_dtdc_dcdwT_d2gdxdc_dcdv;
        d2ldq2.block(4, 0, nv, 4) += dxdt_dtdc_dcdwT_d2gdxdc_dcdv.transpose();

        MatrixXT dxdt_dtdc_dcdwT_d2gdx2_xxxx = 
            dxdt_dtdc_dcdw.transpose() * d2gdx2 * (dxdt_dtdc_dcdv + dxdt_dtdv + dxdv);
        d2ldq2.block(0, 4, 4, nv) += dxdt_dtdc_dcdwT_d2gdx2_xxxx;
        d2ldq2.block(4, 0, nv, 4) += dxdt_dtdc_dcdwT_d2gdx2_xxxx.transpose();

        MatrixXT dtdc_dcdw_T_dgdx_d2xdtdv = (dtdc * dcdw).transpose() * dgdx_d2xdtdv;       
        d2ldq2.block(0, 4, 4, nv) += dtdc_dcdw_T_dgdx_d2xdtdv;
        d2ldq2.block(4, 0, nv, 4) += dtdc_dcdw_T_dgdx_d2xdtdv.transpose();

        Vector<T, 6> dgdc_plus_dgdx_dxdt_dtdc = dgdc + (dgdx.transpose() * dxdt * dtdc).transpose();
        MatrixXT dgdc_plus_dgdx_dxdt_dtdc_tensor_product_d2cdvdw(4, nv);
        dgdc_plus_dgdx_dxdt_dtdc_tensor_product_d2cdvdw.setZero();
        
        Matrix<T, 4, 18> dgdcd2gdvdw_ad;
        Vector<T, 4> w; w << vA.faceCoords.x, vA.faceCoords.y,vB.faceCoords.x,vB.faceCoords.y;
        computedgdc_d2cdwdv(v10, v11, v12, v20, v21, v22, w, dgdc_plus_dgdx_dxdt_dtdc, dgdcd2gdvdw_ad);
        // std::cout << dgdcd2gdvdw_ad << std::endl;
        // std::getchar();
        addJacobianMatrixEntry<2, 3>(dgdc_plus_dgdx_dxdt_dtdc_tensor_product_d2cdvdw,
            {0, 1}, 
            {
                dof_id_map[tri0[0]],
                dof_id_map[tri0[1]],
                dof_id_map[tri0[2]],
                dof_id_map[tri1[0]],
                dof_id_map[tri1[1]],
                dof_id_map[tri1[2]]
            }, dgdcd2gdvdw_ad);
        
        d2ldq2.block(0, 4, 4, nv) += dgdc_plus_dgdx_dxdt_dtdc_tensor_product_d2cdvdw;
        d2ldq2.block(4, 0, nv, 4) += dgdc_plus_dgdx_dxdt_dtdc_tensor_product_d2cdvdw.transpose();
        
        if ((d2ldq2 - d2ldq2.transpose()).norm() > 1e-8)
        {
            // std::cout << "unsymmetric hessian" << std::endl;
            // std::exit(0);
        }
    }
}


// compute the gradient for one way coupling
// computing eqn (7)
// naming convention follows what's in the paper
void IntrinsicSimulation::computeGeodesicLengthGradient(const Edge& edge, Vector<T, 4>& dldw)
{
    dldw.setZero();
    int edge_idx = edge_map[edge];
    SurfacePoint vA = surface_points[edge[0]];
    SurfacePoint vB = surface_points[edge[1]];
    
    // current geodesic distance and path
    T l = current_length[edge_idx];
    std::vector<SurfacePoint> path = paths[edge_idx];
    
    int length = path.size();
    
    TV dldc0, dldc1;
    // the start and end point of a geodesic in world coordinates
    TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
    TV v1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));

    // mesh vertices for the triangle that these points lies in
    TV v10 = toTV(geometry->vertexPositions[vA.face.halfedge().vertex()]);
    TV v11 = toTV(geometry->vertexPositions[vA.face.halfedge().next().vertex()]);
    TV v12 = toTV(geometry->vertexPositions[vA.face.halfedge().next().next().vertex()]);

    TV v20 = toTV(geometry->vertexPositions[vB.face.halfedge().vertex()]);
    TV v21 = toTV(geometry->vertexPositions[vB.face.halfedge().next().vertex()]);
    TV v22 = toTV(geometry->vertexPositions[vB.face.halfedge().next().next().vertex()]);
    
    // analytial derivatives for length
    // d|xi - xj|/dxi = -(xj - xi).normalized()
    // d|xi - xj|/dxj = (xj - xi).normalized()

    if (length == 2 || Euclidean)
    {
        dldc0 = -(v1 - v0).normalized();   
        dldc1 = -dldc0;
    }
    else
    {
        TV ixn0 = toTV(path[1].interpolate(geometry->vertexPositions));
        dldc0 = -(ixn0 - v0).normalized();

        TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
        dldc1 = -(ixn1 - v1).normalized();
    }

    // barycentric Jabocian
    Vector<T, 6> dldc; dldc.setZero();
    dldc.segment<3>(0) = dldc0;
    dldc.segment<3>(3) = dldc1;

    T d_hat_dalpha0 = 1.0, d_hat_dbeta0 = 1.0;
    T d_hat_dalpha1 = 1.0, d_hat_dbeta1 = 1.0;
    if (use_t_wrapper)
    {
        d_hat_dalpha0 = wrapper_t<1>(vA.faceCoords[0], 1e-3);
        d_hat_dbeta0 = wrapper_t<1>(vA.faceCoords[1], 1e-3);
        d_hat_dalpha1 = wrapper_t<1>(vB.faceCoords[0], 1e-3);
        d_hat_dbeta1 = wrapper_t<1>(vB.faceCoords[1], 1e-3);
    }
    
    Matrix<T, 6, 4> dcdw; dcdw.setZero();
    dcdw.block(0, 0, 3, 1) = (v10 - v12) * d_hat_dalpha0;
    dcdw.block(0, 1, 3, 1) = (v11 - v12) * d_hat_dbeta0;

    dcdw.block(3, 2, 3, 1) = (v20 - v22) * d_hat_dalpha1;
    dcdw.block(3, 3, 3, 1) = (v21 - v22) * d_hat_dbeta1;

    // chain rule
    dldw = dldc.transpose() * dcdw;
}

// compute the Hessian for one way coupling
// computing eqn (11)
// naming convention follows what's in the paper
void IntrinsicSimulation::computeGeodesicLengthHessian(const Edge& edge, Matrix<T, 4, 4>& d2ldw2)
{
    int edge_idx = edge_map[edge];
    SurfacePoint vA = surface_points[edge[0]];
    SurfacePoint vB = surface_points[edge[1]];

    T l = current_length[edge_idx];
    std::vector<SurfacePoint> path = paths[edge_idx];
    
    int length = path.size();
    int nt = (length - 2);
    TV dldc0, dldc1;

    // the start and end point of a geodesic in world coordinates
    TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
    TV v1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));

    // mesh vertices for the triangle that these points lies in

    TV v10 = toTV(geometry->vertexPositions[vA.face.halfedge().vertex()]);
    TV v11 = toTV(geometry->vertexPositions[vA.face.halfedge().next().vertex()]);
    TV v12 = toTV(geometry->vertexPositions[vA.face.halfedge().next().next().vertex()]);

    TV v20 = toTV(geometry->vertexPositions[vB.face.halfedge().vertex()]);
    TV v21 = toTV(geometry->vertexPositions[vB.face.halfedge().next().vertex()]);
    TV v22 = toTV(geometry->vertexPositions[vB.face.halfedge().next().next().vertex()]);

    
    T d_hat_dalpha0 = 1.0, d_hat_dbeta0 = 1.0;
    T d_hat_dalpha1 = 1.0, d_hat_dbeta1 = 1.0;
    if (use_t_wrapper)
    {
        d_hat_dalpha0 = wrapper_t<1>(vA.faceCoords[0], 1e-3);
        d_hat_dbeta0 = wrapper_t<1>(vA.faceCoords[1], 1e-3);
        d_hat_dalpha1 = wrapper_t<1>(vB.faceCoords[0], 1e-3);
        d_hat_dbeta1 = wrapper_t<1>(vB.faceCoords[1], 1e-3);
    }

    // barycentric Jacobian
    Matrix<T, 6, 4> dcdw; dcdw.setZero();
    dcdw.block(0, 0, 3, 1) = (v10 - v12) * d_hat_dalpha0;
    dcdw.block(0, 1, 3, 1) = (v11 - v12) * d_hat_dbeta0;

    dcdw.block(3, 2, 3, 1) = (v20 - v22) * d_hat_dalpha1;
    dcdw.block(3, 3, 3, 1) = (v21 - v22) * d_hat_dbeta1;

    Matrix<T, 3, 2> dx0dw0;
    dx0dw0.col(0) = (v10 - v12) * d_hat_dalpha0;
    dx0dw0.col(1) = (v11 - v12) * d_hat_dbeta0;

    Matrix<T, 3, 2> dx1dw1;
    dx1dw1.col(0) = (v20 - v22) * d_hat_dalpha1;
    dx1dw1.col(1) = (v21 - v22) * d_hat_dbeta1;

    TV2 dldw0 = dldc0.transpose() * dx0dw0;
    TV2 dldw1 = dldc1.transpose() * dx1dw1;
    
    Vector<T, 6> dldc; dldc.setZero();
    
    Matrix<T, 6, 6> d2ldx2; d2ldx2.setZero();

    if (length == 2 || Euclidean)
    {
        dldc0 = -(v1 - v0).normalized();
        dldc1 = -dldc0;
        dldc.segment<3>(0) = dldc0;
        dldc.segment<3>(3) = dldc1;
        d2ldx2.block(0, 0, 3, 3) = (TM::Identity() - dldc0 * dldc0.transpose()) / l;
        d2ldx2.block(3, 3, 3, 3) = d2ldx2.block(0, 0, 3, 3);
        d2ldx2.block(3, 0, 3, 3) = -d2ldx2.block(0, 0, 3, 3);
        d2ldx2.block(0, 3, 3, 3) = -d2ldx2.block(0, 0, 3, 3);
    }
    else
    {

        // analytial gradient for length
        // d|xi - xj|/dxi = -(xj - xi).normalized()
        // d|xi - xj|/dxj = (xj - xi).normalized()
        TV ixn0 = toTV(path[1].interpolate(geometry->vertexPositions));
        dldc0 = -(ixn0 - v0).normalized();

        TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
        dldc1 = -(ixn1 - v1).normalized();

        dldc.segment<3>(0) = dldc0;
        dldc.segment<3>(3) = dldc1;
        
        int ixn_dof = (length - 2) * 3;
        MatrixXT dfdc(ixn_dof, 6); dfdc.setZero();
        MatrixXT dfdx(ixn_dof, ixn_dof); dfdx.setZero();
        MatrixXT dxdt(ixn_dof, length-2); dxdt.setZero();
        MatrixXT d2gdcdx(ixn_dof, 6); d2gdcdx.setZero();

        TM dl0dx0 = (TM::Identity() - dldc0 * dldc0.transpose()) / (ixn0 - v0).norm();

        dfdx.block(0, 0, 3, 3) += dl0dx0;
        dfdc.block(0, 0, 3, 3) += -dl0dx0;
        d2gdcdx.block(0, 0, 3, 3) += -dl0dx0;

        // analytial hessian for length
        // d^2|xi - xj|/dxi^2 
        for (int ixn_id = 0; ixn_id < length - 3; ixn_id++)
        {
            Matrix<T, 6, 6> hess;
            TV ixn_i = toTV(path[1 + ixn_id].interpolate(geometry->vertexPositions));
            TV ixn_j = toTV(path[1 + ixn_id+1].interpolate(geometry->vertexPositions));

            edgeLengthHessian(ixn_i, ixn_j, hess);
            dfdx.block(ixn_id*3, ixn_id * 3, 6, 6) += hess;
        }
        // compute dx/dt for the intersection points
        for (int ixn_id = 0; ixn_id < length - 2; ixn_id++)
        {
            T t = ixn_data_list[edge_idx][1+ixn_id].t;
            T d_hat_dt = 1.0;
            if (use_t_wrapper)
            {
                d_hat_dt = wrapper_t<1>(t);
            }

            TV x_start = ixn_data_list[edge_idx][1+ixn_id].start;
            TV x_end = ixn_data_list[edge_idx][1+ixn_id].end;
            dxdt.block(ixn_id * 3, ixn_id, 3, 1) = (x_end - x_start) * d_hat_dt;
        }
        
        TM dlndxn = (TM::Identity() - dldc1 * dldc1.transpose()) / (ixn1 - v1).norm();
        dfdx.block(ixn_dof-3, ixn_dof-3, 3, 3) += dlndxn;
        dfdc.block(ixn_dof-3, 3, 3, 3) += -dlndxn;
        d2gdcdx.block(ixn_dof-3, 3, 3, 3) += -dlndxn;

        // sensitivity analysis for computing dt/dc
        // eqn (9) in paper
        MatrixXT dxdtd2gdxdc = dxdt.transpose() * d2gdcdx;
        MatrixXT dxdtd2gdx2dxdt = dxdt.transpose() * dfdx * dxdt;
        MatrixXT dtdc(nt, 6);
        
        if (dxdt.norm() < 1e-8)
            dtdc.setZero();
        else
            dtdc = dxdtd2gdx2dxdt.colPivHouseholderQr().solve(-dxdtd2gdxdc);


        Matrix<T, 6, 6> d2gdc2; d2gdc2.setZero();
        d2gdc2.block(0, 0, 3, 3) += dl0dx0;
        d2gdc2.block(3, 3, 3, 3) += dlndxn;

        // Eqn (11)
        d2ldx2 = d2gdc2 + d2gdcdx.transpose() * dxdt * dtdc;

        Matrix<T, 6, 6> d2ldx2_approx; d2ldx2_approx.setZero();
        d2ldx2_approx.block(0, 0, 3, 3) = (TM::Identity() - dldc0 * dldc0.transpose()) / l;
        d2ldx2_approx.block(3, 3, 3, 3) = (TM::Identity() - dldc1 * dldc1.transpose()) / l;
        d2ldx2_approx.block(0, 3, 3, 3) = -0.5 * (d2ldx2_approx.block(0, 0, 3, 3) + d2ldx2_approx.block(3, 3, 3, 3));
        d2ldx2_approx.block(3, 0, 3, 3) = -0.5 * (d2ldx2_approx.block(0, 0, 3, 3) + d2ldx2_approx.block(3, 3, 3, 3));

        // if intersections are too close to the mass points,
        // then there could appear super large value in the element hessian
        // we use an approximated hessian in this case
        // it's fine to comment it out since it happens rarely
        if ((dfdx.maxCoeff() > 1e6 || dfdx.minCoeff() < -1e6 || dxdt.norm() < 1e-8) && !run_diff_test)
            d2ldx2 = d2ldx2_approx;
        
    }
    // chain rule
    d2ldw2 = dcdw.transpose() * d2ldx2 * dcdw;
}

void IntrinsicSimulation::computeGeodesicLengthGradientAndHessian(const Edge& edge, 
    Vector<T, 4>& dldw, Matrix<T, 4, 4>& d2ldw2)
{
    int edge_idx = edge_map[edge];
    SurfacePoint vA = surface_points[edge[0]];
    SurfacePoint vB = surface_points[edge[1]];
    
    T l = current_length[edge_idx];
    std::vector<SurfacePoint> path = paths[edge_idx];
    
    int length = path.size();
    // std::cout << "length " << length << std::endl;
    
    int nt = (length - 2);
    TV dldc0, dldc1;
    TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
    TV v1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));

    TV v10 = toTV(geometry->vertexPositions[vA.face.halfedge().vertex()]);
    TV v11 = toTV(geometry->vertexPositions[vA.face.halfedge().next().vertex()]);
    TV v12 = toTV(geometry->vertexPositions[vA.face.halfedge().next().next().vertex()]);

    TV v20 = toTV(geometry->vertexPositions[vB.face.halfedge().vertex()]);
    TV v21 = toTV(geometry->vertexPositions[vB.face.halfedge().next().vertex()]);
    TV v22 = toTV(geometry->vertexPositions[vB.face.halfedge().next().next().vertex()]);

    T d_hat_dalpha0 = 1.0, d_hat_dbeta0 = 1.0;
    T d_hat_dalpha1 = 1.0, d_hat_dbeta1 = 1.0;
    
    // apply mollifier to the geodesic endpoints
    if (use_t_wrapper)
    {
        d_hat_dalpha0 = wrapper_t<1>(vA.faceCoords[0], 1e-6);
        d_hat_dbeta0 = wrapper_t<1>(vA.faceCoords[1], 1e-6);
        d_hat_dalpha1 = wrapper_t<1>(vB.faceCoords[0], 1e-6);
        d_hat_dbeta1 = wrapper_t<1>(vB.faceCoords[1], 1e-6);
    }

    Matrix<T, 3, 2> dx0dw0;
    dx0dw0.col(0) = (v10 - v12) * d_hat_dalpha0;
    dx0dw0.col(1) = (v11 - v12) * d_hat_dbeta0;

    Matrix<T, 3, 2> dx1dw1;
    dx1dw1.col(0) = (v20 - v22) * d_hat_dalpha1;
    dx1dw1.col(1) = (v21 - v22) * d_hat_dbeta1;

    TV2 dldw0 = dldc0.transpose() * dx0dw0;
    TV2 dldw1 = dldc1.transpose() * dx1dw1;

    Matrix<T, 6, 4> dcdw; dcdw.setZero();
    dcdw.block(0, 0, 3, 1) = (v10 - v12) * d_hat_dalpha0;
    dcdw.block(0, 1, 3, 1) = (v11 - v12) * d_hat_dbeta0;

    dcdw.block(3, 2, 3, 1) = (v20 - v22) * d_hat_dalpha1;
    dcdw.block(3, 3, 3, 1) = (v21 - v22) * d_hat_dbeta1;

    
    Vector<T, 6> dldc; dldc.setZero();
    
    Matrix<T, 6, 6> d2ldx2; d2ldx2.setZero();

    if (length == 2 || Euclidean)
    {
        dldc0 = -(v1 - v0).normalized();
        dldc1 = -dldc0;
        dldc.segment<3>(0) = dldc0;
        dldc.segment<3>(3) = dldc1;
        d2ldx2.block(0, 0, 3, 3) = (TM::Identity() - dldc0 * dldc0.transpose()) / l;
        d2ldx2.block(3, 3, 3, 3) = d2ldx2.block(0, 0, 3, 3);
        d2ldx2.block(3, 0, 3, 3) = -d2ldx2.block(0, 0, 3, 3);
        d2ldx2.block(0, 3, 3, 3) = -d2ldx2.block(0, 0, 3, 3);
    }
    else
    {
        
        TV ixn0 = toTV(path[1].interpolate(geometry->vertexPositions));
        dldc0 = -(ixn0 - v0).normalized();

        TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
        dldc1 = -(ixn1 - v1).normalized();

        dldc.segment<3>(0) = dldc0;
        dldc.segment<3>(3) = dldc1;
        
        int ixn_dof = (length - 2) * 3;
        MatrixXT dfdc(ixn_dof, 6); dfdc.setZero();
        VectorXT dgdx(ixn_dof); dgdx.setZero();
        MatrixXT dfdx(ixn_dof, ixn_dof); dfdx.setZero();
        MatrixXT dxdt(ixn_dof, length-2); dxdt.setZero();
        MatrixXT d2gdcdx(ixn_dof, 6); d2gdcdx.setZero();

        TM dl0dx0 = (TM::Identity() - dldc0 * dldc0.transpose()) / (ixn0 - v0).norm();

        dfdx.block(0, 0, 3, 3) += dl0dx0;
        dfdc.block(0, 0, 3, 3) += -dl0dx0;
        d2gdcdx.block(0, 0, 3, 3) += -dl0dx0;

        dgdx.segment<3>(0) += (ixn0 - v0).normalized();
        dgdx.segment<3>(ixn_dof - 3) += -(v1 - ixn1).normalized();
        for (int ixn_id = 0; ixn_id < length - 3; ixn_id++)
        {
            Matrix<T, 6, 6> hess;
            TV ixn_i = toTV(path[1 + ixn_id].interpolate(geometry->vertexPositions));
            TV ixn_j = toTV(path[1 + ixn_id+1].interpolate(geometry->vertexPositions));

            edgeLengthHessian(ixn_i, ixn_j, hess);
            dfdx.block(ixn_id*3, ixn_id * 3, 6, 6) += hess;
            dgdx.segment<3>(ixn_id * 3) += -(ixn_j - ixn_i).normalized();
            dgdx.segment<3>((ixn_id + 1) * 3) += (ixn_j - ixn_i).normalized();
        }
        for (int ixn_id = 0; ixn_id < length - 2; ixn_id++)
        {
            T t = ixn_data_list[edge_idx][1+ixn_id].t;
            T d_hat_dt = 1.0;
            if (use_t_wrapper)
            {
                d_hat_dt = wrapper_t<1>(t);
            }
            TV x_start = ixn_data_list[edge_idx][1+ixn_id].start;
            TV x_end = ixn_data_list[edge_idx][1+ixn_id].end;
            dxdt.block(ixn_id * 3, ixn_id, 3, 1) = (x_end - x_start) * d_hat_dt;
        }
        
        
        TM dlndxn = (TM::Identity() - dldc1 * dldc1.transpose()) / (ixn1 - v1).norm();
        dfdx.block(ixn_dof-3, ixn_dof-3, 3, 3) += dlndxn;
        dfdc.block(ixn_dof-3, 3, 3, 3) += -dlndxn;
        d2gdcdx.block(ixn_dof-3, 3, 3, 3) += -dlndxn;

        MatrixXT dxdtd2gdxdc = dxdt.transpose() * d2gdcdx;
        MatrixXT dxdtd2gdx2dxdt = dxdt.transpose() * dfdx * dxdt;
        MatrixXT dtdc(nt, 6);
        if (dxdt.norm() < 1e-8)
            dtdc.setZero();
        else
            dtdc = dxdtd2gdx2dxdt.colPivHouseholderQr().solve(-dxdtd2gdxdc);
        
        //
        // these are the terms corresponding to the implicit function theorem 
        // in case you'd like to check if this is indeed zero
        //

        // std::cout << "dgdx" << std::endl;
        // std::cout << dgdx << std::endl;
        // std::cout << "dxdt" << std::endl;
        // std::cout << dxdt << std::endl;
        // std::cout << "dgdx * dxdt" << std::endl;
        // std::cout << dgdx * dxdt << std::endl;
        // std::cout << "dgdx * dxdt * dtdc * dcdw" << std::endl;
        // std::cout << dgdx * dxdt * dtdc * dcdw << std::endl;
        // std::getchar();

        Matrix<T, 6, 6> d2gdc2; d2gdc2.setZero();
        d2gdc2.block(0, 0, 3, 3) += dl0dx0;
        d2gdc2.block(3, 3, 3, 3) += dlndxn;

        d2ldx2 = d2gdc2 + d2gdcdx.transpose() * dxdt * dtdc;

        Matrix<T, 6, 6> d2ldx2_approx; d2ldx2_approx.setZero();
        d2ldx2_approx.block(0, 0, 3, 3) = (TM::Identity() - dldc0 * dldc0.transpose()) / l;
        d2ldx2_approx.block(3, 3, 3, 3) = (TM::Identity() - dldc1 * dldc1.transpose()) / l;
        d2ldx2_approx.block(0, 3, 3, 3) = -0.5 * (d2ldx2_approx.block(0, 0, 3, 3) + d2ldx2_approx.block(3, 3, 3, 3));
        d2ldx2_approx.block(3, 0, 3, 3) = -0.5 * (d2ldx2_approx.block(0, 0, 3, 3) + d2ldx2_approx.block(3, 3, 3, 3));

        // if intersections are too close to the mass points,
        // then there could appear super large value in the element hessian
        // we use an approximated hessian in this case
        if ((dfdx.maxCoeff() > 1e6 || dfdx.minCoeff() < -1e6 || dxdt.norm() < 1e-8) && !run_diff_test)
            d2ldx2 = d2ldx2_approx;
    }
    dldw = dldc.transpose() * dcdw;
    d2ldw2 = dcdw.transpose() * d2ldx2 * dcdw;
}