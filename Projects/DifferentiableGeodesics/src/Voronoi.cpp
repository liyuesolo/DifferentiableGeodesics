#include <mutex>
#include <Eigen/CholmodSupport>
#include <igl/readOBJ.h>
#include <igl/edges.h>
#include <igl/per_vertex_normals.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/facet_adjacency_matrix.h>
#include <igl/adjacency_matrix.h>
#include <igl/per_face_normals.h>
#include "../include/VoronoiCells.h"

#include "geometrycentral/surface/edge_length_geometry.h"
#include "geometrycentral/surface/flip_geodesics.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/mesh_graph_algorithms.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "geometrycentral/surface/polygon_soup_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/utilities/timing.h"
#include "geometrycentral/surface/exact_geodesics.h"

#include "../include/Util.h"

#define PARALLEL

void VoronoiCells::computeGeodesicLengthGradientAndHessian(const Edge& edge, int edge_idx,
    Vector<T, 6>& dldw, Matrix<T, 6, 6>& d2ldw2)
{
    
    SurfacePoint vA = unique_ixn_points[edge[0]].first;
    SurfacePoint vB = unique_ixn_points[edge[1]].first;
    
    T l = current_length[edge_idx];
    std::vector<SurfacePoint> path = paths[edge_idx];
    
    int length = path.size();
    int nt = (length - 2);
    TV dldx0, dldx1;
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
    if (use_t_wrapper)
    {
        d_hat_dalpha0 = wrapper_t<1>(vA.faceCoords[0], 1e-3);
        d_hat_dbeta0 = wrapper_t<1>(vA.faceCoords[1], 1e-3);
        d_hat_dalpha1 = wrapper_t<1>(vB.faceCoords[0], 1e-3);
        d_hat_dbeta1 = wrapper_t<1>(vB.faceCoords[1], 1e-3);
    }

    Matrix<T, 3, 2> dx0dw0;
    dx0dw0.col(0) = (v10 - v12) * d_hat_dalpha0;
    dx0dw0.col(1) = (v11 - v12) * d_hat_dbeta0;

    Matrix<T, 3, 2> dx1dw1;
    dx1dw1.col(0) = (v20 - v22) * d_hat_dalpha1;
    dx1dw1.col(1) = (v21 - v22) * d_hat_dbeta1;

    TV2 dldw0 = dldx0.transpose() * dx0dw0;
    TV2 dldw1 = dldx1.transpose() * dx1dw1;

    Matrix<T, 6, 4> dxdw; dxdw.setZero();
    dxdw.block(0, 0, 3, 1) = (v10 - v12) * d_hat_dalpha0;
    dxdw.block(0, 1, 3, 1) = (v11 - v12) * d_hat_dbeta0;

    dxdw.block(3, 2, 3, 1) = (v20 - v22) * d_hat_dalpha1;
    dxdw.block(3, 3, 3, 1) = (v21 - v22) * d_hat_dbeta1;

    
    Vector<T, 6> dldx; dldx.setZero();
    
    Matrix<T, 6, 6> d2ldx2; d2ldx2.setZero();

    if (length == 2)
    {
        dldx0 = -(v1 - v0).normalized();
        dldx1 = -dldx0;
        dldx.segment<3>(0) = dldx0;
        dldx.segment<3>(3) = dldx1;
        d2ldx2.block(0, 0, 3, 3) = (TM::Identity() - dldx0 * dldx0.transpose()) / l;
        d2ldx2.block(3, 3, 3, 3) = d2ldx2.block(0, 0, 3, 3);
        d2ldx2.block(3, 0, 3, 3) = -d2ldx2.block(0, 0, 3, 3);
        d2ldx2.block(0, 3, 3, 3) = -d2ldx2.block(0, 0, 3, 3);
    }
    else
    {
        
        TV ixn0 = toTV(path[1].interpolate(geometry->vertexPositions));
        dldx0 = -(ixn0 - v0).normalized();

        TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
        dldx1 = -(ixn1 - v1).normalized();

        dldx.segment<3>(0) = dldx0;
        dldx.segment<3>(3) = dldx1;
        
        int ixn_dof = (length - 2) * 3;
        MatrixXT dfdc(ixn_dof, 6); dfdc.setZero();
        MatrixXT dfdx(ixn_dof, ixn_dof); dfdx.setZero();
        MatrixXT dxdt(ixn_dof, length-2); dxdt.setZero();
        MatrixXT d2gdcdx(ixn_dof, 6); d2gdcdx.setZero();

        TM dl0dx0 = (TM::Identity() - dldx0 * dldx0.transpose()) / (ixn0 - v0).norm();

        dfdx.block(0, 0, 3, 3) += dl0dx0;
        dfdc.block(0, 0, 3, 3) += -dl0dx0;
        d2gdcdx.block(0, 0, 3, 3) += -dl0dx0;
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
        
        TM dlndxn = (TM::Identity() - dldx1 * dldx1.transpose()) / (ixn1 - v1).norm();
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
        // MatrixXT dtdc = dxdtd2gdx2dxdt.colPivHouseholderQr().solve(-dxdtd2gdxdc);


        Matrix<T, 6, 6> d2gdc2; d2gdc2.setZero();
        d2gdc2.block(0, 0, 3, 3) += dl0dx0;
        d2gdc2.block(3, 3, 3, 3) += dlndxn;

        d2ldx2 = d2gdc2 + d2gdcdx.transpose() * dxdt * dtdc;

        Matrix<T, 6, 6> d2ldx2_approx; d2ldx2_approx.setZero();
        d2ldx2_approx.block(0, 0, 3, 3) = (TM::Identity() - dldx0 * dldx0.transpose()) / l;
        d2ldx2_approx.block(3, 3, 3, 3) = (TM::Identity() - dldx1 * dldx1.transpose()) / l;
        d2ldx2_approx.block(0, 3, 3, 3) = -0.5 * (d2ldx2_approx.block(0, 0, 3, 3) + d2ldx2_approx.block(3, 3, 3, 3));
        d2ldx2_approx.block(3, 0, 3, 3) = -0.5 * (d2ldx2_approx.block(0, 0, 3, 3) + d2ldx2_approx.block(3, 3, 3, 3));

        // if intersections are too close to the mass points,
        // then there could appear super large value in the element hessian
        // we use an approximated hessian in this case
        if ((dfdx.maxCoeff() > 1e6 || dfdx.minCoeff() < -1e6 || dxdt.norm() < 1e-8) && !run_diff_test)
            d2ldx2 = d2ldx2_approx;
    }
    dldw = dldx;
    d2ldw2 = d2ldx2;
}

void VoronoiCells::computeGeodesicLengthGradient(const Edge& edge, int edge_idx, Vector<T, 6>& dldw)
{
    dldw.setZero();
    
    SurfacePoint vA = unique_ixn_points[edge[0]].first.inSomeFace();
    SurfacePoint vB = unique_ixn_points[edge[1]].first.inSomeFace();

    // std::cout << "va barycentric " << vA.faceCoords << std::endl;
    // std::cout << "vb barycentric " << vB.faceCoords << std::endl;
    
    T l = current_length[edge_idx];
    std::vector<SurfacePoint> path = paths[edge_idx];
    
    int length = path.size();
    
    TV dldx0, dldx1;
    TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
    TV v1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));

    TV v10 = toTV(geometry->vertexPositions[vA.face.halfedge().vertex()]);
    TV v11 = toTV(geometry->vertexPositions[vA.face.halfedge().next().vertex()]);
    TV v12 = toTV(geometry->vertexPositions[vA.face.halfedge().next().next().vertex()]);

    TV v20 = toTV(geometry->vertexPositions[vB.face.halfedge().vertex()]);
    TV v21 = toTV(geometry->vertexPositions[vB.face.halfedge().next().vertex()]);
    TV v22 = toTV(geometry->vertexPositions[vB.face.halfedge().next().next().vertex()]);
    
    if (length == 2)
    {
        dldx0 = -(v1 - v0).normalized();   
        dldx1 = -dldx0;
    }
    else
    {
        TV ixn0 = toTV(path[1].interpolate(geometry->vertexPositions));
        dldx0 = -(ixn0 - v0).normalized();

        TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
        dldx1 = -(ixn1 - v1).normalized();
    }

    Vector<T, 6> dldx; dldx.setZero();
    dldx.segment<3>(0) = dldx0;
    dldx.segment<3>(3) = dldx1;


    T d_hat_dalpha0 = 1.0, d_hat_dbeta0 = 1.0;
    T d_hat_dalpha1 = 1.0, d_hat_dbeta1 = 1.0;
    if (use_t_wrapper)
    {
        d_hat_dalpha0 = wrapper_t<1>(vA.faceCoords[0], 1e-3);
        d_hat_dbeta0 = wrapper_t<1>(vA.faceCoords[1], 1e-3);
        d_hat_dalpha1 = wrapper_t<1>(vB.faceCoords[0], 1e-3);
        d_hat_dbeta1 = wrapper_t<1>(vB.faceCoords[1], 1e-3);
    }

    Matrix<T, 6, 4> dxdw; dxdw.setZero();
    dxdw.block(0, 0, 3, 1) = (v10 - v12) * d_hat_dalpha0;
    dxdw.block(0, 1, 3, 1) = (v11 - v12) * d_hat_dbeta0;

    dxdw.block(3, 2, 3, 1) = (v20 - v22) * d_hat_dalpha1;
    dxdw.block(3, 3, 3, 1) = (v21 - v22) * d_hat_dbeta1;

    dldw = dldx;
}

void VoronoiCells::computeExactGeodesic(const SurfacePoint& va, const SurfacePoint& vb, 
        T& dis, std::vector<SurfacePoint>& path, 
        std::vector<IxnData>& ixn_data, bool trace_path)
{
    // if (Euclidean)
    // {
    //     TV xa = toTV(va.interpolate(geometry->vertexPositions));
    //     TV xb = toTV(vb.interpolate(geometry->vertexPositions));
    //     path.resize(2); path[0] = va; path[1] = vb;
    //     dis = (xb - xa).norm();
    //     return;
    // }
    ixn_data.clear();
    int n_tri = extrinsic_indices.rows() / 3;
    std::vector<std::vector<size_t>> mesh_indices_gc(n_tri, std::vector<size_t>(3));
    for (int i = 0; i < n_tri; i++)
        for (int d = 0; d < 3; d++)
            mesh_indices_gc[i][d] = extrinsic_indices[i * 3 + d];
    int n_vtx_extrinsic = extrinsic_vertices.rows() / 3;
    std::vector<gc::Vector3> mesh_vertices_gc(n_vtx_extrinsic);
    
    for (int i = 0; i < n_vtx_extrinsic; i++)
        mesh_vertices_gc[i] = gc::Vector3{extrinsic_vertices(i * 3 + 0), 
            extrinsic_vertices(i * 3 + 1), extrinsic_vertices(i * 3 + 2)};


    // START_TIMING(contruct)
    auto lvals = gcs::makeManifoldSurfaceMeshAndGeometry(mesh_indices_gc, mesh_vertices_gc);
    // FINISH_TIMING_PRINT(contruct)
    std::unique_ptr<gcs::ManifoldSurfaceMesh> sub_mesh;
    std::unique_ptr<gcs::VertexPositionGeometry> sub_geometry;
    std::tie(sub_mesh, sub_geometry) = std::tuple<std::unique_ptr<gcs::ManifoldSurfaceMesh>,
                    std::unique_ptr<gcs::VertexPositionGeometry>>(std::move(std::get<0>(lvals)),  // mesh
                                                             std::move(std::get<1>(lvals))); // geometry
    
    gcs::GeodesicAlgorithmExact mmp(*sub_mesh, *sub_geometry);

    SurfacePoint va_sub(sub_mesh->face(va.face.getIndex()), va.faceCoords);
    SurfacePoint vb_sub(sub_mesh->face(vb.face.getIndex()), vb.faceCoords);


    mmp.propagate(va_sub);
    if (trace_path)
    {
        path = mmp.traceBack(vb_sub, dis);
        std::reverse(path.begin(), path.end());
        
        for (auto& pt : path)
        {
            T edge_t = -1.0; TV start = TV::Zero(), end = TV::Zero();
            bool is_edge_point = (pt.type == gcs::SurfacePointType::Edge);
            bool is_vtx_point = (pt.type == gcs::SurfacePointType::Vertex);
            Edge start_end; start_end.setConstant(-1);
            if (is_edge_point)
            {
                auto he = pt.edge.halfedge();
                SurfacePoint start_extrinsic = he.tailVertex();
                SurfacePoint end_extrinsic = he.tipVertex();
                start = toTV(start_extrinsic.interpolate(sub_geometry->vertexPositions));
                end = toTV(end_extrinsic.interpolate(sub_geometry->vertexPositions));
                start_end[0] = start_extrinsic.vertex.getIndex();
                start_end[1] = end_extrinsic.vertex.getIndex();
                TV ixn = toTV(pt.interpolate(sub_geometry->vertexPositions));
                TV test_interp = pt.tEdge * start + (1.0 - pt.tEdge) * end;
                if ((test_interp - ixn).norm() > 1e-6)
                {
                    std::swap(start, end);
                    std::swap(start_end[0], start_end[1]);
                }
                
                TV dir0 = (end-start).normalized();
                TV dir1 = (ixn-start).normalized();
                if ((dir0.cross(dir1)).norm() > 1e-6)
                {
                    std::cout << "error in cross product" << std::endl;
                    std::exit(0);
                }
                test_interp = pt.tEdge * start + (1.0 - pt.tEdge) * end;
                if ((ixn - test_interp).norm() > 1e-6)
                {
                    std::cout << "error in interpolation" << std::endl;
                    std::exit(0);
                }
                edge_t = pt.tEdge;
                // std::cout << pt.tEdge << " " << " cross " << (dir0.cross(dir1)).norm() << std::endl;
                // std::cout << ixn.transpose() << " " << (pt.tEdge * start + (1.0-pt.tEdge) * end).transpose() << std::endl;
                // std::getchar();
            }            
            else if (is_vtx_point)
            {
                // std::cout << "is vertex point" << std::endl;
                
                auto he = pt.vertex.halfedge();
                SurfacePoint start_extrinsic = he.tailVertex();
                SurfacePoint end_extrinsic = he.tipVertex();
                start = toTV(start_extrinsic.interpolate(sub_geometry->vertexPositions));
                end = toTV(end_extrinsic.interpolate(sub_geometry->vertexPositions));
                // std::cout << start.transpose() << " " << end.transpose() << std::endl;
                start_end[0] = start_extrinsic.vertex.getIndex();
                start_end[1] = end_extrinsic.vertex.getIndex();
                // std::cout << start_end[0] << " " << start_end[1] << std::endl;
                TV ixn = toTV(pt.interpolate(sub_geometry->vertexPositions));
                // std::cout << ixn.transpose() << std::endl;
                TV test_interp = start;
                if ((test_interp - ixn).norm() > 1e-6)
                {
                    std::swap(start, end);
                    std::swap(start_end[0], start_end[1]);
                }
                TV dir0 = (end-start).normalized();
                TV dir1 = (ixn-start).normalized();
                if ((dir0.cross(dir1)).norm() > 1e-6)
                {
                    std::cout << "error in cross product" << std::endl;
                    std::exit(0);
                }
                edge_t = 1.0;
            }
            else
            {
                edge_t = 2.0;
            }
            
            ixn_data.push_back(IxnData(start, end, (1.0-edge_t), start_end[0], start_end[1]));
            pt.edge = mesh->edge(pt.edge.getIndex());
            pt.vertex = mesh->vertex(pt.vertex.getIndex());
            pt.face = mesh->face(pt.face.getIndex());
            
            pt = pt.inSomeFace();
        }
        TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
        TV v1 = toTV(path[path.size() - 1].interpolate(geometry->vertexPositions));
        TV ixn0 = toTV(path[1].interpolate(geometry->vertexPositions));
        TV ixn1 = toTV(path[path.size() - 2].interpolate(geometry->vertexPositions));
        
        // if (path.size() > 2)
        // {
        //     if ((v0 - ixn0).norm() < 1e-6)
        //     {
        //         path.erase(path.begin() + 1);
                
        //     }
        // if (path.size() > 2)
        //     if ((v1 - ixn1).norm() < 1e-6)
        //     {
        //         path.erase(path.end() - 2);
        //     }
        // }
        
    }
    else
        dis = mmp.getDistance(vb_sub);
}

void VoronoiCells::generateMeshForRendering(MatrixXT& V, MatrixXi& F, MatrixXT& C)
{
    bool use_centroid = false;
    V.resize(0, 0); F.resize(0, 0); C.resize(0, 0);
    if (add_coplanar)
    {
        iterateVoronoiCells([&](const VoronoiCellData& cell_data, int cell_idx)
        {
            int n_cell_node = cell_data.cell_vtx_nodes.size();
            int n_vtx_current = V.rows();
            
            MatrixXT points(n_cell_node, 3);
            int cnt = 0;
            
            for (int idx : cell_data.cell_vtx_nodes)
            {
                auto ixn = unique_ixn_points[cell_data.cell_nodes[idx]];
                // std::cout << cell_data.cell_nodes[idx] << std::endl;
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

            T a = coeff[0], b = coeff[1], c = coeff[2];
		
            T denom = std::sqrt(a * a + b * b);
            
            T avg = 0.0;
            for (int i = 0; i < n_cell_node; i++)
            {
                T nom = std::abs(a * points(i, 0) + b * points(i, 1) + c - points(i, 2));
                avg += nom / denom;
            }
            avg / T(n_cell_node);

            TV face_centroid = TV::Zero();
            for (int i = 0; i < points.rows(); i++)
            {
                face_centroid += points.row(i);
            }
            face_centroid /= T(points.rows());
            
            int n_face_current = F.rows();
            if (use_centroid)
            {
                V.conservativeResize(n_vtx_current + n_cell_node + 1, 3);
                V.block(n_vtx_current, 0, n_cell_node, 3) = points;
                V.block(n_vtx_current + n_cell_node, 0, 1, 3) = face_centroid.transpose();
                F.conservativeResize(n_face_current + n_cell_node, 3);
                for (int j = 0; j < n_cell_node; j++)
                    F.row(n_face_current + j) = IV(j, n_cell_node, (j+1)%n_cell_node) + IV::Constant(n_vtx_current);
            }
            else
            {
                V.conservativeResize(n_vtx_current + n_cell_node, 3);
                V.block(n_vtx_current, 0, n_cell_node, 3) = points;
                if (n_cell_node == 3)
                {
                    F.conservativeResize(n_face_current + 1, 3);
                    F.row(n_face_current) = IV(0, 1, 2) + IV::Constant(n_vtx_current);
                }
                else if (n_cell_node == 4)
                {
                    F.conservativeResize(n_face_current + 2, 3);
                    F.row(n_face_current) = IV(0, 3, 1) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+1) = IV(1, 3, 2) + IV::Constant(n_vtx_current);
                }
                else if (n_cell_node == 5)
                {
                    F.conservativeResize(n_face_current + 3, 3);
                    F.row(n_face_current) = IV(0, 4, 1) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+1) = IV(1, 4, 3) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+2) = IV(1, 3, 2) + IV::Constant(n_vtx_current);
                }
                else if (n_cell_node == 6)
                {
                    F.conservativeResize(n_face_current + 4, 3);
                    F.row(n_face_current) = IV(0, 5, 1) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+1) = IV(1, 5, 4) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+2) = IV(1, 4, 2) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+3) = IV(2, 4, 3) + IV::Constant(n_vtx_current);
                }
                else if (n_cell_node == 7)
                {
                    F.conservativeResize(n_face_current + 5, 3);
                    F.row(n_face_current) = IV(3, 2, 4) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+1) = IV(2, 1, 4) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+2) = IV(1, 0, 4) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+3) = IV(0, 6, 4) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+4) = IV(4, 6, 5) + IV::Constant(n_vtx_current);
                }
                else if (n_cell_node == 8)
                {
                    F.conservativeResize(n_face_current + 6, 3);
                    F.row(n_face_current) = IV(0, 7, 1) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+1) = IV(1, 7, 6) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+2) = IV(1, 6, 2) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+3) = IV(2, 6, 5) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+4) = IV(2, 5, 3) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+5) = IV(3, 5, 4) + IV::Constant(n_vtx_current);
                }
                else if (n_cell_node == 9)
                {
                    F.conservativeResize(n_face_current + 7, 3);
                    F.row(n_face_current) = IV(1, 0, 2) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+1) = IV(2, 0, 3) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+2) = IV(3, 0, 8) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+3) = IV(3, 8, 7) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+4) = IV(3, 7, 4) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+5) = IV(4, 7, 6) + IV::Constant(n_vtx_current);
                    F.row(n_face_current+6) = IV(4, 6, 5) + IV::Constant(n_vtx_current);
                }
                else
                {
                    V.conservativeResize(n_vtx_current + n_cell_node + 1, 3);
                    V.block(n_vtx_current, 0, n_cell_node, 3) = points;
                    V.block(n_vtx_current + n_cell_node, 0, 1, 3) = face_centroid.transpose();
                    F.conservativeResize(n_face_current + n_cell_node, 3);
                    for (int j = 0; j < n_cell_node; j++)
                        F.row(n_face_current + j) = IV(j, n_cell_node, (j+1)%n_cell_node) + IV::Constant(n_vtx_current);
                }
                C.conservativeResize(F.rows(), 3);
                C.block(n_face_current, 0, F.rows() - n_face_current, 3) = TV(avg, 0., 0.).transpose();
                
            }
        });
        // MatrixXT N;
        // igl::per_face_normals(V, F, N);
        // C = N;
    }
    else
    {
        vectorToIGLMatrix<int, 3>(extrinsic_indices, F);
        int n_vtx_dof = extrinsic_vertices.rows();
        vectorToIGLMatrix<T, 3>(extrinsic_vertices, V);
        C.resize(F.rows(), 3);
        // C.col(0).setConstant(0.0); C.col(1).setConstant(0.3); C.col(2).setConstant(1.0);
        C.col(0).setConstant(28.0/255.0); C.col(1).setConstant(99.0/255.0); C.col(2).setConstant(227.0/255.0);
        if (use_debug_face_color)
            C = face_color;
    }
}

bool VoronoiCells::computeGeodesicDistanceEdgeFlip(const SurfacePoint& va, const SurfacePoint& vb,
        T& dis, std::vector<SurfacePoint>& path, 
        std::vector<IxnData>& ixn_data, bool trace_path)
{
    ixn_data.clear();
    path.clear();
    int n_tri = extrinsic_indices.rows() / 3;
    std::vector<std::vector<size_t>> mesh_indices_gc(n_tri, std::vector<size_t>(3));
    for (int i = 0; i < n_tri; i++)
        for (int d = 0; d < 3; d++)
            mesh_indices_gc[i][d] = extrinsic_indices[i * 3 + d];
    int n_vtx_extrinsic = extrinsic_vertices.rows() / 3;
    std::vector<gc::Vector3> mesh_vertices_gc(n_vtx_extrinsic);
    
    for (int i = 0; i < n_vtx_extrinsic; i++)
        mesh_vertices_gc[i] = gc::Vector3{extrinsic_vertices(i * 3 + 0), 
            extrinsic_vertices(i * 3 + 1), extrinsic_vertices(i * 3 + 2)};


    // START_TIMING(contruct)
    auto lvals = gcs::makeManifoldSurfaceMeshAndGeometry(mesh_indices_gc, mesh_vertices_gc);
    // FINISH_TIMING_PRINT(contruct)
    std::unique_ptr<gcs::ManifoldSurfaceMesh> sub_mesh;
    std::unique_ptr<gcs::VertexPositionGeometry> sub_geometry;
    std::tie(sub_mesh, sub_geometry) = std::tuple<std::unique_ptr<gcs::ManifoldSurfaceMesh>,
                    std::unique_ptr<gcs::VertexPositionGeometry>>(std::move(std::get<0>(lvals)),  // mesh
                                                             std::move(std::get<1>(lvals))); // geometry
                                                             
    std::unique_ptr<gcs::FlipEdgeNetwork> sub_edgeNetwork = std::unique_ptr<gcs::FlipEdgeNetwork>(
        new gcs::FlipEdgeNetwork(*sub_mesh, *sub_geometry, {}));
    
    // std::unique_ptr<gcs::ManifoldSurfaceMesh> sub_mesh = mesh->copy();
    // std::unique_ptr<gcs::VertexPositionGeometry> sub_geometry = geometry->copy();
    // std::unique_ptr<gcs::FlipEdgeNetwork> sub_edgeNetwork = std::unique_ptr<gcs::FlipEdgeNetwork>(
    //     new gcs::FlipEdgeNetwork(*sub_mesh, *sub_geometry, {}));
    
    // FINISH_TIMING_PRINT(contruct)

    // sub_edgeNetwork->tri->flipToDelaunay();
    // sub_edgeNetwork->posGeom = sub_geometry.get();
    
    SurfacePoint va_sub(sub_mesh->face(va.face.getIndex()), va.faceCoords);
    SurfacePoint vb_sub(sub_mesh->face(vb.face.getIndex()), vb.faceCoords);

    SurfacePoint va_intrinsic = sub_edgeNetwork->tri->equivalentPointOnIntrinsic(va_sub);
    gcVertex va_vtx = sub_edgeNetwork->tri->insertVertex(va_intrinsic);

    SurfacePoint vb_intrinsic = sub_edgeNetwork->tri->equivalentPointOnIntrinsic(vb_sub);
    gcVertex vb_vtx = sub_edgeNetwork->tri->insertVertex(vb_intrinsic);
        
    // START_TIMING(shorten)
    std::vector<gcs::Halfedge> path_geo = shortestEdgePath(*sub_edgeNetwork->tri, va_vtx, vb_vtx);
    sub_edgeNetwork->addPath(path_geo);
    sub_edgeNetwork->iterativeShorten();
    // FINISH_TIMING_PRINT(shorten)
    
    gcEdge ei = sub_edgeNetwork->tri->intrinsicMesh->connectingEdge(va_vtx, vb_vtx);
    if (ei == gcEdge())
    {
        return false;
    }
    dis = sub_edgeNetwork->tri->edgeLengths[ei];
    // START_TIMING(gather_info)
    sub_geometry->requireVertexPositions();
    
    if (trace_path)
    {
        gcs::Halfedge he = ei.halfedge();
        if (he.tailVertex() == vb_vtx)
            he = ei.halfedge().twin();
        
        path = sub_edgeNetwork->tri->traceIntrinsicHalfedgeAlongInput(he, false);
        // std::cout << " flip geodesic " << path.size() << std::endl;
        for (auto& pt : path)
        {
            
            TV bary = toTV(pt.faceCoords);
            
            bool is_face_point = (pt.type == gcs::SurfacePointType::Face);
            bool is_edge_point = (pt.type == gcs::SurfacePointType::Edge);
            bool is_vtx_point = (pt.type == gcs::SurfacePointType::Vertex);
            if (is_face_point) // double check
            {
                std::vector<gcs::Halfedge> half_edges = {pt.face.halfedge(), 
                                            pt.face.halfedge().next(), 
                                            pt.face.halfedge().next().next()};
            
                std::vector<int> zero_indices;
                for (int d = 0; d < 3; d++)
                {
                    if (std::abs(bary[d]) < 1e-8)
                        zero_indices.push_back(d);
                }
                if (zero_indices.size() == 1) // on an edge
                {
                    gcs::Halfedge he = half_edges[(zero_indices[0] + 1) % 3];
                    TV v10 = toTV(sub_geometry->vertexPositions[he.tailVertex()]);
                    TV v11 = toTV(sub_geometry->vertexPositions[he.tipVertex()]);
                    T t = (toTV(pt.interpolate(sub_geometry->vertexPositions)) - v10).norm() / 
                        (v11 - v10).norm();
                    pt = SurfacePoint(he, t);
                    
                }
                else if (zero_indices.size() == 2) // it's a vertex
                {
                    for(gcs::Vertex v : pt.face.adjacentVertices()) 
                    {
                        TV x_pt = toTV(pt.interpolate(sub_geometry->vertexPositions));
                        TV x_v = toTV(sub_geometry->vertexPositions[v]);
                        if ((x_pt - x_v).norm() < 1e-8)
                        {
                            pt = SurfacePoint(v);
                            break;
                        }
                    }
                }
            }
            
            is_edge_point = (pt.type == gcs::SurfacePointType::Edge);
            is_vtx_point = (pt.type == gcs::SurfacePointType::Vertex);

            // std::cout << toTV(pt.interpolate(sub_geometry->vertexPositions)).transpose() << std::endl;
            T edge_t = -1.0; TV start = TV::Zero(), end = TV::Zero();
            
            // std::cout << "is_edge_point " << is_edge_point << " is_vtx_point " << is_vtx_point << " is_face_point " << is_face_point << std::endl;
            Edge start_end; start_end.setConstant(-1);
            if (is_edge_point)
            {
                auto he = pt.edge.halfedge();
                SurfacePoint start_extrinsic = sub_mesh->vertex(he.tailVertex().getIndex());
                SurfacePoint end_extrinsic = sub_mesh->vertex(he.tipVertex().getIndex());
                
                
                // std::getchar();
                start = toTV(start_extrinsic.interpolate(sub_geometry->vertexPositions));
                end = toTV(end_extrinsic.interpolate(sub_geometry->vertexPositions));
                // start = toTV(sub_geometry->vertexPositions[sub_mesh->vertex(he.tailVertex().getIndex())]);
                // end = toTV(sub_geometry->vertexPositions[sub_mesh->vertex(he.tipVertex().getIndex())]);
                
                start_end[0] = start_extrinsic.vertex.getIndex();
                start_end[1] = end_extrinsic.vertex.getIndex();
                TV ixn = toTV(pt.interpolate(sub_geometry->vertexPositions));
                
                TV test_interp = pt.tEdge * start + (1.0 - pt.tEdge) * end;
                if ((test_interp - ixn).norm() > 1e-6)
                {
                    std::swap(start, end);
                    std::swap(start_end[0], start_end[1]);
                }
                // std::cout << "pass" <<std::endl;
                TV dir0 = (end-start).normalized();
                TV dir1 = (ixn-start).normalized();
                if ((dir0.cross(dir1)).norm() > 1e-6)
                {
                    std::cout << "error in cross product" << std::endl;
                    return false;
                }
                test_interp = pt.tEdge * start + (1.0 - pt.tEdge) * end;
                if ((ixn - test_interp).norm() > 1e-6)
                {
                    std::cout << "error in interpolation" << std::endl;
                    return false;
                }
                edge_t = pt.tEdge;
                // std::cout << pt.tEdge << " " << " cross " << (dir0.cross(dir1)).norm() << std::endl;
                // std::cout << ixn.transpose() << " " << (pt.tEdge * start + (1.0-pt.tEdge) * end).transpose() << std::endl;
                // std::getchar();
            }            
            else if (is_vtx_point)
            {
                // std::cout << "is vertex point" << std::endl;
                
                auto he = pt.vertex.halfedge();
                SurfacePoint start_extrinsic = he.tailVertex();
                SurfacePoint end_extrinsic = he.tipVertex();
                start = toTV(start_extrinsic.interpolate(sub_geometry->vertexPositions));
                end = toTV(end_extrinsic.interpolate(sub_geometry->vertexPositions));
                // std::cout << start.transpose() << " " << end.transpose() << std::endl;
                start_end[0] = start_extrinsic.vertex.getIndex();
                start_end[1] = end_extrinsic.vertex.getIndex();
                // std::cout << start_end[0] << " " << start_end[1] << std::endl;
                TV ixn = toTV(pt.interpolate(sub_geometry->vertexPositions));
                // std::cout << ixn.transpose() << std::endl;
                TV test_interp = start;
                if ((test_interp - ixn).norm() > 1e-6)
                {
                    std::swap(start, end);
                    std::swap(start_end[0], start_end[1]);
                }
                TV dir0 = (end-start).normalized();
                TV dir1 = (ixn-start).normalized();
                if ((dir0.cross(dir1)).norm() > 1e-6)
                {
                    std::cout << "error in cross product" << std::endl;
                    return false;
                }
                edge_t = 1.0;
            }
            else
            {
                edge_t = 2.0;
            }

            ixn_data.push_back(IxnData(start, end, (1.0-edge_t), start_end[0], start_end[1]));
            // std::cout << "pass" <<std::endl;
            pt.edge = mesh->edge(pt.edge.getIndex());
            pt.vertex = mesh->vertex(pt.vertex.getIndex());
            pt.face = mesh->face(pt.face.getIndex());
            // std::cout << "pass" <<std::endl;
            pt = pt.inSomeFace();
            // std::cout << "pass" <<std::endl;
        }
        TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
        TV v1 = toTV(path[path.size() - 1].interpolate(geometry->vertexPositions));
        TV ixn0 = toTV(path[1].interpolate(geometry->vertexPositions));
        TV ixn1 = toTV(path[path.size() - 2].interpolate(geometry->vertexPositions));
        
        if (path.size() > 2)
            if ((v0 - ixn0).norm() < 1e-10)
                path.erase(path.begin() + 1);
        if (path.size() > 2)
            if ((v1 - ixn1).norm() < 1e-10)
                path.erase(path.end() - 2);
    }
    sub_geometry->unrequireVertexPositions();
    return true;
}

void VoronoiCells::computeGeodesicDistance(const SurfacePoint& va, const SurfacePoint& vb,
    T& dis, std::vector<SurfacePoint>& path, 
        std::vector<IxnData>& ixn_data, bool trace_path)
{
    dis = 0.0;
    if (trace_path)
    {
        ixn_data.clear();
        path.clear();
    }
    // bool flip_geodesic_succeed = computeGeodesicDistanceEdgeFlip(va, vb, dis, path, ixn_data, trace_path);
    // if (flip_geodesic_succeed)
    //     return;
    int n_tri = extrinsic_indices.rows() / 3;
    std::vector<std::vector<size_t>> mesh_indices_gc(n_tri, std::vector<size_t>(3));
    for (int i = 0; i < n_tri; i++)
        for (int d = 0; d < 3; d++)
            mesh_indices_gc[i][d] = extrinsic_indices[i * 3 + d];
    int n_vtx_extrinsic = extrinsic_vertices.rows() / 3;
    std::vector<gc::Vector3> mesh_vertices_gc(n_vtx_extrinsic);
    
    for (int i = 0; i < n_vtx_extrinsic; i++)
        mesh_vertices_gc[i] = gc::Vector3{extrinsic_vertices(i * 3 + 0), 
            extrinsic_vertices(i * 3 + 1), extrinsic_vertices(i * 3 + 2)};


    // START_TIMING(contruct)
    auto lvals = gcs::makeManifoldSurfaceMeshAndGeometry(mesh_indices_gc, mesh_vertices_gc);
    // FINISH_TIMING_PRINT(contruct)
    std::unique_ptr<gcs::ManifoldSurfaceMesh> sub_mesh;
    std::unique_ptr<gcs::VertexPositionGeometry> sub_geometry;
    std::tie(sub_mesh, sub_geometry) = std::tuple<std::unique_ptr<gcs::ManifoldSurfaceMesh>,
                    std::unique_ptr<gcs::VertexPositionGeometry>>(std::move(std::get<0>(lvals)),  // mesh
                                                             std::move(std::get<1>(lvals))); // geometry
    

    gcs::GeodesicAlgorithmExact mmp(*sub_mesh, *sub_geometry);
    SurfacePoint va_sub(sub_mesh->face(va.face.getIndex()), va.faceCoords);
    SurfacePoint vb_sub(sub_mesh->face(vb.face.getIndex()), vb.faceCoords);
    mmp.propagate(va_sub);
    if (trace_path)
    {
        // std::cout << "trace " << trace_path << std::endl;
        path = mmp.traceBack(vb_sub, dis);
        std::reverse(path.begin(), path.end());
        for (auto& pt : path)
        {
            // std::cout << toTV(pt.faceCoords) << std::endl;
            // std::cout << toTV(pt.interpolate(sub_geometry->vertexPositions)).transpose() << std::endl;
            T edge_t = -1.0; TV start = TV::Zero(), end = TV::Zero();
            bool is_edge_point = (pt.type == gcs::SurfacePointType::Edge);
            bool is_vtx_point = (pt.type == gcs::SurfacePointType::Vertex);
            // std::cout << "is_edge_point " << is_edge_point << " is_vtx_point " << is_vtx_point << std::endl;
            Edge start_end; start_end.setConstant(-1);
            if (is_edge_point)
            {
                auto he = pt.edge.halfedge();
                SurfacePoint start_extrinsic = he.tailVertex();
                SurfacePoint end_extrinsic = he.tipVertex();
                start = toTV(start_extrinsic.interpolate(sub_geometry->vertexPositions));
                end = toTV(end_extrinsic.interpolate(sub_geometry->vertexPositions));
                start_end[0] = start_extrinsic.vertex.getIndex();
                start_end[1] = end_extrinsic.vertex.getIndex();
                TV ixn = toTV(pt.interpolate(sub_geometry->vertexPositions));
                TV test_interp = pt.tEdge * start + (1.0 - pt.tEdge) * end;
                if ((test_interp - ixn).norm() > 1e-6)
                {
                    std::swap(start, end);
                    std::swap(start_end[0], start_end[1]);
                }
                
                TV dir0 = (end-start).normalized();
                TV dir1 = (ixn-start).normalized();
                if ((dir0.cross(dir1)).norm() > 1e-6)
                {
                    std::cout << "start " << start.transpose() << " " << end.transpose() << std::endl;
                    std::cout << "error in cross product " << __FILE__ << std::endl;
                    std::exit(0);
                }
                test_interp = pt.tEdge * start + (1.0 - pt.tEdge) * end;
                if ((ixn - test_interp).norm() > 1e-6)
                {
                    std::cout << "error in interpolation " << __FILE__ << std::endl;
                    std::exit(0);
                }
                edge_t = pt.tEdge;
                // std::cout << pt.tEdge << " " << " cross " << (dir0.cross(dir1)).norm() << std::endl;
                // std::cout << ixn.transpose() << " " << (pt.tEdge * start + (1.0-pt.tEdge) * end).transpose() << std::endl;
                // std::getchar();
            }            
            else if (is_vtx_point)
            {
                // std::cout << "is vertex point" << std::endl;
                
                auto he = pt.vertex.halfedge();
                SurfacePoint start_extrinsic = he.tailVertex();
                SurfacePoint end_extrinsic = he.tipVertex();
                start = toTV(start_extrinsic.interpolate(sub_geometry->vertexPositions));
                end = toTV(end_extrinsic.interpolate(sub_geometry->vertexPositions));
                // std::cout << start.transpose() << " " << end.transpose() << std::endl;
                start_end[0] = start_extrinsic.vertex.getIndex();
                start_end[1] = end_extrinsic.vertex.getIndex();
                // std::cout << start_end[0] << " " << start_end[1] << std::endl;
                TV ixn = toTV(pt.interpolate(sub_geometry->vertexPositions));
                // std::cout << ixn.transpose() << std::endl;
                TV test_interp = start;
                if ((test_interp - ixn).norm() > 1e-6)
                {
                    std::swap(start, end);
                    std::swap(start_end[0], start_end[1]);
                }
                TV dir0 = (end-start).normalized();
                TV dir1 = (ixn-start).normalized();
                if ((dir0.cross(dir1)).norm() > 1e-6)
                {
                    std::cout << "error in cross product " << __FILE__ << std::endl;
                    std::exit(0);
                }
                edge_t = 1.0;
            }
            else
            {
                edge_t = 2.0;
            }
            
            ixn_data.push_back(IxnData(start, end, (1.0-edge_t), start_end[0], start_end[1]));
            pt.edge = mesh->edge(pt.edge.getIndex());
            pt.vertex = mesh->vertex(pt.vertex.getIndex());
            pt.face = mesh->face(pt.face.getIndex());
            
            pt = pt.inSomeFace();
        }
        TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
        TV v1 = toTV(path[path.size() - 1].interpolate(geometry->vertexPositions));
        TV ixn0 = toTV(path[1].interpolate(geometry->vertexPositions));
        TV ixn1 = toTV(path[path.size() - 2].interpolate(geometry->vertexPositions));
        
        if (path.size() > 2)
            if ((v0 - ixn0).norm() < 1e-10)
                path.erase(path.begin() + 1);
        if (path.size() > 2)
            if ((v1 - ixn1).norm() < 1e-10)
                path.erase(path.end() - 2);
        
    }
    else
        dis = mmp.getDistance(vb_sub);
    // std::cout << " mmp geodesic " << path.size() << std::endl;
    // std::cout << "exact " << dis << std::endl;
    // for (auto data : ixn_data)
    // {
    //     std::cout << data.start_vtx_idx << " " << data.end_vtx_idx << " " << data.t << std::endl;
    // }
    // bool flip_geodesic_succeed = computeGeodesicDistanceEdgeFlip(va, vb, dis, path, ixn_data, trace_path);
    // std::cout << "flip geodesic " << dis << std::endl;
    // for (auto data : ixn_data)
    // {
    //     std::cout << data.start_vtx_idx << " " << data.end_vtx_idx << " " << data.t << std::endl;
    // }
    // std::getchar();
    // std::cout << "done MMP" << std::endl;
}


void VoronoiCells::propagateDistanceField(std::vector<SurfacePoint>& samples,
    std::vector<FaceData>& source_data)
{
    int n_samples = samples.size();
    std::vector<gcs::VertexData<T>> dis_to_sources(n_samples);
    // START_TIMING(MMPallNodes)
    if (metric == Geodesic)
    {
        std::mutex m;
        tbb::parallel_for(0, n_samples, [&](int sample_idx)
        {
            int n_tri = extrinsic_indices.rows() / 3;
            std::vector<std::vector<size_t>> mesh_indices_gc(n_tri, std::vector<size_t>(3));
            for (int i = 0; i < n_tri; i++)
                for (int d = 0; d < 3; d++)
                    mesh_indices_gc[i][d] = extrinsic_indices[i * 3 + d];
            int n_vtx_extrinsic = extrinsic_vertices.rows() / 3;
            std::vector<gc::Vector3> mesh_vertices_gc(n_vtx_extrinsic);
            for (int i = 0; i < n_vtx_extrinsic; i++)
                mesh_vertices_gc[i] = gc::Vector3{extrinsic_vertices(i * 3 + 0), 
                    extrinsic_vertices(i * 3 + 1), extrinsic_vertices(i * 3 + 2)};


            // START_TIMING(contruct)
            auto lvals = gcs::makeManifoldSurfaceMeshAndGeometry(mesh_indices_gc, mesh_vertices_gc);
            // FINISH_TIMING_PRINT(contruct)
            std::unique_ptr<gcs::ManifoldSurfaceMesh> sub_mesh;
            std::unique_ptr<gcs::VertexPositionGeometry> sub_geometry;
            std::tie(sub_mesh, sub_geometry) = std::tuple<std::unique_ptr<gcs::ManifoldSurfaceMesh>,
                            std::unique_ptr<gcs::VertexPositionGeometry>>(std::move(std::get<0>(lvals)),  // mesh
                                                                    std::move(std::get<1>(lvals))); // geometry
            
            gcs::GeodesicAlgorithmExact mmp(*sub_mesh, *sub_geometry);
            SurfacePoint sample_i(sub_mesh->face(samples[sample_idx].face.getIndex()), 
                samples[sample_idx].faceCoords);
            
            mmp.propagate(sample_i.inSomeFace());
            gcs::VertexData<T> distance_sub = mmp.getDistanceFunction();
            m.lock();
            gcs::VertexData<T> distances(*mesh);
            for (gcVertex v : mesh->vertices()) 
            {
                distances[v] = distance_sub[sub_mesh->vertex(v.getIndex())];
            }
            dis_to_sources[sample_idx] = distances;
            m.unlock();
        });
    }
    // FINISH_TIMING_PRINT(MMPallNodes)
    std::queue<std::pair<int, int>> queue;

    int sample_cnt = 0;
    for (SurfacePoint& pt : samples)
    {
        if (pt.type == gcs::SurfacePointType::Vertex)
        {
            // std::cout << "vertex" << std::endl;
            auto he = pt.vertex.halfedge();
            do
            {
                queue.push(std::make_pair(he.face().getIndex(), sample_cnt));
                auto next_he = he.twin().next();
                he = next_he;
            } while (he != pt.vertex.halfedge());
            
        }
        else if (pt.type == gcs::SurfacePointType::Edge)
        {
            // std::cout << "edge" << std::endl;
            auto he = pt.edge.halfedge();
            queue.push(std::make_pair(he.face().getIndex(), sample_cnt));
            queue.push(std::make_pair(he.twin().face().getIndex(), sample_cnt));    
        }
        else if (pt.type == gcs::SurfacePointType::Face)
        {
            int face_idx = pt.face.getIndex();
            queue.push(std::make_pair(face_idx, sample_cnt));    
        }
        else
        {
            std::cout << "point type error " << __FILE__ << std::endl;
            std::exit(0);
        }
        sample_cnt++;
    }

    while (queue.size())
    {
        // std::pair<int, int> data_top = queue.top();
        std::pair<int, int> data_top = queue.front();
        queue.pop();

        int face_idx = data_top.first;
        int site_idx = data_top.second;

        SurfacePoint pt = samples[site_idx].inSomeFace();
        TV site_location = toTV(pt.interpolate(geometry->vertexPositions));
        TV v0 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().vertex()]);
        TV v1 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().next().vertex()]);
        TV v2 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().next().next().vertex()]);
        TV current_distance = TV::Zero();
        if (metric == Euclidean)
        {
            current_distance[0] = (site_location - v0).norm();
            current_distance[1] = (site_location - v1).norm();
            current_distance[2] = (site_location - v2).norm();
        }
        else if (metric == Geodesic)
        {
            
            current_distance[0] = dis_to_sources[site_idx][mesh->face(face_idx).halfedge().vertex()];
            current_distance[1] = dis_to_sources[site_idx][mesh->face(face_idx).halfedge().next().vertex()];
            current_distance[2] = dis_to_sources[site_idx][mesh->face(face_idx).halfedge().next().next().vertex()];
        }
        else
        {
            std::cout << "Unknown metric!!!! set to Euclidean" << std::endl;
            metric = Euclidean;
        }
        
        bool updated = true;
        
        for (int i = 0; i < source_data[face_idx].site_indices.size(); i++)
        {
            TV existing_distance = source_data[face_idx].distances[i];
            bool larger_for_all_vertices = true;
            for (int d = 0; d < 3; d++)
            {
                if (current_distance[d] <= existing_distance[d])
                {
                    larger_for_all_vertices = false;
                }
            }
            if (larger_for_all_vertices || (current_distance - existing_distance).norm() < 1e-8) 
                updated = false;
        }
        
        
        if (updated)
        {
            source_data[face_idx].site_indices.push_back(site_idx);
            source_data[face_idx].distances.push_back(current_distance);
            for (auto face : mesh->face(face_idx).adjacentFaces())
            {
                queue.push(std::make_pair(face.getIndex(), site_idx));
            }
        }
        
    }
}


void VoronoiCells::intersectPrisms(std::vector<SurfacePoint>& samples,
    const std::vector<FaceData>& source_data, 
    std::vector<std::pair<SurfacePoint, std::vector<int>>>& ixn_data)
{
    
    T max_h = 1.0;
    T EPSILON = 1e-14;
    ixn_data.clear();
    // check if pt is the same as v0, v1, v2
    auto check_if_existing_projection = [&](const TV2& v0, 
        const TV2& v1, const TV2& v2, const TV2& pt) -> bool
    {
        if ((pt - v0).norm() < EPSILON)
            return true;
        if ((pt - v1).norm() < EPSILON)
            return true;
        if ((pt - v2).norm() < EPSILON)
            return true;
        return false;
    };
    
    // Check if p is on the edge of v1 and v2
    auto is_colinear = [EPSILON] (Eigen::RowVector2d p, Eigen::RowVector2d v1, Eigen::RowVector2d v2) {
        double a=v2.y()-v1.y(), b=-(v2.x()-v1.x()), c=-v1.x()*v2.y()+v2.x()*v1.y();
        double p2l_dist_squared = pow(a*p.x()+b*p.y()+c,2) / (a*a+b*b);
        // std::cout << p << " " << v1 << " " << v2 << " " << p2l_dist_squared << std::endl;
        return p2l_dist_squared <= sqrt(EPSILON); // empirical threshold
    };
    auto is_colinear_ = [] (std::vector<std::vector<Eigen::RowVector2d>> bes, Eigen::RowVector2d v1, Eigen::RowVector2d v2) {
        for (auto &bvs : bes) {
            auto bv1 = std::find(bvs.begin(), bvs.end(), v1);
            auto bv2 = std::find(bvs.begin(), bvs.end(), v2);
            if (bv1!=bvs.end() && bv2!=bvs.end())
                return true;
        }
        return false; // empirical threshold
    };

    // For parallelization
    std::vector<std::vector<std::pair<SurfacePoint, std::vector<int>>>> ixn_data_thread(source_data.size(),
        std::vector<std::pair<SurfacePoint, std::vector<int>>>());

    std::vector<std::vector<IV2>> edge_graph_thread(source_data.size(), std::vector<IV2>());
// #ifdef PARALLEL
//     tbb::parallel_for(0, (int)source_data.size(), [&](int face_idx)
// #else
    for (int face_idx = 0; face_idx < source_data.size(); face_idx++) //source_data.size()
// #endif
    {
        // START_TIMING(face)
        FaceData face_data = source_data[face_idx];
        
        // don't consider 1 vertex case
        if (face_data.site_indices.size() == 1)
// #ifdef PARALLEL
//             return;
// #else
            continue;
// #endif
        TV v0 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().vertex()]);
        TV v1 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().next().vertex()]);
        TV v2 = toTV(geometry->vertexPositions[mesh->face(face_idx).halfedge().next().next().vertex()]);
        // std::cout << "geometry done" << std::endl;

        // rigidly transform to center
        TV trans = (v0 + v1 + v2) / 3.0;
        TV v0_prime = v0 - trans;
        TV v1_prime = v1 - trans;
        TV v2_prime = v2 - trans;

        TV face_normal = -(v2_prime - v0_prime).normalized().cross(
            (v1_prime-v0_prime).normalized()).normalized();

        Matrix<T, 3, 3> R = Eigen::Quaternion<T>().setFromTwoVectors(face_normal, TV(0, 0, 1)).toRotationMatrix();

        v0_prime = R * v0_prime;
        v1_prime = R * v1_prime;
        v2_prime = R * v2_prime;

        TV2 v0_proj = v0_prime.segment<2>(0);
        TV2 v1_proj = v1_prime.segment<2>(0);
        TV2 v2_proj = v2_prime.segment<2>(0);
        // std::cout << v0_proj.transpose() << " " << v1_proj.transpose() << " " << v2_proj.transpose() << std::endl;

        // Initialize the prism as edges
        const Eigen::MatrixXd P1 = (Eigen::MatrixXd(9,3)<<
            v0_prime(0),v0_prime(1),-max_h,
            v1_prime(0),v1_prime(1),-max_h,
            v2_prime(0),v2_prime(1),-max_h,
            v0_prime(0),v0_prime(1),max_h,
            v1_prime(0),v1_prime(1),max_h,
            v2_prime(0),v2_prime(1),max_h,
            v0_prime(0),v0_prime(1),-max_h,
            v1_prime(0),v1_prime(1),-max_h,
            v2_prime(0),v2_prime(1),-max_h).finished();
        const Eigen::MatrixXd P2 = (Eigen::MatrixXd(9,3)<<
            v0_prime(0),v0_prime(1),max_h,
            v1_prime(0),v1_prime(1),max_h,
            v2_prime(0),v2_prime(1),max_h,
            v1_prime(0),v1_prime(1),max_h,
            v2_prime(0),v2_prime(1),max_h,
            v0_prime(0),v0_prime(1),max_h,
            v1_prime(0),v1_prime(1),-max_h,
            v2_prime(0),v2_prime(1),-max_h,
            v0_prime(0),v0_prime(1),-max_h).finished();
        std::vector<std::pair<Eigen::RowVector3d, Eigen::RowVector3d>> edges, bounding_edges;
        for (int i = 0; i < 9; i++) {
            edges.push_back(std::make_pair(P1.row(i), P2.row(i)));
        }
        std::vector<std::vector<Eigen::RowVector2d>> boundary_pts;
        boundary_pts.push_back({v0_prime.head(2), v1_prime.head(2)});
        boundary_pts.push_back({v1_prime.head(2), v2_prime.head(2)});
        boundary_pts.push_back({v2_prime.head(2), v0_prime.head(2)});

        // Matrix used to compute the half-plane equation
        Eigen::MatrixXd A(3, 3);
        A << v0_prime(0), v0_prime(1), 1, v1_prime(0), v1_prime(1), 1, v2_prime(0), v2_prime(1), 1;

        if (face_data.site_indices.size() != face_data.distances.size())
            std::exit(0);

        // voronoi_sites.resize(face_data.site_indices.size()*3);
        // std::cout << v0.transpose() << " " << v1.transpose() << " " << v2.transpose() << std::endl;
        for (int i = 0; i < face_data.site_indices.size(); i++)
        {
            SurfacePoint pt = samples[face_data.site_indices[i]];
            TV site_location = toTV(pt.interpolate(geometry->vertexPositions));
            // voronoi_sites.segment(3*i, 3) = site_location;
            // std::cout << "site: " << (R*(site_location-trans)).transpose() << std::endl;
            
            TV current_distance = face_data.distances[i];

            // Half-plane equation induced by this site
            Eigen::VectorXd equ = A.colPivHouseholderQr().solve(current_distance.cwiseProduct(current_distance));
            // std::cout << "equ: " << equ.transpose() << std::endl;

            int n_edges = edges.size();
            std::vector<Eigen::RowVector3d> new_pts;
            Eigen::RowVector2d center;
            center.setZero();

            for (int e = 0; e < n_edges; e++)
            {
                Eigen::RowVector3d v1 = edges[e].first, v2 = edges[e].second;
                // Half-plane equation value
                double piv1 = v1(2) - (equ(0)*v1(0)+equ(1)*v1(1)+equ(2));
                double piv2 = v2(2) - (equ(0)*v2(0)+equ(1)*v2(1)+equ(2));
                // std::cout << v1 << " " << piv1 << " " << v2 << " " << piv2 << std::endl;
                auto be = std::find(bounding_edges.begin(), bounding_edges.end(), edges[e]);
                if (abs(piv1) < sqrt(EPSILON) && abs(piv2) < sqrt(EPSILON)) { // two planes intersect at the same boundary edge
                    if (std::find(bounding_edges.begin(), bounding_edges.end(), edges[e]) == bounding_edges.end())
                        bounding_edges.push_back(edges[e]);
                    // std::cout << edges[e].first << " " << edges[e].second << std::endl;
                } else if ((piv1>-sqrt(EPSILON) && piv2>-sqrt(EPSILON))) // edge above plane or one above and another one on the plane
                {
                    // std::cout << v1 << "# " << v2 << (be != bounding_edges.end()) << std::endl;
                    if (be != bounding_edges.end()) 
                        bounding_edges.erase(be);
                    // for (auto be : bounding_edges)
                    //     std::cout << be.first << "@ " << be.second << std::endl;
                    // for (auto &bvs : boundary_pts) {
                    //     auto bv1 = std::find(bvs.begin(), bvs.end(), edges[e].first);
                    //     if (bv1 != bvs.end())
                    //         bvs.erase(bv1);
                    //     auto bv2 = std::find(bvs.begin(), bvs.end(), edges[e].second);
                    //     if (bv2 != bvs.end())
                    //         bvs.erase(bv2);
                    // }
                    
                    edges.erase(edges.begin()+e);
                    e -= 1; n_edges -= 1;

                    if ((piv1 < sqrt(EPSILON)) && (std::find(new_pts.begin(), new_pts.end(), v1)==new_pts.end()))
                        new_pts.push_back(v1);
                    else if ((piv2 < sqrt(EPSILON)) && (std::find(new_pts.begin(), new_pts.end(), v2)==new_pts.end()))
                        new_pts.push_back(v2);
                } else if (piv1>sqrt(EPSILON)) { // one of the vertices above plane
                    v1 = piv1*v2 / (piv1-piv2) - piv2*v1 / (piv1-piv2);
                    for (auto &bvs : boundary_pts) {
                        auto bv1 = std::find(bvs.begin(), bvs.end(), edges[e].first.head(2));
                        auto bv2 = std::find(bvs.begin(), bvs.end(), edges[e].second.head(2));
                        // std::cout << edges[e].first << " " << edges[e].second << " " << bvs[0] << " " << bvs[1] << std::endl;
                        if (bv1 != bvs.end() && bv2 != bvs.end()) {
                            // std::cout << "@" << v1 << std::endl;
                            // bvs.erase(bv1);
                            bvs.push_back(v1.head(2));
                        }
                    }
                    edges[e].first = v1;
                    new_pts.push_back(v1);
                    if (be != bounding_edges.end()) 
                        (*be).first = v1;
                    
                } else if (piv2>sqrt(EPSILON)) {
                    v2 = piv2*v1 / (piv2-piv1) - piv1*v2 / (piv2-piv1);
                    for (auto &bvs : boundary_pts) {
                        auto bv1 = std::find(bvs.begin(), bvs.end(), edges[e].first.head(2));
                        auto bv2 = std::find(bvs.begin(), bvs.end(), edges[e].second.head(2));
                        // std::cout << edges[e].first << " " << edges[e].second << " " << bvs[0] << " " << bvs[1] << std::endl;
                        if (bv1 != bvs.end() && bv2 != bvs.end()) {
                            // std::cout << "@" << v2 << std::endl;
                            // bvs.erase(bv2);
                            bvs.push_back(v2.head(2));
                        }
                    }
                    edges[e].second = v2;
                    new_pts.push_back(v2);
                    if (be != bounding_edges.end()) 
                        (*be).second = v2;
                    
                }
            }
            // For sorting
            for (auto &p : new_pts)
                center += p.segment(0,2);
            center /= new_pts.size();

            // Sort newly generated points in order to build new edges
            std::sort(new_pts.begin(), new_pts.end(), 
            [&](Eigen::RowVector3d a, Eigen::RowVector3d b) {
                Eigen::RowVector2d ap = (a.segment(0,2)-center).normalized();
                Eigen::RowVector2d bp = (b.segment(0,2)-center).normalized();
                if (ap.y() >= 0 && bp.y() >= 0) 
                    return ap.x() < bp.x();
                else if (ap.y() < 0 && bp.y() < 0) 
                    return ap.x() > bp.x();
                else
                    return ap.y() < 0;
            });
            // for (int j = 0; j < new_pts.size(); j++)
            // {
            //     std::cout << new_pts[j] << std::endl;
            // }
            // std::cout << std::endl;

            for (int pi = 0; pi < new_pts.size(); pi++)
            {
                Eigen::RowVector3d v1=new_pts[pi], v2=new_pts[(pi+1)%new_pts.size()];
                Eigen::RowVector2d v12d = v1.head(2), v22d = v2.head(2);
                edges.push_back(std::make_pair(v1, v2));

                // Add new edges only if they are not on boundary
                // if (!((is_colinear(v12d, v0_proj, v1_proj) && is_colinear(v22d, v0_proj, v1_proj)) || 
                //         (is_colinear(v12d, v1_proj, v2_proj) && is_colinear(v22d, v1_proj, v2_proj)) || 
                //         (is_colinear(v12d, v2_proj, v0_proj) && is_colinear(v22d, v2_proj, v0_proj)))) 
                if (!is_colinear_(boundary_pts, v12d, v22d))
                    bounding_edges.push_back(edges.back());
            }
        // for (auto be : bounding_edges)
        //     std::cout << be.first << " " << be.second << std::endl;
        }
        // for (auto &bvs : boundary_pts) {
        //     for (auto &bv : bvs)
        //         std::cout << bv << " ";
        //     std::cout << std::endl;
        // }

        for (auto edge : bounding_edges)
        {
            TV edge_vtx0 = edge.first;
            TV edge_vtx1 = edge.second;
            TV2 edge_vtx0_proj = edge_vtx0.segment<2>(0);
            TV2 edge_vtx1_proj = edge_vtx1.segment<2>(0);
            // std::cout << "@" << edge_vtx0_proj.transpose() << " " << edge_vtx1_proj.transpose() << std::endl;

            // compute in-triangle projection
            TV vtx0_proj = edge_vtx0; vtx0_proj[2] = 0.0;
            TV vtx1_proj = edge_vtx1; vtx1_proj[2] = 0.0;
            vtx0_proj = (R.transpose() * vtx0_proj + trans);
            vtx1_proj = (R.transpose() * vtx1_proj + trans);

            // check if four points are co-plannar
            auto check_coplanar = [&](const TV& vi)
            {
                std::vector<int> connecting_sites;
                for (int i = 0; i < face_data.site_indices.size(); i++)
                {
                    MatrixXT testing_vertices(3, 3);
                    TV current_distance = face_data.distances[i];

                    TV v0_prime_lifted = TV(v0_prime[0], v0_prime[1], std::pow(current_distance[0], 2));
                    TV v1_prime_lifted = TV(v1_prime[0], v1_prime[1], std::pow(current_distance[1], 2));
                    TV v2_prime_lifted = TV(v2_prime[0], v2_prime[1], std::pow(current_distance[2], 2));

                    TV normal = -((v2_prime_lifted - v0_prime_lifted).normalized().cross((v1_prime_lifted-v0_prime_lifted).normalized())).normalized();
                    TV v = vi - v0_prime_lifted;
                    T dis_square = std::abs(v.dot(normal));
                    if (dis_square < sqrt(EPSILON))
                    {
                        // std::cout << "coplanar" << std::endl;
                        connecting_sites.push_back(face_data.site_indices[i]);
                    }
                }
                return connecting_sites;
            };
            // std::cout << "check # connection" << std::endl;
            std::vector<int> connecting_sites_v0 = check_coplanar(edge_vtx0);
            std::vector<int> connecting_sites_v1 = check_coplanar(edge_vtx1);
            TV bary0 = computeBarycentricCoordinates(vtx0_proj, v0, v1, v2);
            TV bary1 = computeBarycentricCoordinates(vtx1_proj, v0, v1, v2);
            // std::cout << connecting_sites_v0.size() << " " << connecting_sites_v1.size() << " " << edge_vtx0_proj.transpose() << " " << edge_vtx1_proj.transpose() << std::endl;
            // SurfacePoint p0 = SurfacePoint(mesh->face(face_idx), Vector3{bary0[0], bary0[1], bary0[2]});
            // SurfacePoint p1 = SurfacePoint(mesh->face(face_idx), Vector3{bary1[0], bary1[1], bary1[2]});

            // Correct surface point type
            SurfacePoint p0, p1;
            // if (bary0[0]<1e-5) {
            //     p0 = SurfacePoint(mesh->face(face_idx).halfedge().next(), bary0[2]);
            // } else if (bary0[1]<1e-5) {
            //     p0 = SurfacePoint(mesh->face(face_idx).halfedge().next().next(), bary0[0]);
            // } else if (bary0[2]<1e-5) {
            //     p0 = SurfacePoint(mesh->face(face_idx).halfedge(), bary0[1]);
            // } else
                p0 = SurfacePoint(mesh->face(face_idx), Vector3{bary0[0], bary0[1], bary0[2]});
            // if (bary1[0]<1e-5) {
            //     p1 = SurfacePoint(mesh->face(face_idx).halfedge().next(), bary1[2]);
            // } else if (bary1[1]<1e-5) {
            //     p1 = SurfacePoint(mesh->face(face_idx).halfedge().next().next(), bary1[0]);
            // } else if (bary1[2]<1e-5) {
            //     p1 = SurfacePoint(mesh->face(face_idx).halfedge(), bary1[1]);
            // } else
                p1 = SurfacePoint(mesh->face(face_idx), Vector3{bary1[0], bary1[1], bary1[2]});
            // std::cout << p0.interpolate(geometry->vertexPositions) << " " << p1.interpolate(geometry->vertexPositions) << std::endl;
            ixn_data_thread[face_idx].push_back(std::make_pair(p0, connecting_sites_v0));
            ixn_data_thread[face_idx].push_back(std::make_pair(p1, connecting_sites_v1));

        }

        // FINISH_TIMING_PRINT(face)
        
    }
// #ifdef PARALLEL
//             );
// #endif
    
    for (const auto& data : ixn_data_thread)
    {
        if (data.size())
        {
            for (const auto& pair : data)
                ixn_data.push_back(pair);
        }
    }
}


void VoronoiCells::saveVoronoiDiagram(const std::string& filename)
{
    std::ofstream out(filename);
    out << samples.size() << std::endl;
    for (SurfacePoint& pt : samples)
    {
        pt = pt.inSomeFace();
        out << std::setprecision(16);
        out << pt.face.getIndex() << " " << pt.faceCoords.x << " " << 
            pt.faceCoords.y << " " << pt.faceCoords.z << std::endl;
    }
    out.close();
}

void VoronoiCells::reset()
{
    voronoi_sites.resize(0);
    valid_VD_edges.clear();
    source_data.clear();
    unique_ixn_points.clear();
    samples = samples_rest;
    n_sites = samples.size();
}

void VoronoiCells::resample(int resolution)
{
    samples.clear();
    voronoi_sites.resize(0);
    valid_VD_edges.clear();
    source_data.clear();
    unique_ixn_points.clear();
    gcs::PoissonDiskSampler poissonSampler(*mesh, *geometry);
    samples = poissonSampler.sample(resolution);
    n_sites = samples.size();
    samples_rest = samples;
}

void VoronoiCells::sampleSitesPoissonDisk(T resolution, bool perturb)
{
    gcs::PoissonDiskSampler poissonSampler(*mesh, *geometry);
    // minimum distance between samples, expressed as a multiple of the mean edge length    

    samples = poissonSampler.sample(resolution);
    
    if (perturb)
        tbb::parallel_for(0, (int)samples.size(), [&](int i)
        {
            TV2 dx(0.1, 0.5);
            updateSurfacePoint(samples[i], dx);
        });
    n_sites = samples.size();
    cell_weights.resize(n_sites);
    cell_weights.setConstant(1.0);
    std::cout << "# sites " << samples.size() << std::endl;
    samples_rest = samples;
}
void VoronoiCells::sampleSitesByFace(const TV2& bary, T resolution)
{
    int cnt = 0;
    for (auto face : mesh->faces())
    {
        cnt ++;
        if (cnt % 2 == 0 && (resolution == 0 || resolution == 1 || resolution == 2))
            continue;
        if (cnt % 3 == 0 && (resolution == 0))
            continue;
        if (cnt % 5 == 0 && (resolution == 0 || resolution == 1))
            continue;
        samples.push_back(SurfacePoint(face, Vector3{bary[0], bary[1], 1.0 - bary[0] - bary[1]}));
    }
    n_sites = samples.size();
    cell_weights.resize(n_sites);
    cell_weights.setConstant(1.0);
    std::cout << "# sites " << samples.size() << " # faces " << extrinsic_indices.size() / 3 << std::endl;
    samples_rest = samples;
}

void VoronoiCells::loadGeometry(const std::string& filename)
{
    samples.clear();
    voronoi_sites.resize(0);
    valid_VD_edges.clear();
    source_data.clear();
    unique_ixn_points.clear();
    
    MatrixXT V; MatrixXi F;
    // igl::readOBJ("../../../Projects/IntrinsicSimulation/data/torus.obj", V, F);
    // igl::readOBJ("../../../Projects/IntrinsicSimulation/data/fertility.obj", V, F);
    // igl::readOBJ("../../../Projects/IntrinsicSimulation/data/grid.obj", V, F);
    // igl::readOBJ("../../../Projects/IntrinsicSimulation/data/sphere.obj", V, F);
    // igl::readOBJ("../../../Projects/IntrinsicSimulation/data/rocker_arm_simplified.obj", 
    //     V, F);
    // igl::readOBJ("../../../Projects/IntrinsicSimulation/data/fertility.obj", 
    //     V, F);
    // igl::readOBJ("../../../Projects/IntrinsicSimulation/data/3holes_simplified.obj", 
    //     V, F);
    // igl::readOBJ("../../../Projects/IntrinsicSimulation/data/cactus_simplified.obj", 
    //     V, F);
    igl::readOBJ(filename, 
        V, F);
    
    TV min_corner = V.colwise().minCoeff();
    TV max_corner = V.colwise().maxCoeff();
    
    TV center = 0.5 * (min_corner + max_corner);

    T bb_diag = (max_corner - min_corner).norm();

    T scale = 1.0 / bb_diag;

    for (int i = 0; i < V.rows(); i++)
    {
        V.row(i) -= center;
    }
    V *= scale;
    

    MatrixXT N;
    igl::per_vertex_normals(V, F, N);


    iglMatrixFatten<T, 3>(V, extrinsic_vertices);
    iglMatrixFatten<int, 3>(F, extrinsic_indices);

    int n_tri = extrinsic_indices.rows() / 3;
    std::vector<std::vector<size_t>> mesh_indices_gc(n_tri, std::vector<size_t>(3));
    for (int i = 0; i < n_tri; i++)
        for (int d = 0; d < 3; d++)
            mesh_indices_gc[i][d] = extrinsic_indices[i * 3 + d];
    int n_vtx_extrinsic = extrinsic_vertices.rows() / 3;
    std::vector<gc::Vector3> mesh_vertices_gc(n_vtx_extrinsic);
    for (int i = 0; i < n_vtx_extrinsic; i++)
        mesh_vertices_gc[i] = gc::Vector3{extrinsic_vertices(i * 3 + 0), 
            extrinsic_vertices(i * 3 + 1), extrinsic_vertices(i * 3 + 2)};
    
    auto lvals = gcs::makeManifoldSurfaceMeshAndGeometry(mesh_indices_gc, mesh_vertices_gc);
    std::tie(mesh, geometry) = std::tuple<std::unique_ptr<gcs::ManifoldSurfaceMesh>,
                    std::unique_ptr<gcs::VertexPositionGeometry>>(std::move(std::get<0>(lvals)),  // mesh
                                                             std::move(std::get<1>(lvals))); // geometry
    
}



void VoronoiCells::updateSurfacePoint(SurfacePoint& xi_current, const TV2& search_direction)
{
    int n_tri = extrinsic_indices.rows() / 3;
    std::vector<std::vector<size_t>> mesh_indices_gc(n_tri, std::vector<size_t>(3));
    for (int i = 0; i < n_tri; i++)
        for (int d = 0; d < 3; d++)
            mesh_indices_gc[i][d] = extrinsic_indices[i * 3 + d];
    int n_vtx_extrinsic = extrinsic_vertices.rows() / 3;
    std::vector<gc::Vector3> mesh_vertices_gc(n_vtx_extrinsic);
    
    for (int i = 0; i < n_vtx_extrinsic; i++)
        mesh_vertices_gc[i] = gc::Vector3{extrinsic_vertices(i * 3 + 0), 
            extrinsic_vertices(i * 3 + 1), extrinsic_vertices(i * 3 + 2)};


    // START_TIMING(contruct)
    auto lvals = gcs::makeManifoldSurfaceMeshAndGeometry(mesh_indices_gc, mesh_vertices_gc);
    // FINISH_TIMING_PRINT(contruct)
    std::unique_ptr<gcs::ManifoldSurfaceMesh> sub_mesh;
    std::unique_ptr<gcs::VertexPositionGeometry> sub_geometry;
    std::tie(sub_mesh, sub_geometry) = std::tuple<std::unique_ptr<gcs::ManifoldSurfaceMesh>,
                    std::unique_ptr<gcs::VertexPositionGeometry>>(std::move(std::get<0>(lvals)),  // mesh
                                                             std::move(std::get<1>(lvals))); // geometry

    gcFace face = sub_mesh->face(xi_current.face.getIndex());
    
    gcs::TraceOptions options; 
    options.includePath = true;
    Vector3 trace_vec{search_direction[0], search_direction[1], 
        0.0 - search_direction[0] - search_direction[1]};
    gcs::TraceGeodesicResult result = gcs::traceGeodesic(*sub_geometry, 
                                        face, 
                                        xi_current.faceCoords, 
                                        trace_vec, options);
    if (result.pathPoints.size() != 1)
    {
        SurfacePoint endpoint = result.endPoint.inSomeFace();
        xi_current = SurfacePoint(mesh->face(endpoint.face.getIndex()), endpoint.faceCoords);
    }
}

void VoronoiCells::optimizeForExactVD()
{
    // START_TIMING(ExactVoronoi)
    
    int n_ixn = unique_ixn_points.size();
    

#ifdef PARALLEL
    tbb::parallel_for(0, n_ixn, [&](int i)
#else
    for (int i = 0; i < n_ixn; i++)
#endif
    {
        
        SurfacePoint& xi = unique_ixn_points[i].first;
        xi = xi.inSomeFace();
        // std::cout << toTV(xi.interpolate(geometry->vertexPositions)).transpose() << std::endl;
        // updateSurfacePoint(xi, TV2(0.5, 0.1));
        // std::cout << toTV(xi.interpolate(geometry->vertexPositions)).transpose() << std::endl;
        // std::cout << toTV(unique_ixn_points[i].first.interpolate(geometry->vertexPositions)).transpose() << std::endl;
        // return;
        std::vector<int> site_indices = unique_ixn_points[i].second;
    

        auto fdCheckGradient = [&]()
        {
            T eps = 1e-6;
            T E0, E1;
            TV2 grad;
            updateSurfacePoint(xi, TV2(-0.002, 0.004));
            SurfacePoint xi_tmp = xi;
            // std::cout << toTV(xi.interpolate(geometry->vertexPositions)).transpose() << std::endl;

            T g_norm = computeDistanceMatchingGradient(site_indices, xi, grad, E0);
            std::cout << "dgdw: " << grad.transpose() << std::endl;
            updateSurfacePoint(xi, TV2(eps, 0));
            E0 = computeDistanceMatchingEnergy(site_indices, xi);
            xi = xi_tmp;
            updateSurfacePoint(xi, TV2(-2.0 * eps, 0));
            E1 = computeDistanceMatchingEnergy(site_indices, xi);
            xi = xi_tmp;
            
            // std::cout << toTV(xi.interpolate(geometry->vertexPositions)).transpose() << std::endl;
            std::cout << "dgdw0 " << (E0 - E1) / eps / 2.0 << std::endl;
            updateSurfacePoint(xi, TV2(0.0, eps));
            E0 = computeDistanceMatchingEnergy(site_indices, xi);
            xi = xi_tmp;
            updateSurfacePoint(xi, TV2(0.0, -2.0 * eps));
            E1 = computeDistanceMatchingEnergy(site_indices, xi);
            xi = xi_tmp;
            std::cout << "dgdw1 " << (E0 - E1) / eps / 2.0 << std::endl;
            std::getchar();
        };

        auto fdCheckGradientScale = [&]()
        {
            T eps = 1e-4;
            T E0, E1;
            TV2 grad;
            updateSurfacePoint(xi, TV2(-0.002, 0.004));
            SurfacePoint xi_tmp = xi;
            // std::cout << toTV(xi.interpolate(geometry->vertexPositions)).transpose() << std::endl;

            T g_norm = computeDistanceMatchingGradient(site_indices, xi, grad, E0);
            TV2 dx;
            dx.setRandom();
            dx *= 1.0 / dx.norm();
            dx *= 0.01;
            
            T previous = 0.0;
            for (int i = 0; i < 10; i++)
            {
                xi = xi_tmp;
                updateSurfacePoint(xi, dx);
                T E1 = computeDistanceMatchingEnergy(site_indices, xi);
                T dE = E1 - E0;
                dE -= grad.dot(dx);
                // std::cout << "dE " << dE << std::endl;
                if (i > 0)
                {
                    std::cout << (previous/dE) << std::endl;
                }
                previous = dE;
                dx *= 0.5;
            }
            
            std::getchar();
        };

        // if (xi.type == gcs::SurfacePointType::Face)
        // {
        //     fdCheckGradientScale();
        // }

        T E0;
        TV2 grad;
        T tol = 1e-6;
        TM2 hess;
        int max_iter = 50;
        int iter = 0;
        while (true)
        {
            iter ++;
            if (iter > max_iter)
                break;
            // T g_norm = computeDistanceMatchingGradient(site_indices, xi, grad, E0);
            T g_norm = computeDistanceMatchingEnergyGradientHessian(site_indices, xi, hess, grad, E0);
            // std::getchar();
            T dis_error = std::sqrt(2.0*E0);
            if (g_norm < tol)
            {
                // std::cout << "|g|: " << g_norm << " obj: " << E0 << std::endl;
                break;
            }
            T alpha = 1.0;
            SurfacePoint xi_current = xi;
            // computeDistanceMatchingHessian(site_indices, xi, hess);
            TV2 dw = hess.colPivHouseholderQr().solve(-grad);
            
            for (int ls = 0; ls < 12; ls++)
            {
                xi = xi_current;
                updateSurfacePoint(xi, dw * alpha);
                T E1 = computeDistanceMatchingEnergy(site_indices, xi);
                // std::cout << "E0 " << E0 << " E1 " << E1 << std::endl;
                // std::getchar();
                if (E1 < E0)
                    break;
                alpha *= 0.5;
            }

        }
    }
#ifdef PARALLEL
    );
#endif
    
    voronoi_edges.resize(0);
    for (int i = 0; i < valid_VD_edges.size(); i++)
    {
        int idx0 = valid_VD_edges[i][0];
        int idx1 = valid_VD_edges[i][1];
        TV v0 = toTV(unique_ixn_points[idx0].first.interpolate(geometry->vertexPositions));
        TV v1 = toTV(unique_ixn_points[idx1].first.interpolate(geometry->vertexPositions));
        voronoi_edges.push_back(std::make_pair(v0, v1));
    }   
    // FINISH_TIMING_PRINT(ExactVoronoi)
}

bool VoronoiCells::linearSolve(StiffnessMatrix& K, const VectorXT& residual, VectorXT& du)
{
    START_TIMING(LinearSolve)
    Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> solver;
    // Eigen::CholmodSupernodalLLT<StiffnessMatrix> solver;
    
    T alpha = 1e-6;
    StiffnessMatrix H(K.rows(), K.cols());
    H.setIdentity(); H.diagonal().array() = 1e-10;
    K += H;
    solver.analyzePattern(K);
    // T time_analyze = t.elapsed_sec();
    // std::cout << "\t analyzePattern takes " << time_analyze << "s" << std::endl;
    
    int indefinite_count_reg_cnt = 0, invalid_search_dir_cnt = 0, invalid_residual_cnt = 0;

    for (int i = 0; i < 50; i++)
    {
        solver.factorize(K);
        // std::cout << "factorize" << std::endl;
        if (solver.info() == Eigen::NumericalIssue)
        {
            K.diagonal().array() += alpha;
            alpha *= 10;
            indefinite_count_reg_cnt++;
            continue;
        }

        du = solver.solve(residual);
        
        T dot_dx_g = du.normalized().dot(residual.normalized());

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;

        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-6;
        if (!search_dir_correct_sign)
        {   
            invalid_search_dir_cnt++;
        }
        
        // bool solve_success = true;
        // bool solve_success = (K * du - residual).norm() / residual.norm() < 1e-6;
        bool solve_success = du.norm() < 1e3;
        
        if (!solve_success)
            invalid_residual_cnt++;
        // std::cout << "PD: " << positive_definte << " direction " 
        //     << search_dir_correct_sign << " solve " << solve_success << std::endl;

        if (positive_definte && search_dir_correct_sign && solve_success)
        {
            
            // if (verbose)
            {
                std::cout << "\t===== Linear Solve ===== " << std::endl;
                std::cout << "\tnnz: " << K.nonZeros() << std::endl;
                // std::cout << "\t takes " << t.elapsed_sec() << "s" << std::endl;
                std::cout << "\t# regularization step " << i 
                    << " indefinite " << indefinite_count_reg_cnt 
                    << " invalid search dir " << invalid_search_dir_cnt
                    << " invalid solve " << invalid_residual_cnt << std::endl;
                std::cout << "\tdot(search, -gradient) " << dot_dx_g << std::endl;
                std::cout << "\t======================== " << std::endl;
                FINISH_TIMING_PRINT(LinearSolve)
            }
            return true;
        }
        else
        {
            K.diagonal().array() += alpha;
            alpha *= 10;
        }
    }
    return false;
}


// Implementation of
// SurfaceVoronoi: Efficiently Computing Voronoi Diagrams Over Mesh Surfaces with Arbitrary Distance Solvers

void VoronoiCells::constructVoronoiDiagram(bool exact, bool load_from_file)
{
    START_TIMING(constructVoronoiDiagram)
    voronoi_sites.resize(0);
    valid_VD_edges.clear();
    voronoi_edges.clear();
    source_data.clear();
    unique_ixn_points.clear();
    int n_tri = mesh->nFaces();
    // std::cout << "# sites " << samples.size() << std::endl;
    // std::cout << "# faces " << n_tri << std::endl;
    if (load_from_file)
    {
        std::ifstream in("samples.txt");
        int n_samples; in >> n_samples;
        samples.resize(n_samples);
        for (int i = 0; i < n_samples; i++)
        {
            int face_idx;
            T wx, wy, wz;
            in >> face_idx >> wx >> wy >> wz;
            samples[i] = SurfacePoint(mesh->face(face_idx), Vector3{wx, wy, wz});
        }
        in.close();
    }

    n_sites = samples.size();
    voronoi_sites.resize(n_sites * 3);
    int cnt = 0;
    for (SurfacePoint& pt : samples)
    {
        pt = pt.inSomeFace();
        voronoi_sites.segment<3>(cnt * 3) = toTV(pt.interpolate(geometry->vertexPositions));
        cnt++;
    }
    
    source_data.resize(n_tri, FaceData());
    // START_TIMING(PropogateDistance)
    propagateDistanceField(samples, source_data);
    // FINISH_TIMING_PRINT(PropogateDistance)
    
    // START_TIMING(PrismCutting)
    std::vector<std::pair<SurfacePoint, std::vector<int>>> ixn_data;

    intersectPrisms(samples, source_data, ixn_data);
    // FINISH_TIMING_PRINT(PrismCutting)
    
    // remove duplication
    
    // START_TIMING(RemoveDuplicates)
    
    MatrixXT ixn_points(ixn_data.size(), 3);
    for (int i = 0; i < ixn_data.size(); i++)
    {
        ixn_points.row(i) = toTV(ixn_data[i].first.interpolate(geometry->vertexPositions));
    }
    MatrixXT unique_points;
    Eigen::VectorXi c2f, f2c;
    igl::remove_duplicate_vertices(ixn_points, 1e-8, unique_points, c2f, f2c);
    
    unique_ixn_points.resize(c2f.rows());
    for (int i = 0; i < c2f.rows(); i++)
    {
        unique_ixn_points[i] = ixn_data[c2f[i]];
    }
    FINISH_TIMING_PRINT(constructVoronoiDiagram)
    // FINISH_TIMING_PRINT(RemoveDuplicates)
    
    // START_TIMING(ConstructConnectivity)
    constructVoronoiCellConnectivity();
    // FINISH_TIMING_PRINT(ConstructConnectivity)
    
    for (int i = 0; i < ixn_data.size(); i+=2)
    {
        int idx0 = f2c[i];
        int idx1 = f2c[i+1];
        valid_VD_edges.push_back(Edge(idx0, idx1));
        TV v0 = toTV(unique_ixn_points[idx0].first.interpolate(geometry->vertexPositions));
        TV v1 = toTV(unique_ixn_points[idx1].first.interpolate(geometry->vertexPositions));
        voronoi_edges.push_back(std::make_pair(v0, v1));
    }

    if (exact)
    {
        optimizeForExactVD();   
    }
    
    use_debug_face_color = false;
    if (use_debug_face_color)
        updateFaceColor();
    
}

void VoronoiCells::updateFaceColor()
{
    int n_tri = mesh->nFaces();
    face_color.resize(n_tri, 3);
    face_color.setZero();
    
    tbb::parallel_for(0, n_tri, [&](int i)
    {
        if (source_data[i].site_indices.size() == 0)
        {
            std::cout << "error" << std::endl;
        }
        else if (source_data[i].site_indices.size() == 1)
            face_color.row(i) = TV(153.0/255.0, 204/255.0, 1.0);
        else if (source_data[i].site_indices.size() == 2)
            face_color.row(i) = TV(153.0/255.0, 153.0/255.0, 1.0);
        else if (source_data[i].site_indices.size() == 3)
            face_color.row(i) = TV(178.0/255.0, 102.0/255.0, 1.0);
        else if (source_data[i].site_indices.size() == 4)
            face_color.row(i) = TV(1.0, 0.0, 1.0);
        else if (source_data[i].site_indices.size() == 5)
            face_color.row(i) = TV(1.0, 0.0, 0.2);
        else if (source_data[i].site_indices.size() > 5)
            face_color.row(i) = TV(1.0, 0.0, 0.0);
    });
}

void VoronoiCells::computeSurfacePointdxds(const SurfacePoint& pt, Matrix<T, 3, 2>& dxdw)
{
    TV v0 = toTV(geometry->vertexPositions[pt.face.halfedge().vertex()]);
    TV v1 = toTV(geometry->vertexPositions[pt.face.halfedge().next().vertex()]);
    TV v2 = toTV(geometry->vertexPositions[pt.face.halfedge().next().next().vertex()]);

    dxdw.col(0) = v0 - v2;
    dxdw.col(1) = v1 - v2;
}


void VoronoiCells::computeDualIDT(std::vector<std::pair<TV, TV>>& idt_edge_vertices,
        std::vector<IV>& idt_indices)
{
    idt_indices.resize(0);
    for (auto ixn : unique_ixn_points)
    {
        if (ixn.second.size() == 3)
        {
            idt_indices.push_back(IV(ixn.second[0], ixn.second[1], ixn.second[2]));
        }
    }
    MatrixXi idt_faces(idt_indices.size(), 3);
    for (int i = 0; i < idt_indices.size(); i++)
        idt_faces.row(i) = idt_indices[i];
    
    MatrixXi idt_edges;
    igl::edges(idt_faces, idt_edges);
    int n_idt_edges = idt_edges.rows();
    std::vector<std::vector<std::pair<TV, TV>>> edges_thread(n_idt_edges, std::vector<std::pair<TV, TV>>());
    tbb::parallel_for(0, n_idt_edges, [&](int i)
    {
        T geo_dis; std::vector<SurfacePoint> path;
        std::vector<IxnData> ixn_data;
        computeGeodesicDistance(samples[idt_edges(i, 0)], samples[idt_edges(i, 1)], 
            geo_dis, path, ixn_data, true);
        for(int j = 0; j < path.size() - 1; j++)
        {
            edges_thread[i].push_back(std::make_pair(
                toTV(path[j].interpolate(geometry->vertexPositions)),
                toTV(path[j+1].interpolate(geometry->vertexPositions))
            ));
        }
    });
    for (int i = 0; i < n_idt_edges; i++)
    {
        idt_edge_vertices.insert(idt_edge_vertices.end(), 
            edges_thread[i].begin(), edges_thread[i].end());
    }
    
}

void VoronoiCells::constructVoronoiCellConnectivity()
{
    geometry->requireVertexNormals();
    std::vector<std::unordered_set<int>> cell_vtx_idx_lists(samples.size());
    for (int i = 0; i < unique_ixn_points.size(); i++)
    {
        for (int j = 0; j < unique_ixn_points[i].second.size(); j++)
        {
            cell_vtx_idx_lists[unique_ixn_points[i].second[j]].insert(i);
        }
    }
    // std::cout << "construct set" << std::endl;
    std::vector<VtxList> oriented(cell_vtx_idx_lists.size(), VtxList());
    int n_cells = cell_vtx_idx_lists.size();
    tbb::parallel_for(0, n_cells, [&](int i)
    // for (int i = 0; i < cell_vtx_idx_lists.size(); i++)
    {
        auto indices = cell_vtx_idx_lists[i];
        VtxList cell_i;
        std::vector<TV> vertices;
        for (int idx : indices)
        {
            cell_i.push_back(idx);
            TV xi = toTV(unique_ixn_points[idx].first.interpolate(geometry->vertexPositions));
            vertices.push_back(xi);
        }
        TV site_location = toTV(samples[i].interpolate(geometry->vertexPositions));
        
        TV x0 = vertices[0];
        
        std::sort(cell_i.begin(), cell_i.end(), [&](int a, int b)
        {
            TV xi = toTV(unique_ixn_points[a].first.interpolate(geometry->vertexPositions));
            TV xj = toTV(unique_ixn_points[b].first.interpolate(geometry->vertexPositions));
            
            TV E0 = (xi - site_location).normalized();
            TV E1 = (xj - site_location).normalized();
            TV ref = (x0 - site_location).normalized();
            T dot_sign0 = std::max(std::min(E0.dot(ref), 1.0), -1.0);
            T dot_sign1 = std::max(std::min(E1.dot(ref), 1.0), -1.0);
            TV cross_sin0 = E0.cross(ref);
            TV cross_sin1 = E1.cross(ref);
            // use normal and cross product to check if it's larger than 180 degree
            TV normal = toTV(samples[i].interpolate(geometry->vertexNormals));
            T angle_a = cross_sin0.dot(normal) > 0 ? std::acos(dot_sign0) : 2.0 * M_PI - std::acos(dot_sign0);
            T angle_b = cross_sin1.dot(normal) > 0 ? std::acos(dot_sign1) : 2.0 * M_PI - std::acos(dot_sign1);
            
            return angle_a < angle_b;
        });
        // shift the indices such that the first one in the list is a vtx node not an edge node
        // just for the convenience of later operations.
        VtxList vtx_list = cell_i;
        int vtx_node_idx = -1;
        for (int j = 0; j < vtx_list.size(); j++)
        {
            if (unique_ixn_points[vtx_list[j]].second.size() != 2)
            {
                vtx_node_idx = j; 
                break;
            }
        }
        VtxList vtx_list_shifted = cell_i;
        for (int j = vtx_node_idx; j < vtx_node_idx + vtx_list.size(); j++)
        {
            cell_i[j - vtx_node_idx] = vtx_list_shifted[(j) % vtx_list.size()];
        }
        oriented[i] = cell_i;
    // }
    });

    geometry->unrequireVertexNormals();

    // compute segment lengths
    if (edge_weighting || add_coplanar || add_length)
    {
        voronoi_cell_vertices = oriented;
        voronoi_cell_data.clear();
        voronoi_cell_data.resize(voronoi_cell_vertices.size());
        for (int i = 0; i < voronoi_cell_vertices.size(); i++)
        {
            VtxList vtx_list = voronoi_cell_vertices[i];
            VtxList cell_vtx_nodes;
            VoronoiCellData cell_data;
            cell_data.cell_nodes = vtx_list;
            
            for (int j = 0; j < vtx_list.size(); j++)
            {
                auto ixn_point = unique_ixn_points[vtx_list[j]];
                if (ixn_point.second.size() != 2)
                {
                    cell_data.cell_vtx_nodes.push_back(j);
                }
                else
                {
                    cell_data.cell_edge_nodes.push_back(j);
                }
            }
            int n_cell_vtx = cell_data.cell_vtx_nodes.size();
            cell_data.cell_edge_lengths.resize(n_cell_vtx);
            
            T length = 0.0;
            int cnt = 0;
            // std::cout << "# of nodes " << vtx_list.size() << std::endl;
            for (int j = 0; j < vtx_list.size(); j++)
            {
                // std::cout << j << "/" << vtx_list.size() << " " << 
                //     (j+1)%vtx_list.size() << " " << cnt << " " <<
                //     (cnt+1)%cell_data.cell_vtx_nodes.size() << " " << 
                //     cell_data.cell_vtx_nodes[(cnt+1)%cell_data.cell_vtx_nodes.size()] << std::endl;
                // std::getchar();
                // std::cout << j << " " << (j+1)%vtx_list.size() << std::endl;
                TV xj = toTV(unique_ixn_points[vtx_list[j]].first.interpolate(geometry->vertexPositions));
                TV xk = toTV(unique_ixn_points[vtx_list[(j+1)%vtx_list.size()]].first.interpolate(geometry->vertexPositions));
                length += (xk - xj).norm();
                if ((j+1)%vtx_list.size() == 
                    cell_data.cell_vtx_nodes[(cnt+1)%cell_data.cell_vtx_nodes.size()])
                {
                    cell_data.cell_edge_lengths[cnt] = length;
                    cnt += 1;
                    length = 0;
                }
            }

            

            voronoi_cell_data[i] = cell_data;
        }   
    }
    if (add_length)
    {
        find3DVoronoiEdges();
    }
}


void VoronoiCells::traceGeodesics()
{
    int n_springs = unique_voronoi_edges.size();
    current_length.resize(n_springs);
    paths.resize(n_springs);
    ixn_data_list.resize(n_springs);
#ifdef PARALLEL_GEODESIC
    tbb::parallel_for(0, n_springs, [&](int i)
#else
	for(int i = 0; i < n_springs; i++)
#endif
    {
        SurfacePoint vA = unique_ixn_points[unique_voronoi_edges[i][0]].first;
        SurfacePoint vB = unique_ixn_points[unique_voronoi_edges[i][1]].first;
        
        T geo_dis = 0.0; std::vector<SurfacePoint> path;
		std::vector<IxnData> ixn_data;
        computeExactGeodesic(vA, vB, geo_dis, path, ixn_data, true);
        // computeExactGeodesicEdgeFlip(vA, vB, geo_dis, path, ixn_data, true);
        // computeGeodesicHeatMethod(vA, vB, geo_dis, path, ixn_data, true);
        ixn_data_list[i] = ixn_data;
        paths[i] = path;
        current_length[i] = geo_dis;
    }
#ifdef PARALLEL_GEODESIC
    );
#endif
    retrace = false;
}

void VoronoiCells::computeReferenceLength()
{
    if (!add_length)
        return;
    find3DVoronoiEdges();
    target_length = current_length.mean();
}

void VoronoiCells::find3DVoronoiEdges()
{
    if (!add_length)
        return;
    // std::unordered_set<Edge, VectorHash<2>> unique_set;
    std::unordered_map<Edge, int, VectorHash<2>> hash_map;
    unique_voronoi_edges.clear();
    int cnt = 0;
    iterateVoronoiCells([&](const VoronoiCellData& cell_data, int cell_idx)
    {
        int n_cell_node = cell_data.cell_vtx_nodes.size();
        for (int i = 0; i < n_cell_node; i++)
        {
            Edge ei = Edge(
                cell_data.cell_nodes[cell_data.cell_vtx_nodes[i]],
                cell_data.cell_nodes[cell_data.cell_vtx_nodes[(i+1)%n_cell_node]]
            );
            if (hash_map.count(ei) == 0 && hash_map.count(Edge(ei[1], ei[0])) == 0)
            {
                hash_map[ei] = cnt++;
            }
        }
    });
    for (const auto& data : hash_map)
    {
        Edge edge = data.first;
        unique_voronoi_edges.emplace_back(edge[0], edge[1]);
    }
    std::cout << "# voronoi edges " << unique_voronoi_edges.size() << std::endl;

    traceGeodesics();
}