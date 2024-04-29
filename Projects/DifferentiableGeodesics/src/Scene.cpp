#include <igl/readOBJ.h>
#include <igl/readDMAT.h>
#include <igl/writeOBJ.h>
#include <igl/edges.h>
#include <igl/per_vertex_normals.h>
#include <igl/facet_adjacency_matrix.h>
#include <igl/barycenter.h>
#include <igl/colormap.h>
#include <igl/readTGF.h>
#include <igl/readMSH.h>
#include <igl/boundary_facets.h>
#include "../include/IntrinsicSimulation.h"
#include "../autodiff/Elasticity.h"
#include <mutex>
void IntrinsicSimulation::updateBaseMesh(const MatrixXT& new_pos)
{
    int n_vtx_extrinsic = extrinsic_vertices.rows() / 3;
    for (int i = 0; i < n_vtx_extrinsic; i++)
    {
        TV vi = extrinsic_vertices.segment<3>(i * 3);
        extrinsic_vertices.segment<3>(i * 3) = new_pos.row(i);
    }
    geometry->requireVertexPositions();
    for (int i = 0; i < n_vtx_extrinsic; i++)
    {
        geometry->vertexPositions[mesh->vertex(i)] = 
            gc::Vector3{extrinsic_vertices(i * 3 + 0), 
            extrinsic_vertices(i * 3 + 1), 
            extrinsic_vertices(i * 3 + 2)};
        
    }
    geometry->unrequireVertexPositions();
    geometry->refreshQuantities();
    retrace = true;
    traceGeodesics();
}
void IntrinsicSimulation::expandBaseMesh(T increment, int dir)
{
    
    TV center = TV::Zero();
    int n_vtx_extrinsic = extrinsic_vertices.rows() / 3;
    for (int i = 0; i < n_vtx_extrinsic; i++)
    {
        center += extrinsic_vertices.segment<3>(i * 3);
    }
    center /= T(n_vtx_extrinsic);
    for (int i = 0; i < n_vtx_extrinsic; i++)
    {
        TV vi = extrinsic_vertices.segment<3>(i * 3);
        // extrinsic_vertices.segment<3>(i * 3) = vi + increment * (vi - center);
        if (vi[dir] > 1e-2)
            extrinsic_vertices[i * 3 + dir] = vi[dir] + increment;
        else if (vi[dir] < 1e-2)
            extrinsic_vertices[i * 3 + dir] = vi[dir] - increment;
    }
    geometry->requireVertexPositions();
    for (int i = 0; i < n_vtx_extrinsic; i++)
    {
        geometry->vertexPositions[mesh->vertex(i)] = 
            gc::Vector3{extrinsic_vertices(i * 3 + 0), 
            extrinsic_vertices(i * 3 + 1), 
            extrinsic_vertices(i * 3 + 2)};
        
    }
    geometry->unrequireVertexPositions();
    geometry->refreshQuantities();
    retrace = true;
    traceGeodesics();
}
void IntrinsicSimulation::buildMeshGeometry(const MatrixXT& V, const MatrixXi& F)
{
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
    TV min_corner = V.colwise().minCoeff();
    TV max_corner = V.colwise().maxCoeff();
    scene_bb_diag = (max_corner - min_corner).norm();
}

void IntrinsicSimulation::assignFaceColorBasedOnGeodesicTriangle
    (MatrixXT& V, MatrixXi& F, MatrixXT& C, VectorXT& face_quantity)
{
    VectorXT face_quantity_coarse(triangles.size());

    iterateTriangleSerial([&](Triangle tri, int tri_idx){
        Edge e0, e1, e2;
 		getTriangleEdges(tri, e0, e1, e2);
        TV l0; l0 << undeformed_length[edge_map[e0]], undeformed_length[edge_map[e1]], undeformed_length[edge_map[e2]];
        TV l; l << current_length[edge_map[e0]], current_length[edge_map[e1]], current_length[edge_map[e2]];
        TM A_inv; T Ai = layout2D(l0, A_inv);
        computeGeodesicNHEnergyWithC(lambda, mu, l, A_inv, Ai, face_quantity_coarse[tri_idx]);
        // face_quantity_coarse[tri_idx] = tri_idx%8;
     });
    MatrixXT color(triangles.size(),3);
    color.setZero();
    color.col(0) = face_quantity_coarse;
    // igl::colormap(igl::COLOR_MAP_TYPE_JET, face_quantity_coarse, true, color);

    V.resize(0, 3);
    F.resize(0, 3);
    C.resize(0, 3);
    START_TIMING(remesh);
    std::mutex m;
    tbb::parallel_for(0, (int)triangles.size(), [&](int i)
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

        std::vector<SurfacePoint> pts;
        std::unordered_map<gcVertex, int> vtxs;
        std::unordered_map<gcFace, std::vector<int>> faces;
        for (int j = 0; j < 3; j++)
        {
            Vector3 vec_inner = surface_points[triangles[i][(j+2)%3]].interpolate(geometry->vertexPositions);
            SurfacePoint pj(sub_mesh->face(surface_points[triangles[i][j]].face.getIndex()), surface_points[triangles[i][j]].faceCoords);
            SurfacePoint nxt(sub_mesh->face(surface_points[triangles[i][(j+1)%3]].face.getIndex()), surface_points[triangles[i][(j+1)%3]].faceCoords);
            gcs::GeodesicAlgorithmExact mmp(*sub_mesh, *sub_geometry);
            std::vector<SurfacePoint> source;
            source.push_back(pj);
            mmp.propagate(source);
            std::vector<SurfacePoint> p = mmp.traceBack(nxt);
            pts.insert(pts.end(), p.begin(), p.begin()+p.size()-1);

            for (int k = 1; k < p.size()-1; k++)
            {
                SurfacePoint pk = p[k];
                Vector3 veck = pk.interpolate(sub_geometry->vertexPositions);
                Vector3 vecknxt = p[k+1].interpolate(sub_geometry->vertexPositions);

                if (pk.type == gcs::SurfacePointType::Edge) {
                    for(auto v : pk.edge.adjacentVertices()) {
                        Vector3 vec = sub_geometry->vertexPositions[v];
                        if (gc::dot(gc::cross(vec-veck, vecknxt-veck), gc::cross(vec_inner-veck, vecknxt-veck))>0) {
                            vtxs[v] += 1;
                        } else {
                            vtxs[v] -= 1;
                        }
                    }
                } else {
                    // std::cout << "not edge!" << std::endl;
                }
                if (pk.type == gcs::SurfacePointType::Vertex) {
                    for(auto v : pk.vertex.adjacentVertices()) {
                        Vector3 vec = sub_geometry->vertexPositions[v];
                        if (gc::dot(gc::cross(vec-veck, vecknxt-veck), gc::cross(vec_inner-veck, vecknxt-veck))>0) {
                            vtxs[v] += 1;
                        } else {
                            vtxs[v] -= 1;
                        }
                    }
                }
            }
            
            
        }

        for (int j = 0; j < pts.size(); j++)
        {
            SurfacePoint pj = pts[j];
            if (pj.type == gcs::SurfacePointType::Edge) {
                for (auto f : pj.edge.adjacentFaces()) {
                    faces[f].push_back(j);
                }

            } else if (pj.type == gcs::SurfacePointType::Vertex) {
                for (auto f : pj.vertex.adjacentFaces()) {
                    faces[f].push_back(j);
                }
            } else {
                faces[pj.face].push_back(j);
            }
        }

        bool is_done = false;
        while (!is_done)
        {
            is_done = true;
            for (const auto &[vtx, value] : vtxs) {
                if (value > 0 && std::find(pts.begin(), pts.end(), SurfacePoint(vtx))==pts.end() ) {
                    for (auto f : vtx.adjacentFaces()) {
                        if (faces[f].size() == 0) {
                            for (auto v : f.adjacentVertices()) {
                                if (vtxs[v] <= 0) {
                                    is_done = false;
                                    vtxs[v] = 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        for (const auto &[vtx, value] : vtxs) {
            if (value > 0 && std::find(pts.begin(), pts.end(), SurfacePoint(vtx))==pts.end() ) {
                for (auto f : vtx.adjacentFaces()) {
                    faces[f].push_back(pts.size());
                }
                pts.push_back(SurfacePoint(vtx));
            }
        }

        int tri_cnt = 0;
        for (auto [face, idx] : faces) {
            if (idx.size() == 3)
                tri_cnt++;
            if (idx.size() == 1) {
                // if (pts[idx[0]].type != gcs::SurfacePointType::Vertex) {
                    // std::cout << "Triangle " << i << ": Face has only one point!!!" << " " << pts[idx[0]] << std::endl;
                    // std::exit(0);
                // }
                tri_cnt++;
            }
            // C.row(face.getIndex()).setOnes();
        }
        int n_addp = faces.size() - tri_cnt;

        int n_v = pts.size();
        Eigen::MatrixXd sub_V(n_v+n_addp, 3);
        for (int j = 0; j < n_v; j++)
        {
            sub_V.row(j) = toTV(pts[j].interpolate(sub_geometry->vertexPositions)).transpose();
        }
        Eigen::MatrixXi sub_F(0, 3);
        Eigen::MatrixXd sub_C(0, 3);

        int fid = 0;
        for (auto [face, idx] : faces)
        {
            gcFace cur_face = face;
            // if (cur_face.getIndex() != 400)
            //     continue;
            // std::cout << i << " " << idx.size() << std::endl;
            std::vector<int> cur_idx = idx;
            Eigen::Vector3d center; center.setZero();
            
            if (cur_idx.size() < 2) {
                continue;
            } 

            // Find center and sort
            for (int j = 0; j < cur_idx.size(); j++)
            {
                SurfacePoint pj = pts[cur_idx[j]];
                // std::cout << pj << std::endl;
                center += toTV(pj.interpolate(sub_geometry->vertexPositions));
            }
            center /= cur_idx.size();
            Eigen::Vector3d x0 = toTV(pts[cur_idx[0]].interpolate(sub_geometry->vertexPositions));

            std::sort(cur_idx.begin(), cur_idx.end(), [&](int a, int b)
            {
                TV xi = toTV(pts[a].interpolate(sub_geometry->vertexPositions));
                TV xj = toTV(pts[b].interpolate(sub_geometry->vertexPositions));
                
                TV E0 = (xi - center).normalized();
                TV E1 = (xj - center).normalized();
                TV ref = (x0 - center).normalized();
                T dot_sign0 = E0.dot(ref);
                T dot_sign1 = E1.dot(ref);
                TV cross_sin0 = E0.cross(ref);
                TV cross_sin1 = E1.cross(ref);
                // use normal and cross product to check if it's larger than 180 degree
                TV normal = toTV(sub_geometry->faceNormal(cur_face));
                T angle_a = cross_sin0.dot(normal) > 0 ? std::acos(dot_sign0) : 2.0 * M_PI - std::acos(dot_sign0);
                T angle_b = cross_sin1.dot(normal) > 0 ? std::acos(dot_sign1) : 2.0 * M_PI - std::acos(dot_sign1);
                
                return angle_a < angle_b;
            });

            // Manually simplify triangularization for faces with 3 vertices
            if (cur_idx.size() == 3) {
                MatrixXi cur_f(1, 3);
                cur_f << cur_idx[2], cur_idx[1], cur_idx[0];
                int offset_f = sub_F.rows();
                sub_F.conservativeResize(sub_F.rows() + 1, 3);
                sub_F.bottomRows(1) = cur_f;
                sub_C.conservativeResize(sub_C.rows() + 1, 3);
                sub_C.bottomRows(1) = color.row(i);
                // std::cout << i << ": " << color.row(i) << std::endl;
                continue;
            }

            // For normal cases
            int cur_n = cur_idx.size();
            MatrixXi cur_f(cur_n, 3);
            int cen = n_v + fid;
            for (int j = 0; j < cur_n; j++)
            {
                int cur = cur_idx[j], nxt = cur_idx[(j+1)%cur_n];
                cur_f.row(j)   << cur, cen, nxt;
            }
            sub_V.row(n_v + fid) = center.transpose();
            
            int offset_f = sub_F.rows();
            sub_F.conservativeResize(sub_F.rows() + cur_f.rows(), 3);
            sub_F.bottomRows(cur_f.rows()) = cur_f;
            sub_C.conservativeResize(sub_C.rows() + cur_f.rows(), 3);
            sub_C.bottomRows(cur_f.rows()).rowwise() = color.row(i);
            fid++;
        
        }

        m.lock();
        int offset_v = V.rows();
        int offset_f = F.rows();
        V.conservativeResize(offset_v+sub_V.rows(), 3);
        V.bottomRows(sub_V.rows()) = sub_V;

        for (int fi = 0; fi < sub_F.rows(); fi++)
        {
            sub_F.row(fi) += Eigen::RowVector3i::Constant(offset_v);
        }
        F.conservativeResize(offset_f+sub_F.rows(), 3);
        F.bottomRows(sub_F.rows()) = sub_F;

        C.conservativeResize(offset_f+sub_C.rows(), 3);
        C.bottomRows(sub_C.rows()) = sub_C;
        m.unlock();
        
    });
    face_quantity = C.col(0);
    // MatrixXT SV, SC;
    // VectorXi SVI, SVJ, IA,IC;
    // MatrixXi SF, UF;
    // igl::remove_duplicate_vertices(V, F, 1e-8, SV, SVI, SVJ, SF);
    // UF = SF;
    // SC.resize(UF.rows(), 3); 
    // face_quantity.resize(SC.rows());
    // for (int i = 0; i < UF.rows(); i++)
    // {
    //     // face_quantity[i] = C(IA(i), 0);
    //     // SC.row(i) = C.row(IA(i));
    //     SC.row(i) = C.row(i);
    // }
    // std::cout << I.segment<10>(0).transpose() << std::endl;
    // V = SV; 
    // F = UF;
    // C = SC;

    // igl::unique_simplices(SF, UF, IA, IC);
    // igl::writeOBJ("remove_duplicate_vtx.obj", SV, SF);
    // igl::writeOBJ("unique.obj", SV, UF);
    // igl::bfs_orient(SF, FF, FC);
    // igl::resolve_duplicated_faces(FF, NF, I);
    // igl::writeOBJ("remove_duplicate_face.obj", SV, NF);
    // std::cout << V.rows() << " " << SV.rows() << " " << F.rows() << " " << SF.rows() << " " << NF.rows() << std::endl;
    // SC.resize(UF.rows(), 3); 
    // face_quantity.resize(SC.rows());
    // for (int i = 0; i < UF.rows(); i++)
    // {
    //     face_quantity[i] = C(IA(i), 0);
    //     SC.row(i) = C.row(IA(i));
    // }
    // // std::cout << I.segment<10>(0).transpose() << std::endl;
    // V = SV; 
    // F = UF;
    // C = SC;
    // V = SV; 
    // F = SF;
    FINISH_TIMING_PRINT(remesh);
    
    
}

void IntrinsicSimulation::generateMeshForRendering(MatrixXT& V, MatrixXi& F, MatrixXT& C)
{
    
    if (triangles.size())
    {
        VectorXT face_quantity;
        std::cout << "F " << F.rows() << std::endl;
        assignFaceColorBasedOnGeodesicTriangle(V, F, C, face_quantity);
        std::cout << "F after " << F.rows() << std::endl;
        igl::writeOBJ("big_mesh.obj", V, F);
    }
    else
    {
        vectorToIGLMatrix<int, 3>(extrinsic_indices, F);
        int n_vtx_dof = extrinsic_vertices.rows();
        if (two_way_coupling)
        {
            if (use_FEM)
            {
                V.resize(num_nodes, 3);
                for (int i = 0; i < num_nodes; i++)
                {
                    V.row(i) = deformed.segment<3>(fem_dof_start + surface_to_tet_node_map[i] * 3);
                }
            }
            else
                vectorToIGLMatrix<T, 3>(deformed.segment(deformed.rows() - n_vtx_dof, n_vtx_dof), V);
        }
        else
            vectorToIGLMatrix<T, 3>(extrinsic_vertices, V);
        C.resize(F.rows(), 3);
        C.col(0).setConstant(28.0/255.0); C.col(1).setConstant(99.0/255.0); C.col(2).setConstant(227.0/255.0);
    }
    
}

void IntrinsicSimulation::initializeTrianglePlottingScene()
{
    use_Newton = true;

    MatrixXT V; MatrixXi F;
    // igl::readOBJ("../../../Projects/DifferentiableGeodesics/data/sphere642.obj", 
    igl::readOBJ("../../../Projects/DifferentiableGeodesics/data/grid.obj", 
        V, F);

    buildMeshGeometry(V, F);

    int n_mass_point = 3;    

    intrinsic_vertices_barycentric_coords.resize(n_mass_point * 2);
    
    VectorXT mass_point_Euclidean(n_mass_point * 3);
    int valid_cnt = 0;
    
    for (int face_idx : {81, 62, 60})
    {
        T alpha = 0.2, beta = 0.5;
        TV vi = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 0] * 3);
        TV vj = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 1] * 3);
        TV vk = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 2] * 3);
        TV current = vi * alpha + vj * beta + vk * (1.0 - alpha - beta);
        
        intrinsic_vertices_barycentric_coords.segment<2>(valid_cnt * 2) = TV2(alpha, beta);
        gcs::Face f = mesh->face(face_idx);
        SurfacePoint new_pt(f, Vector3{alpha, beta, 1.0 - alpha - beta});
        new_pt = new_pt.inSomeFace();
        surface_points.push_back(new_pt);
        valid_cnt++;
    }
    // intrinsic_vertices_barycentric_coords.conservativeResize(valid_cnt * 2);
    // mass_point_Euclidean.conservativeResize(valid_cnt * 3);

    surface_points_undeformed = surface_points;
    undeformed = intrinsic_vertices_barycentric_coords;
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();
    delta_u.resize(undeformed.rows()); delta_u.setZero();

    std::cout << "add mass point" << std::endl;
    std::cout << "#dof : " << undeformed.rows() << " #points " << undeformed.rows() / 2 << std::endl;
    for (int i = 0; i < 2; i++)
    {
        dirichlet_data[i] = 0.0;
    }


    MatrixXi igl_tri, igl_edges;

    igl_edges.resize(3, 2);
    igl_edges.row(0) = Eigen::RowVector2i(0, 1);
    igl_edges.row(1) = Eigen::RowVector2i(0, 2);
    igl_edges.row(2) = Eigen::RowVector2i(1, 2);

    triangles.push_back(Triangle(0, 1, 2));

    all_intrinsic_edges.resize(0);

    connectSpringEdges(igl_edges);

    updateCurrentState();
    surface_points_undeformed = surface_points;
    verbose = true; 
    add_area_term = true;
    computeAllTriangleArea(rest_area);
    rest_area.setZero();
    wa = 1;
    we = 0; 
}

void IntrinsicSimulation::initializeTriangleDebugScene()
{
    use_Newton = true;

    MatrixXT V; MatrixXi F;
    // igl::readOBJ("../../../Projects/DifferentiableGeodesics/data/sphere642.obj", 
    igl::readOBJ("../../../Projects/DifferentiableGeodesics/data/sphere.obj", 
        V, F);

    buildMeshGeometry(V, F);

    int n_mass_point = 4;    

    intrinsic_vertices_barycentric_coords.resize(n_mass_point * 2);
    
    VectorXT mass_point_Euclidean(n_mass_point * 3);
    int valid_cnt = 0;
    
    for (int face_idx : {0, 1, 5, 6})
    {
        T alpha = 0.2, beta = 0.5;
        TV vi = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 0] * 3);
        TV vj = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 1] * 3);
        TV vk = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 2] * 3);
        TV current = vi * alpha + vj * beta + vk * (1.0 - alpha - beta);
        
        intrinsic_vertices_barycentric_coords.segment<2>(valid_cnt * 2) = TV2(alpha, beta);
        gcs::Face f = mesh->face(face_idx);
        SurfacePoint new_pt(f, Vector3{alpha, beta, 1.0 - alpha - beta});
        new_pt = new_pt.inSomeFace();
        surface_points.push_back(new_pt);
        valid_cnt++;
    }
    // intrinsic_vertices_barycentric_coords.conservativeResize(valid_cnt * 2);
    // mass_point_Euclidean.conservativeResize(valid_cnt * 3);

    surface_points_undeformed = surface_points;
    undeformed = intrinsic_vertices_barycentric_coords;
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();
    delta_u.resize(undeformed.rows()); delta_u.setZero();

    std::cout << "add mass point" << std::endl;
    std::cout << "#dof : " << undeformed.rows() << " #points " << undeformed.rows() / 2 << std::endl;
    for (int i = 0; i < 2; i++)
    {
        dirichlet_data[i] = 0.0;
    }


    MatrixXi igl_tri, igl_edges;

    igl_edges.resize(6, 2);
    igl_edges.row(0) = Eigen::RowVector2i(0, 1);
    igl_edges.row(1) = Eigen::RowVector2i(0, 2);
    igl_edges.row(2) = Eigen::RowVector2i(1, 2);
    igl_edges.row(3) = Eigen::RowVector2i(1, 3);
    igl_edges.row(4) = Eigen::RowVector2i(1, 2);
    igl_edges.row(5) = Eigen::RowVector2i(2, 3);

    triangles.push_back(Triangle(0, 1, 2));
    triangles.push_back(Triangle(1, 3, 2));

    all_intrinsic_edges.resize(0);

    connectSpringEdges(igl_edges);

    updateCurrentState();
    surface_points_undeformed = surface_points;
    verbose = true; 
    add_area_term = true;
    computeAllTriangleArea(rest_area);
    rest_area.setZero();
    wa = 1;
    we = 0;                                   
}


void IntrinsicSimulation::initializeDiscreteShell()
{
    MatrixXT V; MatrixXi F;
    
    igl::readOBJ("../../../Projects/DifferentiableGeodesics/data/grid.obj", 
        V, F);

    TV min_corner = V.colwise().minCoeff();
    TV max_corner = V.colwise().maxCoeff();
    
    TV center = 0.5 * (min_corner + max_corner);

    for (int i = 0; i < V.rows(); i++)
    {
        V.row(i) -= center;
    }
    
    buildMeshGeometry(V, F);

    two_way_coupling = true;

    if (two_way_coupling)
    {
        int n_dof = undeformed.rows();
        shell_dof_start = n_dof;
        undeformed.conservativeResize(n_dof + extrinsic_vertices.rows());
        undeformed.segment(n_dof, extrinsic_vertices.rows()) = extrinsic_vertices;
        deformed = undeformed;
        u.resize(undeformed.rows()); u.setZero();
        delta_u.resize(undeformed.rows()); delta_u.setZero();
        faces = extrinsic_indices;
        computeRestShape();
        buildHingeStructure();

        E_shell = 1e4;
        updateShellLameParameters();
    }

    computeAllTriangleArea(rest_area);
    undeformed_area = rest_area;
    rest_area.setZero();
    wa = 0.0;
    // we = 0;                             
}

void IntrinsicSimulation::initializeSceneEuclideanDistance(int test_case)
{
    use_Newton = true;
    MatrixXT V; MatrixXi F;
    
    igl::readOBJ("../../../Projects/DifferentiableGeodesics/data/grid.obj", 
        V, F);
    TV min_corner = V.colwise().minCoeff();
    TV max_corner = V.colwise().maxCoeff();
    TV center = 0.5 * (min_corner + max_corner);


    for (int i = 0; i < V.rows(); i++)
    {
        V.row(i) -= center;
        std::swap(V(i, 1), V(i, 2));
    }

    for (int i = 0; i < F.rows(); i++)
    {
        std::swap(F(i, 0), F(i, 1));
    }

    for (int row = 4; row < 104; row += 10)
    {
        V(row, 1) += 0.05;
    }
    if (test_case == 2)
    {
        for (int row = 3; row < 103; row += 10)
        {
            V(row, 0) -= 0.033;
        }
    }


    buildMeshGeometry(V, F);

    int n_faces = 3;    

    intrinsic_vertices_barycentric_coords.resize(n_faces * 2);
    
    VectorXT mass_point_Euclidean(n_faces * 3);
    int valid_cnt = 0;
    std::vector<int> face_indices;
    if (test_case == 0)
        face_indices = {48, 39, 41};
    else if (test_case == 1)
        face_indices = {48, 39, 46};
    else if (test_case == 2)
        face_indices = {48, 39, 41};
    // 1, 5 is a discontinuity
    // for (int face_idx : ) //right
    for (int face_idx : face_indices) //left
    {
        // T alpha = 1.0/3.0, beta = 1.0/3.0;
        T alpha = 1.0/2.0, beta = 1.0/2.0;
        TV vi = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 0] * 3);
        TV vj = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 1] * 3);
        TV vk = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 2] * 3);
        TV current = vi * alpha + vj * beta + vk * (1.0 - alpha - beta);
        
        intrinsic_vertices_barycentric_coords.segment<2>(valid_cnt * 2) = TV2(alpha, beta);
        gcs::Face f = mesh->face(face_idx);
        SurfacePoint new_pt(f, Vector3{alpha, beta, 1.0 - alpha - beta});
        new_pt = new_pt.inSomeFace();
        surface_points.push_back(new_pt);
        valid_cnt++;
    }

    surface_points_undeformed = surface_points;
    undeformed = intrinsic_vertices_barycentric_coords;
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();
    delta_u.resize(undeformed.rows()); delta_u.setZero();

    std::cout << "add mass point" << std::endl;
    std::cout << "#dof : " << undeformed.rows() << " #points " << undeformed.rows() / 2 << std::endl;
    for (int i = 0; i < 4; i++)
    {
        dirichlet_data[i] = 0.0;
    }


    MatrixXi igl_tri, igl_edges;

    igl_edges.resize(2, 2);
    igl_edges.row(0) = Eigen::RowVector2i(0, 2);
    igl_edges.row(1) = Eigen::RowVector2i(2, 1);


    all_intrinsic_edges.resize(0);

    connectSpringEdges(igl_edges);
    updateCurrentState();
    ref_dis = 0.1;
    add_length_term = true;
    add_area_term = false;
    add_geo_elasticity = false;
    add_volume = false;
    two_way_coupling = false;

}


void IntrinsicSimulation::initializeTeaserScene(const std::string& folder)
{
    use_Newton = true;
    MatrixXT V; MatrixXi F;
    std::string mesh_name = "bunny";
    igl::readOBJ(folder + "/data/"+mesh_name+".obj", V, F);
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
    
    mesh_center = center;
    mesh_scale = scale;

    buildMeshGeometry(V, F);

    SurfacePoint source = SurfacePoint(mesh->face(6030), Vector3{0.2,0.4,0.4});

    gcs::GeodesicAlgorithmExact mmp(*mesh, *geometry);
    mmp.propagate({source});

    gcs::VertexData<T> distToSource = mmp.getDistanceFunction();
    VectorXT distances(mesh->nVertices());
    for (gcVertex vtx : mesh->vertices())
    {
        T dis = distToSource[vtx];
        distances[vtx.getIndex()] = dis;
    }
    FILE *stream;
    if ((stream = fopen((folder + "/results/teaser/"+mesh_name+"_distance_field.data").c_str(), "wb")) != NULL)
    {
        int len = distances.rows();
        size_t temp;
        temp = fwrite(&len, sizeof(int), 1, stream);
        temp = fwrite(distances.data(), sizeof(double), len, stream);
    }
    else 
    {
        std::cout << "Unable to write into file " << (folder + "/results/teaser/"+mesh_name+"_distance_field.data").c_str() << std::endl;
    }
    fclose(stream);
    
    int n_faces = 2;    

    intrinsic_vertices_barycentric_coords.resize(n_faces * 2);
    
    VectorXT mass_point_Euclidean(n_faces * 3);
    int valid_cnt = 0;
    std::vector<int> face_list = {6030, 738};
    
    for (int face_idx : face_list) //screwdriver
    {
        T alpha = 0.2, beta = 0.4;
        
        TV vi = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 0] * 3);
        TV vj = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 1] * 3);
        TV vk = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 2] * 3);
        TV current = vi * alpha + vj * beta + vk * (1.0 - alpha - beta);
        
        intrinsic_vertices_barycentric_coords.segment<2>(valid_cnt * 2) = TV2(alpha, beta);
        gcs::Face f = mesh->face(face_idx);
        SurfacePoint new_pt(f, Vector3{alpha, beta, 1.0 - alpha - beta});
        new_pt = new_pt.inSomeFace();
        surface_points.push_back(new_pt);
        valid_cnt++;
    }

    surface_points_undeformed = surface_points;
    undeformed = intrinsic_vertices_barycentric_coords;
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();
    delta_u.resize(undeformed.rows()); delta_u.setZero();

    std::cout << "add mass point" << std::endl;
    std::cout << "#dof : " << undeformed.rows() << " #points " << undeformed.rows() / 2 << std::endl;
    for (int i = 0; i < 2; i++)
    {
        dirichlet_data[i] = 0.0;
    }


    MatrixXi igl_edges;

    igl_edges.resize(1, 2);
    igl_edges.row(0) = Eigen::RowVector2i(0, 1);
    

    all_intrinsic_edges.resize(0);

    connectSpringEdges(igl_edges);

    add_area_term = false;
    add_geo_elasticity = false;
    add_volume = false;

    two_way_coupling = false;
}

void IntrinsicSimulation::initializeTorusExpandingScene()
{
    std::string base_folder = 
        "../../../Projects/DifferentiableGeodesics/data/";
    std::string mesh_name = "torus_dense";
    
    use_Newton = true;
    MatrixXT V; MatrixXi F;
    
    igl::readOBJ(base_folder + mesh_name + ".obj", V, F);

    std::cout << "#Triangles " << F.rows() << std::endl;
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
    
    mesh_center = center;
    mesh_scale = scale;

    buildMeshGeometry(V, F);

    int n_faces = extrinsic_indices.rows()/ 3;    

    intrinsic_vertices_barycentric_coords.resize(n_faces * 2);
    
    std::vector<TV> mass_point_vec;
    std::vector<TV2> bary_vec;

    VectorXT mass_point_Euclidean(n_faces * 3);
    std::vector<Edge> edges;
    std::vector<SurfacePoint> samples;
    std::ifstream in(base_folder + mesh_name + "_samples.txt");
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
    in.open(base_folder + mesh_name + "_idt.txt");
    int n_tri; in >> n_tri;
    MatrixXi idt_tri(n_tri, 3);
    for (int i = 0; i < n_tri; i++)
    {
        int a, b, c;
        in >> a >> b >> c;
        triangles.push_back(IV(a, b, c));
        idt_tri.row(i) = IV(a, b, c);
    }
    in.close();
    MatrixXi idt_edges;
    igl::edges(idt_tri, idt_edges);
    for (int i = 0; i < idt_edges.rows(); i++)
    {
        edges.push_back(idt_edges.row(i));
    }
    for (SurfacePoint& pt : samples)
    {
        pt = pt.inSomeFace();
        mass_point_vec.push_back(toTV(pt.interpolate(geometry->vertexPositions)));
        bary_vec.push_back(TV2(pt.faceCoords.x, pt.faceCoords.y));
        surface_points.push_back(pt);
    }

    mass_point_Euclidean.resize(mass_point_vec.size() * 3);
    for (int i = 0; i < mass_point_vec.size(); i++)
        mass_point_Euclidean.segment<3>(i * 3) = mass_point_vec[i];
    intrinsic_vertices_barycentric_coords.resize(bary_vec.size() * 2);
    for (int i = 0; i < bary_vec.size(); i++)
        intrinsic_vertices_barycentric_coords.segment<2>(i * 2) = bary_vec[i];

    surface_points_undeformed = surface_points;
    undeformed = intrinsic_vertices_barycentric_coords;
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();
    delta_u.resize(undeformed.rows()); delta_u.setZero();

    // std::cout << "add mass point" << std::endl;
    std::cout << "#dof : " << undeformed.rows() << " #points " << undeformed.rows() / 2 << std::endl;

    dirichlet_vertices = {0};
    
    for (int idx : dirichlet_vertices)
    {
        dirichlet_data[idx * 2 + 0] = 0.0;
        dirichlet_data[idx * 2 + 1] = 0.0;
    }

    add_length_term = false;
    add_area_term = false;
    add_geo_elasticity = true;
    add_volume = false;
    all_intrinsic_edges.resize(0);

    connectSpringEdges(edges);

    // computeAllTriangleArea(rest_area);
    // rest_area *= 0.0;
    // undeformed_area = rest_area;
    // wa = 1.0;
    
    use_reference_C_tensor = false;
    reference_C_entries = TV(0.25,0.0,0.25);
    E = 1e0;
    nu = 0.3;
    updateLameParameters();
    // computeGeodesicTriangleRestShape();

    verbose = false;

}

void IntrinsicSimulation::initializeKarcherMeanSceneMancinellPuppo(const std::vector<int>& vtx_list)
{
    use_Newton = true;
    MatrixXT V; MatrixXi F;
    
    igl::readOBJ("../../../Projects/DifferentiableGeodesics/data/sphere642.obj", V, F);
    // TV min_corner = V.colwise().minCoeff();
    // TV max_corner = V.colwise().maxCoeff();
    // TV center = 0.5 * (min_corner + max_corner);
    // // std::cout << min_corner.transpose() << " " << max_corner.transpose() << std::endl;
    // T bb_diag = (max_corner - min_corner).norm();
    // T scale = 1.0 / bb_diag;
    // for (int i = 0; i < V.rows(); i++)
    // {
    //     V.row(i) -= center;
    // }
    // V *= scale;
    

    buildMeshGeometry(V, F);

    int n_faces = 6;    

    intrinsic_vertices_barycentric_coords.resize(n_faces * 2);
    
    VectorXT mass_point_Euclidean(n_faces * 3);
    int valid_cnt = 0;

    // for (int face_idx : face_list) //screwdriver
    // {
    //     T alpha = 1.0, beta = 0.0;
    //     // if (valid_cnt == 5)
    //     // {
    //     //     alpha = 0.3;
    //     //     beta = 0.3;
    //     // }
    //     // std::cout << face_idx << std::endl;
        
    //     intrinsic_vertices_barycentric_coords.segment<2>(valid_cnt * 2) = TV2(alpha, beta);
    //     gcs::Face f = mesh->face(face_idx);
    //     // SurfacePoint new_pt(f, Vector3{1.0 - alpha - beta, alpha, beta}); // to match their code
    //     SurfacePoint new_pt(f, Vector3{beta, 1.0 - alpha - beta, alpha}); // to match their code
    //     new_pt = new_pt.inSomeFace();
    //     surface_points.push_back(new_pt);
    //     valid_cnt++;
    // }

    for (int vtx_idx : vtx_list)
    {
        intrinsic_vertices_barycentric_coords.segment<2>(valid_cnt * 2) = TV2(1, 0);
        SurfacePoint pt = mesh->vertex(vtx_idx);
        surface_points.push_back(pt.inSomeFace());
    }
    

    surface_points.back().faceCoords = Vector3{0.5, 0.3, 0.2};
    surface_points.back() = surface_points.back().inSomeFace();

    surface_points_undeformed = surface_points;
    undeformed = intrinsic_vertices_barycentric_coords;
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();
    delta_u.resize(undeformed.rows()); delta_u.setZero();

    // std::cout << "add mass point" << std::endl;
    // std::cout << "#dof : " << undeformed.rows() << " #points " << undeformed.rows() / 2 << std::endl;
    for (int i = 0; i < 10; i++)
    {
        dirichlet_data[i] = 0.0;
    }


    MatrixXi igl_tri, igl_edges;

    igl_edges.resize(5, 2);
    for (int i = 0; i < 5; i++)
    {
        igl_edges.row(i) = Eigen::RowVector2i(i, 5);    
    }
    

    all_intrinsic_edges.resize(0);
    // START_TIMING(connect_geodesics)
    connectSpringEdges(igl_edges);
    // FINISH_TIMING_PRINT(connect_geodesics)
    add_area_term = false;
    add_geo_elasticity = false;
    add_volume = false;

    two_way_coupling = false;
}

void IntrinsicSimulation::initializeKarcherMeanScene(const std::string& filename, int config)
{
    use_Newton = true;
    MatrixXT V; MatrixXi F;
    
    igl::readOBJ(filename, V, F);
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
    

    buildMeshGeometry(V, F);

    int n_faces = 6;    

    intrinsic_vertices_barycentric_coords.resize(n_faces * 2);
    
    VectorXT mass_point_Euclidean(n_faces * 3);
    int valid_cnt = 0;
    std::vector<int> face_list;
    if (config == 0)
        face_list = {592, 576, 611, 516, 551, 1030}; //sphere
    else if (config == 1)
        face_list = {1576, 1273, 1434, 2052, 2419, 6271}; //screwdriver
    else if (config == 2)
        face_list = {9013, 9184, 21897, 2630, 7755, 16429}; // complex geometry
    else if (config == 3)
        face_list = {2120, 2857, 3315, 2522, 1862, 9310}; // ear_simplified
    else if (config == 4)
        face_list = {23666, 23602, 20134, 17521, 11818, 42996}; // protein
    else if (config == 5)
        face_list = {5174, 8568, 13088, 12635, 8576, 36476}; // ear
    else if (config == 6)
        // face_list = {38, 3677, 3806, 3695, 4452, 11785}; // bunny
        // face_list = {38, 3677, 3806, 3695, 4452, 842}; // bunny
        face_list = {38, 3677, 3806, 3695, 4452, 11107}; // bunny

    for (int face_idx : face_list) //screwdriver
    {
        T alpha = 1.0/3.0, beta = 1.0/3.0;
        if (valid_cnt < 5)
        {
            // alpha = 1.0; beta = 0.0;
        }
        // T alpha = 1.0/2.0, beta = 1.0/2.0;
        TV vi = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 0] * 3);
        TV vj = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 1] * 3);
        TV vk = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 2] * 3);
        TV current = vi * alpha + vj * beta + vk * (1.0 - alpha - beta);
        
        intrinsic_vertices_barycentric_coords.segment<2>(valid_cnt * 2) = TV2(alpha, beta);
        gcs::Face f = mesh->face(face_idx);
        SurfacePoint new_pt(f, Vector3{alpha, beta, 1.0 - alpha - beta});
        new_pt = new_pt.inSomeFace();
        surface_points.push_back(new_pt);
        valid_cnt++;
    }

    surface_points_undeformed = surface_points;
    undeformed = intrinsic_vertices_barycentric_coords;
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();
    delta_u.resize(undeformed.rows()); delta_u.setZero();

    std::cout << "add mass point" << std::endl;
    std::cout << "#dof : " << undeformed.rows() << " #points " << undeformed.rows() / 2 << std::endl;
    for (int i = 0; i < 10; i++)
    {
        dirichlet_data[i] = 0.0;
    }


    MatrixXi igl_tri, igl_edges;

    igl_edges.resize(5, 2);
    for (int i = 0; i < 5; i++)
    {
        igl_edges.row(i) = Eigen::RowVector2i(i, 5);    
    }
    

    all_intrinsic_edges.resize(0);
    // START_TIMING(connect_geodesics)
    connectSpringEdges(igl_edges);
    // FINISH_TIMING_PRINT(connect_geodesics)
    add_area_term = false;
    add_geo_elasticity = false;
    add_volume = false;

    two_way_coupling = false;

}

void IntrinsicSimulation::connectSpringEdges(const std::vector<Edge>& edges)
{
    int n_springs = edges.size();

    std::vector<std::vector<std::pair<TV, TV>>> 
        sub_pairs(n_springs, std::vector<std::pair<TV, TV>>());
    rest_length.resize(n_springs);
    spring_edges.resize(n_springs);
    VectorXT ref_lengths(n_springs);
#ifdef PARALLEL_GEODESIC
    tbb::parallel_for(0, n_springs, [&](int i)
#else
    for (int i = 0; i < n_springs; i++)
#endif
    {
        SurfacePoint vA = surface_points[edges[i][0]];
        SurfacePoint vB = surface_points[edges[i][1]];
        spring_edges[i] = Edge(edges[i][0], edges[i][1]);

        T geo_dis; std::vector<SurfacePoint> path;
        std::vector<IxnData> ixn_data;
        // START_TIMING(compute_geodesic)
        computeExactGeodesic(vA, vB, geo_dis, path, ixn_data, true);
        // computeExactGeodesicEdgeFlip(vA, vB, geo_dis, path, ixn_data, true);
        // FINISH_TIMING_PRINT(compute_geodesic)
        // std::getchar();
        // computeGeodesicHeatMethod(vA, vB, geo_dis, path, ixn_data, true);
        // std::cout << "geodesic " << geo_dis << std::endl;
        rest_length[i] = geo_dis;
        ref_lengths[i] = geo_dis;
        for(int j = 0; j < path.size() - 1; j++)
        {
            sub_pairs[i].emplace_back(toTV(path[j].interpolate(geometry->vertexPositions)),
                toTV(path[j+1].interpolate(geometry->vertexPositions)));
        }
    }
#ifdef PARALLEL_GEODESIC
    );
#endif

    // ref_dis = ref_lengths.sum() / ref_lengths.rows();
    // ref_dis *= 0.3;

    current_length = rest_length;
    for (int i = 0; i < n_springs; i++)
    {
        edge_map[spring_edges[i]] = i;
        edge_map[Edge(spring_edges[i][1], spring_edges[i][0])] = i;
    }
    undeformed_length.resize(current_length.size());
    for (int i = 0; i < current_length.size(); i++)
        undeformed_length[i] = 1.0 * current_length[i];

    for (int i = 0; i < current_length.size(); i++)
    {
        // T rand_float = static_cast <T> (rand()) / static_cast <T> (RAND_MAX);
        // rest_length[i] *= rand_float;
        // rest_length[i] *= 0.0;
        // if (i%2==0)
        //     rest_length[i] *= 0.3;
        // else
        //     rest_length[i] *= 0.9;
        rest_length[i] *= 0.0;
    }
}
void IntrinsicSimulation::connectSpringEdges(const MatrixXi& igl_edges)
{
    int n_springs = igl_edges.rows();
    std::vector<std::vector<std::pair<TV, TV>>> 
        sub_pairs(n_springs, std::vector<std::pair<TV, TV>>());
    rest_length.resize(n_springs);
    spring_edges.resize(n_springs);
#ifdef PARALLEL_GEODESIC
    tbb::parallel_for(0, n_springs, [&](int i)
#else
    for (int i = 0; i < n_springs; i++)
#endif
    {
        SurfacePoint vA = surface_points[igl_edges(i, 0)];
        SurfacePoint vB = surface_points[igl_edges(i, 1)];
        spring_edges[i] = Edge(igl_edges(i, 0), igl_edges(i, 1));

        T geo_dis; std::vector<SurfacePoint> path;
        std::vector<IxnData> ixn_data;
        computeExactGeodesic(vA, vB, geo_dis, path, ixn_data, true);
        // computeExactGeodesicEdgeFlip(vA, vB, geo_dis, path, ixn_data, true);
        rest_length[i] = geo_dis;
        for(int j = 0; j < path.size() - 1; j++)
        {
            sub_pairs[i].emplace_back(toTV(path[j].interpolate(geometry->vertexPositions)),
                toTV(path[j+1].interpolate(geometry->vertexPositions)));
        }
    }
#ifdef PARALLEL_GEODESIC
    );
#endif
    for (int i = 0; i < igl_edges.rows(); i++)
    {
        all_intrinsic_edges.insert(all_intrinsic_edges.end(), sub_pairs[i].begin(), sub_pairs[i].end());
    }

    current_length = rest_length;
    for (int i = 0; i < n_springs; i++)
    {
        edge_map[spring_edges[i]] = i;
        edge_map[Edge(spring_edges[i][1], spring_edges[i][0])] = i;
    }
    undeformed_length.resize(current_length.size());
    for (int i = 0; i < current_length.size(); i++)
        undeformed_length[i] = current_length[i];
    for (int i = 0; i < current_length.size(); i++)
    {
        rest_length[i] = 0.0;
    }
}

void IntrinsicSimulation::initializeSceneCheckingSmoothness()
{
    use_Newton = true;
    MatrixXT V; MatrixXi F;
    
    igl::readOBJ("../../../Projects/DifferentiableGeodesics/data/grid.obj", 
        V, F);

    V(44, 2) -= 0.01;

    for (int row : {11, 22, 33, 55, 66, 77, 88})
    {
        V(row, 2) -= 0.02;
    }

    buildMeshGeometry(V, F);

    // int n_faces = extrinsic_indices.rows()/ 3 / 10;  
    // int n_faces = extrinsic_indices.rows()/ 3;    
    int n_faces = 6;    

    intrinsic_vertices_barycentric_coords.resize(n_faces * 2);
    
    VectorXT mass_point_Euclidean(n_faces * 3);
    int valid_cnt = 0;
    // 1, 5 is a discontinuity
    // for (int face_idx : {67, 27, 91, 131, 42, 82})
    for (int face_idx : {102, 119, 22, 39, 42, 59})
    // for (int face_idx : {3, 5})
    {
        T alpha = 1.0/3.0, beta = 1.0/3.0;
        TV vi = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 0] * 3);
        TV vj = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 1] * 3);
        TV vk = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 2] * 3);
        TV current = vi * alpha + vj * beta + vk * (1.0 - alpha - beta);
        
        intrinsic_vertices_barycentric_coords.segment<2>(valid_cnt * 2) = TV2(alpha, beta);
        gcs::Face f = mesh->face(face_idx);
        SurfacePoint new_pt(f, Vector3{alpha, beta, 1.0 - alpha - beta});
        new_pt = new_pt.inSomeFace();
        surface_points.push_back(new_pt);
        valid_cnt++;
    }
    // intrinsic_vertices_barycentric_coords.conservativeResize(valid_cnt * 2);
    // mass_point_Euclidean.conservativeResize(valid_cnt * 3);

    surface_points_undeformed = surface_points;
    undeformed = intrinsic_vertices_barycentric_coords;
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();
    delta_u.resize(undeformed.rows()); delta_u.setZero();

    std::cout << "add mass point" << std::endl;
    std::cout << "#dof : " << undeformed.rows() << " #points " << undeformed.rows() / 2 << std::endl;
    for (int i = 0; i < 8; i++)
    {
        dirichlet_data[i] = 0.0;
    }


    MatrixXi igl_tri, igl_edges;

    igl_edges.resize(7, 2);
    igl_edges.row(0) = Eigen::RowVector2i(0, 1);
    igl_edges.row(1) = Eigen::RowVector2i(2, 3);
    igl_edges.row(2) = Eigen::RowVector2i(4, 5);
    igl_edges.row(3) = Eigen::RowVector2i(0, 4);
    igl_edges.row(4) = Eigen::RowVector2i(4, 2);
    igl_edges.row(5) = Eigen::RowVector2i(1, 5);
    igl_edges.row(6) = Eigen::RowVector2i(5, 3);


    all_intrinsic_edges.resize(0);
    connectSpringEdges(igl_edges);

    add_area_term = false;
    add_geo_elasticity = false;
    add_volume = false;

    two_way_coupling = true;

    if (two_way_coupling)
    {
        int n_dof = undeformed.rows();
        shell_dof_start = n_dof;
        undeformed.conservativeResize(n_dof + extrinsic_vertices.rows());
        undeformed.segment(n_dof, extrinsic_vertices.rows()) = extrinsic_vertices;
        deformed = undeformed;
        u.resize(undeformed.rows()); u.setZero();
        delta_u.resize(undeformed.rows()); delta_u.setZero();
        faces = extrinsic_indices;
        computeRestShape();
        buildHingeStructure();

        E_shell = 1e4;
        updateShellLameParameters();
    }

    computeAllTriangleArea(rest_area);
    undeformed_area = rest_area;
    rest_area.setZero();
    wa = 0.0;
    // we = 0;
    
    updateCurrentState();
    surface_points_undeformed = surface_points;
    verbose = false;

    
    // geometry->requireVertexGaussianCurvatures();
    // for (int i = 0; i < n_vtx_extrinsic; i++)
    // {
    //     T ki = geometry->vertexGaussianCurvature(mesh->vertex(i));
    //     if (std::abs(ki) > 1e-6)
    //         std::cout << "vtx "<< i << " K: " << ki << std::endl;
    // }
    // geometry->unrequireVertexGaussianCurvatures();

    if (two_way_coupling)
        for (int idx : {0, 9, 90, 99})
        {
            for (int d = 0; d < 3; d++)
            {
                dirichlet_data[idx * 3 + d + shell_dof_start] = 0.0;
            }
        }
    
    use_t_wrapper = true;
}

void IntrinsicSimulation::initializePlottingScene(std::string mesh_file, int exp_idx)
{
    use_Newton = true;
    two_way_coupling = false;

    MatrixXT V; MatrixXi F;
    // igl::readOBJ("../../../Projects/DifferentiableGeodesics/data/sphere642.obj", 
    igl::readOBJ(mesh_file, 
        V, F);
    // V *= 0.005;
    // V *= 5.0;

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
    
    buildMeshGeometry(V, F);
    int n_faces = 2;    

    intrinsic_vertices_barycentric_coords.resize(n_faces * 2);
    
    VectorXT mass_point_Euclidean(n_faces * 3);
    int valid_cnt = 0;
    std::vector<int> face_list;
    if (exp_idx == 3)
        face_list = {17927, 19757};
    else if (exp_idx == 2)
        face_list = {5, 3};
    else
        face_list = {5, 2};
    for (int face_idx : face_list) 
    {
        // T alpha = 1.0/3.0, beta = 1.0/3.0;
        T alpha = 0.33, beta = 0.33;
        TV vi = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 0] * 3);
        TV vj = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 1] * 3);
        TV vk = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 2] * 3);
        TV current = vi * alpha + vj * beta + vk * (1.0 - alpha - beta);
        
        intrinsic_vertices_barycentric_coords.segment<2>(valid_cnt * 2) = TV2(alpha, beta);
        gcs::Face f = mesh->face(face_idx);
        SurfacePoint new_pt(f, Vector3{alpha, beta, 1.0 - alpha - beta});
        new_pt = new_pt.inSomeFace();
        surface_points.push_back(new_pt);
        valid_cnt++;
    }

    surface_points_undeformed = surface_points;
    undeformed = intrinsic_vertices_barycentric_coords;
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();
    delta_u.resize(undeformed.rows()); delta_u.setZero();

    for (int i = 0; i < 2; i++)
    {
        dirichlet_data[i] = 0.0;
    }

    MatrixXi igl_edges;

    igl_edges.resize(1, 2);
    igl_edges.row(0) = Eigen::RowVector2i(0, 1);

    all_intrinsic_edges.resize(0);
    connectSpringEdges(igl_edges);

    updateCurrentState();
    verbose = false;
    
    ref_dis = 0.1;
    add_area_term = false;
    add_geo_elasticity = false;
    add_volume = false;
    add_length_term = true;
    we = 1.0;
    
}

void IntrinsicSimulation::initializeMassSpringDebugScene()
{
    use_Newton = true;
    MatrixXT V; MatrixXi F;
    // igl::readOBJ("../../../Projects/DifferentiableGeodesics/data/sphere642.obj", 
    igl::readOBJ("../../../Projects/DifferentiableGeodesics/data/sphere.obj", 
        V, F);

    buildMeshGeometry(V, F);

    // int n_faces = extrinsic_indices.rows()/ 3 / 10;  
    // int n_faces = extrinsic_indices.rows()/ 3;    
    int n_faces = 2;    

    intrinsic_vertices_barycentric_coords.resize(n_faces * 2);
    
    VectorXT mass_point_Euclidean(n_faces * 3);
    int valid_cnt = 0;
    // 1, 5 is a discontinuity
    for (int face_idx : {1, 2})
    {
        T alpha = 0.2, beta = 0.5;
        TV vi = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 0] * 3);
        TV vj = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 1] * 3);
        TV vk = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 2] * 3);
        TV current = vi * alpha + vj * beta + vk * (1.0 - alpha - beta);
        
        intrinsic_vertices_barycentric_coords.segment<2>(valid_cnt * 2) = TV2(alpha, beta);
        gcs::Face f = mesh->face(face_idx);
        SurfacePoint new_pt(f, Vector3{alpha, beta, 1.0 - alpha - beta});
        new_pt = new_pt.inSomeFace();
        surface_points.push_back(new_pt);
        valid_cnt++;
    }
    // intrinsic_vertices_barycentric_coords.conservativeResize(valid_cnt * 2);
    // mass_point_Euclidean.conservativeResize(valid_cnt * 3);

    surface_points_undeformed = surface_points;
    undeformed = intrinsic_vertices_barycentric_coords;
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();
    delta_u.resize(undeformed.rows()); delta_u.setZero();

    std::cout << "add mass point" << std::endl;
    std::cout << "#dof : " << undeformed.rows() << " #points " << undeformed.rows() / 2 << std::endl;
    for (int i = 0; i < 2; i++)
    {
        dirichlet_data[i] = 0.0;
    }


    MatrixXi igl_tri, igl_edges;

    igl_edges.resize(1, 2);
    igl_edges.row(0) = Eigen::RowVector2i(0, 1);

    all_intrinsic_edges.resize(0);
    connectSpringEdges(igl_edges);

    two_way_coupling = true;
 
    if (two_way_coupling)
    {
        int n_dof = undeformed.rows();
        shell_dof_start = n_dof;
        undeformed.conservativeResize(n_dof + extrinsic_vertices.rows());
        undeformed.segment(n_dof, extrinsic_vertices.rows()) = extrinsic_vertices;
        deformed = undeformed;
        u.resize(undeformed.rows()); u.setZero();
        delta_u.resize(undeformed.rows()); delta_u.setZero();
        faces = extrinsic_indices;
        computeRestShape();
        buildHingeStructure();

        E_shell = 0;
        updateShellLameParameters();
    }

    computeAllTriangleArea(rest_area);
    undeformed_area = rest_area;
    rest_area.setZero();
    wa = 0.0;
    // we = 0;

    add_geo_elasticity = false;
    if (add_geo_elasticity)
    {
        E = 1e6;
        updateLameParameters();
        computeGeodesicTriangleRestShape();
        
    }

    add_volume = false && two_way_coupling;
    if (add_volume)
    {
        rest_volume = computeVolume();
        wv = 1e3;
        woodbury = false;
    }
    // std::cout << "rest volume " << rest_volume << std::endl;
    
    updateCurrentState();
    surface_points_undeformed = surface_points;
    verbose = false;

    
}


void IntrinsicSimulation::initializeNetworkData(const std::vector<Edge>& edges)
{
    dirichlet_data.clear();
    for (int idx : dirichlet_vertices)
    {
        dirichlet_data[idx * 2 + 0] = 0.0;
        dirichlet_data[idx * 2 + 1] = 0.0;
    }
    
    if (two_way_coupling)
    {       
        faces = extrinsic_indices;
        computeRestShape();
        buildHingeStructure();

        E_shell = 1e4;
        updateShellLameParameters();

        for (int d = 0; d < 3; d++)
        {
            dirichlet_data[shell_dof_start + d] = 0.0;
        }
    }

    all_intrinsic_edges.resize(0);

    START_TIMING(connect_geodesics)
    connectSpringEdges(edges);
    FINISH_TIMING_PRINT(connect_geodesics)

    if (add_area_term)
    {
        computeAllTriangleArea(rest_area);
        undeformed_area = rest_area;
        rest_area.setZero();
        wa = 1e2;
    }
    // we = 0;

    // add_geo_elasticity = false;
    if (add_geo_elasticity)
    {
        E = 1e2;
        updateLameParameters();
        computeGeodesicTriangleRestShape();
        
    }

    
    if (add_volume)
    {
        rest_volume = computeVolume(true);
        wv = 1e3;
        woodbury = true;
    }
    // std::cout << "rest volume " << rest_volume << std::endl;
    
    updateCurrentState();
    surface_points_undeformed = surface_points;
    verbose = false;
}

void IntrinsicSimulation::initializeTwowayCouplingScene()
{
    use_Newton = true;
    two_way_coupling = true;
    use_FEM = true;

    std::string base_folder = "../../../Projects/DifferentiableGeodesics/data/";
    std::string mesh_name = "bunny";
    // std::string mesh_name = "donut_duck";
    MatrixXT V, Vtet; MatrixXi F, Ftet, Ttet; VectorXi tri_flag, tet_flag;
    // igl::readOBJ(base_folder + mesh_name + "_sf.obj", V, F);
    igl::readMSH(base_folder + mesh_name + ".msh", Vtet, Ftet, Ttet, tri_flag, tet_flag);

    MatrixXi surface_F;
    igl::boundary_facets(Ttet, surface_F);

    for (int i = 0; i < surface_F.rows(); i++)
    {
        std::swap(surface_F(i, 1), surface_F(i, 2));
    }
    

    // igl::writeOBJ("./surface.obj", Vtet, surface_F);
    // std::cout << surface_F.maxCoeff() << " " <<  surface_F.minCoeff() << " " << V.rows() << std::endl;

    std::unordered_set<int> unique_vtx;
    for (int i = 0; i < surface_F.rows(); i++)
        for (int d = 0; d < 3; d++)
        {
            unique_vtx.insert(surface_F(i, d));
        }
    int n_vtx = unique_vtx.size();
    std::cout << n_vtx << std::endl;
    
    int cnt = 0;
    for (auto idx : unique_vtx)
    {
        surface_to_tet_node_map[cnt] = idx;
        tet_node_surface_map[idx] = cnt++;

    }
    F.resize(surface_F.rows(), 3);
    for (int i = 0; i < F.rows(); i++)
        for (int d = 0; d < 3; d++)
        {
            F(i, d) = tet_node_surface_map[surface_F(i, d)];
        }
    V.resize(n_vtx, 3);
    cnt = 0;
    for (auto idx : unique_vtx)
        V.row(cnt++) = Vtet.row(idx);
    // igl::writeOBJ("./surface_buggy.obj", V, F);

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

    for (int i = 0; i < Vtet.rows(); i++)
    {
        Vtet.row(i) -= center;
    }
    Vtet *= scale;

    
    mesh_center = center;
    mesh_scale = scale;
    min_corner = V.colwise().minCoeff();
    max_corner = V.colwise().maxCoeff();
    std::cout << max_corner.transpose() << " " << min_corner.transpose() << std::endl;

    buildMeshGeometry(V, F);

    std::cout << "#Triangles " << F.rows() << std::endl;

    bool load_scene = true;
    MatrixXi igl_edges;
    std::vector<Edge> edges;
    if (load_scene)
    {
        std::vector<SurfacePoint> samples;
        
        std::ifstream in(base_folder + mesh_name + "_samples.txt");
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
        for (SurfacePoint& pt : samples)
        {
            pt = pt.inSomeFace();
            surface_points.push_back(pt);
        }
        in.open(base_folder + mesh_name + "_idt.txt");
        int n_tri; in >> n_tri;
        MatrixXi idt_tri(n_tri, 3);
        for (int i = 0; i < n_tri; i++)
        {
            int a, b, c;
            in >> a >> b >> c;
            // triangles.push_back(IV(a, b, c));
            idt_tri.row(i) = IV(a, b, c);
        }
        in.close();

        MatrixXi idt_edges;
        igl::edges(idt_tri, idt_edges);
        for (int i = 0; i < idt_edges.rows(); i++)
        {
            edges.push_back(idt_edges.row(i));
        }
        surface_points.emplace_back(mesh->face(583), Vector3{0.437417, 0.289199, 0.273384});
        surface_points.emplace_back(mesh->face(1550), Vector3{0.344861, 0.375581, 0.279558});
        surface_points.emplace_back(mesh->face(1264), Vector3{0.353802, 0.216634, 0.429565});
        surface_points.emplace_back(mesh->face(1262), Vector3{0.436999, 0.372122, 0.190879});
        surface_points.emplace_back(mesh->face(4951), Vector3{0.382904, 0.402571, 0.214525});
        surface_points.emplace_back(mesh->face(4010), Vector3{0.319918, 0.198149, 0.481933});
        surface_points.emplace_back(mesh->face(4302), Vector3{0.36393, 0.309998, 0.326072});
        surface_points.emplace_back(mesh->face(3383), Vector3{0.210512, 0.47095, 0.318538});
        surface_points.emplace_back(mesh->face(4120), Vector3{0.24546, 0.183417, 0.571123});
        surface_points.emplace_back(mesh->face(4758), Vector3{0.343174, 0.317879, 0.338947});
        surface_points.emplace_back(mesh->face(5183), Vector3{0.268349, 0.439763, 0.291888});
        surface_points.emplace_back(mesh->face(5996), Vector3{0.221266, 0.404115, 0.374619});
        surface_points.emplace_back(mesh->face(2021), Vector3{0.246703, 0.428905, 0.324392});
        surface_points.emplace_back(mesh->face(1267), Vector3{0.191395, 0.448974, 0.359631});
        surface_points.emplace_back(mesh->face(1643), Vector3{0.23109, 0.387695, 0.381215});
        surface_points.emplace_back(mesh->face(9333), Vector3{0.259224, 0.475223, 0.265552});
        surface_points.emplace_back(mesh->face(1671), Vector3{0.260122, 0.488442, 0.251435});

        surface_points.emplace_back(mesh->face(6270), Vector3{0.202274, 0.432051, 0.365675});
        surface_points.emplace_back(mesh->face(1976), Vector3{0.50546, 0.0951121, 0.399428});
        surface_points.emplace_back(mesh->face(3643), Vector3{0.376335, 0.203782, 0.419884});
        // surface_points.emplace_back(mesh->face(3193), Vector3{0.191301, 0.411843, 0.396856});
        surface_points.emplace_back(mesh->face(2226), Vector3{0.190355, 0.165665, 0.643979});

        edges.emplace_back(4,7);

        edges.emplace_back(27,0);
        edges.emplace_back(27,6);
        edges.emplace_back(27,20);

        edges.emplace_back(42,12);
        edges.emplace_back(42,22);
        edges.emplace_back(42,16);

        edges.emplace_back(7,34);
        edges.emplace_back(34,32);
        edges.emplace_back(34,33);
        edges.emplace_back(32,33);
        edges.emplace_back(33,31);
        edges.emplace_back(32,31);

        edges.emplace_back(34,35);
        edges.emplace_back(35,33);

        edges.emplace_back(34,35);
        edges.emplace_back(36,35);
        edges.emplace_back(36,37);
        edges.emplace_back(36,14);
        edges.emplace_back(37,14);

        edges.emplace_back(37,38);
        edges.emplace_back(33,38);

        edges.emplace_back(14,38);
        edges.emplace_back(36,20);
        edges.emplace_back(33,37);
        edges.emplace_back(14,31);
        edges.emplace_back(14,32);

        edges.emplace_back(1,43);

        edges.emplace_back(30,43);
        edges.emplace_back(30,28);
        edges.emplace_back(4,28);

        edges.emplace_back(41,30);
        edges.emplace_back(41,7);

        edges.emplace_back(40,29);
        edges.emplace_back(40,28);

        edges.emplace_back(41,29);
        edges.emplace_back(30,29);

        edges.emplace_back(4,39);
        edges.emplace_back(40,39);
        edges.emplace_back(41,39);

        edges.emplace_back(43,7);

        edges.emplace_back(44,31);
        edges.emplace_back(44,38);

        // edges.emplace_back(48,7);
        // edges.emplace_back(48,43);
        // edges.emplace_back(48,30);
        // edges.emplace_back(48,4);
        // edges.emplace_back(48,28);

        edges.emplace_back(47,7);
        edges.emplace_back(47,43);
        edges.emplace_back(47,30);
        edges.emplace_back(47,4);
        edges.emplace_back(47,28);

        edges.emplace_back(45,4);
        edges.emplace_back(45,39);
        edges.emplace_back(45,28);

        edges.emplace_back(46,36);
        edges.emplace_back(46,20);
        edges.emplace_back(46,1);
        
        dirichlet_vertices = {14, 44, 45, 4, 39, 5};
    }
    else
    {
        // for (int face_idx : {2311, 3165, 8878, 6506})
        // for (int face_idx : {4096, 2001, 470, 1754})
        for (int face_idx : {1424, 6093, 1648, 6177, 9161, 712})
        {
            T alpha = 1.0/3.0, beta = 1.0/3.0;
            gcs::Face f = mesh->face(face_idx);
            SurfacePoint new_pt(f, Vector3{alpha, beta, 1.0 - alpha - beta});
            new_pt = new_pt.inSomeFace();
            surface_points.push_back(new_pt);
        }

        // igl_edges.resize(4, 2);
        // igl_edges.row(0) = Eigen::RowVector2i(0, 1);
        // igl_edges.row(1) = Eigen::RowVector2i(1, 2);
        // igl_edges.row(2) = Eigen::RowVector2i(2, 3);
        // igl_edges.row(3) = Eigen::RowVector2i(3, 0);

        igl_edges.resize(5, 2);
        for (int i = 0; i < 5; i++)
        {
            igl_edges.row(i) = Eigen::RowVector2i(i, 5);    
        }
        dirichlet_vertices = {0, 1, 2, 3, 4};
    }

    surface_points_undeformed = surface_points;
    undeformed = VectorXT::Zero(surface_points.size() * 2);
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();
    delta_u.resize(undeformed.rows()); delta_u.setZero();

    
    
    
    
    for (int idx : dirichlet_vertices)
    {
        dirichlet_data[idx * 2 + 0] = 0.0;
        dirichlet_data[idx * 2 + 1] = 0.0;
    }

    if (two_way_coupling)
    {
        int n_dof = undeformed.rows();
        std::cout << n_dof << " " << undeformed.rows() << std::endl;
        if (use_FEM)
        {
            fem_dof_start = n_dof;
            // initializeFEMData(V, F);
            num_nodes = Vtet.rows();
            undeformed.conservativeResize(fem_dof_start + num_nodes * 3);
            tbb::parallel_for(0, num_nodes, [&](int i)
            {
                undeformed.segment<3>(fem_dof_start + i * 3) = Vtet.row(i);
            });
            deformed = undeformed;    
            delta_u = deformed; delta_u.setZero();
            u = deformed; u.setZero();
            num_ele = Ttet.rows();
            indices.resize(num_ele * 4);
            external_force.resize(num_nodes*3);
            external_force.setZero();
            // for (int i = 0; i < num_nodes; i++)
            // {
            //     external_force[i*3+1] = 1.0;
            // }
            
            tbb::parallel_for(0, num_ele, [&](int i)
            {
                indices.segment<4>(i * 4) = Ttet.row(i);
            });
            E_solids.resize(num_ele); E_solids.setConstant(1e1);
            std::cout << "initialize FEM data done" << std::endl;
            E_solid = 1e2;
            nu_solid = 0.49;
            // for (int i = 0; i < 9; i++)
            // {
            //     dirichlet_data[fem_dof_start + i * 3] = 0.0;
            // }
            
            for (int i = 0; i < num_nodes; i++)
            {
                TV xi = deformed.segment<3>(fem_dof_start + i * 3);
                if (xi[1] < min_corner[1] + 1e-1)
                {
                    dirichlet_data[fem_dof_start + i * 3] = 0.0;
                    dirichlet_data[fem_dof_start + i * 3 + 1] = 0.0;
                    dirichlet_data[fem_dof_start + i * 3 + 2] = 0.0;
                    // external_force[i * 3 + 1] = 0.0;
                }
            }
            iterateElementParallel([&](const EleNodes& x_deformed, 
                const EleNodes& x_undeformed, const VtxList& indices, int tet_idx)
            {
                TV center = TV::Zero();
                for (int i = 0; i < 4; i++)
                {
                    center += x_undeformed.row(i);
                }
                center /= 4.0;
                if (center[1] > max_corner[1] - 0.15)
                    E_solids[tet_idx] = 1e3;
            });
            
        }
        else
        {
            shell_dof_start = n_dof;
            faces = extrinsic_indices;
            undeformed.conservativeResize(n_dof + extrinsic_vertices.rows());
            undeformed.segment(n_dof, extrinsic_vertices.rows()) = extrinsic_vertices;
            deformed = undeformed;
            u.resize(undeformed.rows()); u.setZero();
            delta_u.resize(undeformed.rows()); delta_u.setZero();
            computeRestShape();
            buildHingeStructure();
            for (int d = 0; d < 3; d++)
            {
                dirichlet_data[shell_dof_start + d] = 0.0;
            }
            E_shell = 1e1;
            nu_shell = 0.3;
            updateShellLameParameters();
        }
    }    
    std::cout << "#dof : " << undeformed.rows() << std::endl;
    all_intrinsic_edges.resize(0);
    if (load_scene)
        connectSpringEdges(edges);
    else
        connectSpringEdges(igl_edges);
    std::cout << "connect spring edges done" << std::endl;

    add_area_term = false;
    add_geo_elasticity = false;
    verbose = false;
    add_length_term = true;

    we = 10.0;

    updateCurrentState();
    surface_points_undeformed = surface_points;
    ref_dis = 0.02;
}


void IntrinsicSimulation::initializeInteractiveScene()
{
    use_Newton = true;
    two_way_coupling = false;
    std::string base_folder = "../../../Projects/DifferentiableGeodesics/data/";
    std::string mesh_name = "bunny";
    MatrixXT V; MatrixXi F;
    igl::readOBJ(base_folder + mesh_name + "_sf.obj", V, F);

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
    
    mesh_center = center;
    mesh_scale = scale;

    buildMeshGeometry(V, F);


    std::vector<SurfacePoint> samples;
    std::vector<Edge> edges;
    std::ifstream in(base_folder + mesh_name + "_network.txt");
    int n_samples; in >> n_samples;
    samples.resize(n_samples);
    for (int i = 0; i < n_samples; i++)
    {
        int face_idx;
        T wx, wy, wz;
        in >> wx >> wy >> wz >> face_idx;
        samples[i] = SurfacePoint(mesh->face(face_idx), Vector3{wx, wy, wz});
    }
    int n_edges;
    in >> n_edges;
    edges.resize(n_edges);
    for (int i = 0; i < n_edges; i++)
    {
        int vi, vj;
        in >> vi, vj;
        edges[i] = Edge(vi, vj);
    }
    in.close();
    
    
    
    for (SurfacePoint& pt : samples)
    {
        pt = pt.inSomeFace();
        surface_points.push_back(pt);
    }

    for (int idx : dirichlet_vertices)
    {
        dirichlet_data[idx * 2] = 0.0;
        dirichlet_data[idx * 2 + 1] = 0.0;
    }
    
    

    // gcs::PoissonDiskSampler poissonSampler(*mesh, *geometry);
    // std::vector<SurfacePoint> samples = poissonSampler.sample(8.0);
    // int cnt = 0;
    // for (SurfacePoint& pt : samples)
    // {
    //     pt = pt.inSomeFace();
    //     surface_points.push_back(pt);
    // }

    int n_surface_point = samples.size();
    surface_points_undeformed = surface_points;
    undeformed = VectorXT::Zero(n_surface_point * 2);
    deformed = VectorXT::Zero(n_surface_point * 2);
    u = VectorXT::Zero(n_surface_point * 2);
    delta_u = VectorXT::Zero(n_surface_point * 2);
    delta_u.setRandom();
    delta_u *= 20.0 / delta_u.norm();
    
    
    // VectorXT mass_point_Euclidean(n_surface_point * 3);
    // for (int i = 0; i < n_surface_point; i++)
    // {
    //     mass_point_Euclidean.segment<3>(i*3) = 
    //         toTV(surface_points[i].interpolate(geometry->vertexPositions));
    // }
    

    // VectorXi triangulation;
    // triangulatePointCloud(mass_point_Euclidean, triangulation);

    // MatrixXi igl_tri, igl_edges;
    // vectorToIGLMatrix<int, 3>(triangulation, igl_tri);
    // igl::edges(igl_tri, igl_edges);
    
    // for (int i = 0; i < igl_edges.rows(); i++)
    //     edges.push_back(igl_edges.row(i));

    connectSpringEdges(edges);

    updateCurrentState(/*trace = */true);
    delta_u.setZero();

    add_length_term = true;
    add_area_term = false;
    add_geo_elasticity = false;
    add_volume = true;
    all_intrinsic_edges.resize(0);
    
    
    updateCurrentState();

    ref_dis = 0.05;
}


void IntrinsicSimulation::initializeElasticNetworkScene(std::string mesh_name, int type)
{
    use_Newton = true;
    two_way_coupling = false;
    max_newton_iter = 3000;
    use_FEM = false;
    std::string base_folder = "../../../Projects/DifferentiableGeodesics/data/";
    // std::string mesh_name = "torus_dense";

    // std::string mesh_name = "donut_duck";
    MatrixXT V; MatrixXi F;
    igl::readOBJ(base_folder + mesh_name + ".obj", V, F);

    buildMeshGeometry(V, F);

    int n_faces = extrinsic_indices.rows()/ 3;    

    intrinsic_vertices_barycentric_coords.resize(n_faces * 2);
    
    std::vector<TV> mass_point_vec;
    std::vector<TV2> bary_vec;

    VectorXT mass_point_Euclidean(n_faces * 3);

    int valid_cnt = 0;
    std::vector<Edge> edges;

    for (int i = 0; i < n_faces; i += 1)
    {
        T alpha, beta;
        if (type == 0) // near vertices
        {
            alpha = 0.99;
            beta = 1.0 * IRREGULAR_EPSILON;
        }
        else if (type == 1) // near middle of edges
        {
            alpha = 0.5;
            beta = 0.5 - IRREGULAR_EPSILON;
        }
        // T alpha = 0.3, beta = 0.3;
        TV vi = extrinsic_vertices.segment<3>(extrinsic_indices[i * 3 + 0] * 3);
        TV vj = extrinsic_vertices.segment<3>(extrinsic_indices[i * 3 + 1] * 3);
        TV vk = extrinsic_vertices.segment<3>(extrinsic_indices[i * 3 + 2] * 3);
        TV current = vi * alpha + vj * beta + vk * (1.0 - alpha - beta);
        // mass_point_Euclidean.segment<3>(i * 3) =  current;
        mass_point_vec.push_back(current);
        bary_vec.push_back(TV2(alpha, beta));
        // intrinsic_vertices_barycentric_coords.segment<2>(i * 2) = TV2(alpha, beta);
        gcs::Face f = mesh->face(i);
        SurfacePoint new_pt(f, Vector3{alpha, beta, 1.0 - alpha - beta});
        new_pt = new_pt.inSomeFace();
        surface_points.push_back(new_pt);
    }
    mass_point_Euclidean.resize(mass_point_vec.size() * 3);
    for (int i = 0; i < mass_point_vec.size(); i++)
        mass_point_Euclidean.segment<3>(i * 3) = mass_point_vec[i];
    intrinsic_vertices_barycentric_coords.resize(bary_vec.size() * 2);
    for (int i = 0; i < bary_vec.size(); i++)
        intrinsic_vertices_barycentric_coords.segment<2>(i * 2) = bary_vec[i];


    surface_points_undeformed = surface_points;
    undeformed = intrinsic_vertices_barycentric_coords;
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();
    delta_u.resize(undeformed.rows()); delta_u.setZero();

    std::cout << "add mass point" << std::endl;
    std::cout << "#dof : " << undeformed.rows() << " #points " << undeformed.rows() / 2 << std::endl;

    // dirichlet_vertices = {4, 3, 6};

    dirichlet_vertices = {0};

    // std::vector<int> dirichlet_vertices = {0};
    
    for (int idx : dirichlet_vertices)
    {
        dirichlet_data[idx * 2 + 0] = 0.0;
        dirichlet_data[idx * 2 + 1] = 0.0;
    }

    VectorXi triangulation;
    triangulatePointCloud(mass_point_Euclidean, triangulation);

    MatrixXi igl_tri, igl_edges;
    vectorToIGLMatrix<int, 3>(triangulation, igl_tri);
    igl::edges(igl_tri, igl_edges);

    MatrixXT igl_vertices;
    vectorToIGLMatrix<T, 3>(mass_point_Euclidean, igl_vertices);
    
    std::cout << "#triangles " << igl_tri.rows() << std::endl;
    // formEdgesFromConnection(F, igl_edges);
    
    for (int i = 0; i < igl_edges.rows(); i++)
        edges.push_back(igl_edges.row(i));

    add_area_term = false;
    
    add_geo_elasticity = false;
    add_volume = true && two_way_coupling;
    initializeNetworkData(edges);
    verbose = false;
    add_length_term =true;
    ref_dis = 0.1;
    we = 1.0;
}

void IntrinsicSimulation::initializeShellTwowayCouplingScene()
{
    use_Newton = true;
    use_lbfgs = false;
    two_way_coupling = true;
    max_newton_iter = 3000;
    use_FEM = false;
    std::string base_folder = "../../../Projects/DifferentiableGeodesics/data/";
    // std::string mesh_name = "torus_dense";

    std::string mesh_name = "sphere642";
    MatrixXT V; MatrixXi F;
    igl::readOBJ(base_folder + mesh_name + ".obj", V, F);

    buildMeshGeometry(V, F);

    int n_faces = extrinsic_indices.rows()/ 3;    

    intrinsic_vertices_barycentric_coords.resize(n_faces * 2);
    
    std::vector<TV> mass_point_vec;
    gcs::PoissonDiskSampler poissonSampler(*mesh, *geometry);
    std::vector<SurfacePoint> samples = poissonSampler.sample(5.0);
    int cnt = 0;
    for (SurfacePoint& pt : samples)
    {
        pt = pt.inSomeFace();
        mass_point_vec.push_back(toTV(pt.interpolate(geometry->vertexPositions)));
        surface_points.push_back(pt);
        cnt++;
    }

    surface_points_undeformed = surface_points;
    undeformed = VectorXT::Zero(samples.size() * 2);
    for (int i = 0; i < samples.size(); i++)
    {
        undeformed[i*2+0] = samples[i].faceCoords[0];
        undeformed[i*2+1] = samples[i].faceCoords[1];
    }
    
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();
    delta_u.resize(undeformed.rows()); delta_u.setZero();

    std::cout << "add mass point" << std::endl;
    std::cout << "#dof : " << undeformed.rows() << " #points " << undeformed.rows() / 2 << std::endl;

    // dirichlet_vertices = {4, 3, 6};

    dirichlet_vertices = {0};

    // std::vector<int> dirichlet_vertices = {0};
    
    for (int idx : dirichlet_vertices)
    {
        dirichlet_data[idx * 2 + 0] = 0.0;
        dirichlet_data[idx * 2 + 1] = 0.0;
    }

    VectorXT mass_point_Euclidean(mass_point_vec.size() * 3);
    for (int i = 0; i < mass_point_vec.size(); i++)
        mass_point_Euclidean.segment<3>(i * 3) = mass_point_vec[i];
   
    VectorXi triangulation;
    triangulatePointCloud(mass_point_Euclidean, triangulation);

    MatrixXi igl_tri, igl_edges;
    vectorToIGLMatrix<int, 3>(triangulation, igl_tri);
    igl::edges(igl_tri, igl_edges);
    std::vector<Edge> edges;
    for (int i = 0; i < igl_edges.rows(); i++)
        edges.push_back(igl_edges.row(i));

    if (two_way_coupling)
    {
        int n_dof = undeformed.rows();
        shell_dof_start = n_dof;
        undeformed.conservativeResize(n_dof + extrinsic_vertices.rows());
        undeformed.segment(n_dof, extrinsic_vertices.rows()) = extrinsic_vertices;
        deformed = undeformed;
        u.resize(undeformed.rows()); u.setZero();
        delta_u.resize(undeformed.rows()); delta_u.setZero();
    }    
    add_area_term = false;
    
    add_geo_elasticity = false;
    add_volume = true && two_way_coupling;
    initializeNetworkData(edges);
    verbose = false;
    add_length_term =true;
    ref_dis = 0.1;
    we = 10.0;
}


void IntrinsicSimulation::initializeMassSpringSceneExactGeodesic(std::string mesh_name, T resolution, bool use_Poisson_sampler)
{
    use_Newton = true;
    two_way_coupling = false;
    max_newton_iter = 3000;
    use_FEM = false;
    std::string base_folder = "../../../Projects/DifferentiableGeodesics/data/";
    // std::string mesh_name = "torus_dense";

    // std::string mesh_name = "donut_duck";
    MatrixXT V; MatrixXi F;
    igl::readOBJ(base_folder + mesh_name + ".obj", V, F);
    
    MatrixXT N;
    igl::per_vertex_normals(V, F, N);

    // V.row(10) += 3.0 * N.row(10);

    buildMeshGeometry(V, F);

    int n_faces = extrinsic_indices.rows()/ 3;    

    intrinsic_vertices_barycentric_coords.resize(n_faces * 2);
    
    std::vector<TV> mass_point_vec;
    std::vector<TV2> bary_vec;

    VectorXT mass_point_Euclidean(n_faces * 3);

    bool load_sites = false;

    // if (fileExist(base_folder + mesh_name + "_samples.txt"))
    //     load_sites = true;

    int valid_cnt = 0;
    std::vector<Edge> edges;
    if (load_sites)
    {
        std::vector<SurfacePoint> samples;
        std::ifstream in(base_folder + mesh_name + "_samples.txt");
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
        in.open(base_folder + mesh_name + "_idt.txt");
        int n_tri; in >> n_tri;
        MatrixXi idt_tri(n_tri, 3);
        for (int i = 0; i < n_tri; i++)
        {
            int a, b, c;
            in >> a >> b >> c;
            triangles.push_back(IV(a, b, c));
            idt_tri.row(i) = IV(a, b, c);
        }
        in.close();

        MatrixXi idt_edges;
        igl::edges(idt_tri, idt_edges);
        for (int i = 0; i < idt_edges.rows(); i++)
        {
            edges.push_back(idt_edges.row(i));
        }
        
        
        for (SurfacePoint& pt : samples)
        {
            pt.faceCoords = Vector3{0.9, 0.05, 1.0 - 0.9 - 0.05};
            pt = pt.inSomeFace();
            mass_point_vec.push_back(toTV(pt.interpolate(geometry->vertexPositions)));
            bary_vec.push_back(TV2(pt.faceCoords.x, pt.faceCoords.y));
            surface_points.push_back(pt);
        }
    }
    else
    {
        if (use_Poisson_sampler)
        {
            gcs::PoissonDiskSampler poissonSampler(*mesh, *geometry);
            std::vector<SurfacePoint> samples = poissonSampler.sample(resolution);
            // std::vector<SurfacePoint> samples = poissonSampler.sample(1.0);
            // std::vector<SurfacePoint> samples = poissonSampler.sample(0.8);
            // intrinsic_vertices_barycentric_coords.resize(samples.size() * 2);
            // mass_point_Euclidean.resize(samples.size() * 3);
            int cnt = 0;
            for (SurfacePoint& pt : samples)
            {
                pt.faceCoords = Vector3{0.9, 0.05, 1.0 - 0.9 - 0.05};
                pt = pt.inSomeFace();
                // mass_point_Euclidean.segment<3>(cnt * 3) = toTV(pt.interpolate(geometry->vertexPositions));
                mass_point_vec.push_back(toTV(pt.interpolate(geometry->vertexPositions)));
                bary_vec.push_back(TV2(pt.faceCoords.x, pt.faceCoords.y));
                // intrinsic_vertices_barycentric_coords.segment<2>(cnt * 2) = TV2(pt.faceCoords.x, pt.faceCoords.y);
                surface_points.push_back(pt);
                cnt++;
            }
        }
        else
        {
            for (int i = 0; i < n_faces; i += 1)
            {
                // T alpha = 0.5, beta = 0.5 - IRREGULAR_EPSILON;
                T alpha = 0.99, beta = 1.0 * IRREGULAR_EPSILON;
                // T alpha = 0.3, beta = 0.3;
                TV vi = extrinsic_vertices.segment<3>(extrinsic_indices[i * 3 + 0] * 3);
                TV vj = extrinsic_vertices.segment<3>(extrinsic_indices[i * 3 + 1] * 3);
                TV vk = extrinsic_vertices.segment<3>(extrinsic_indices[i * 3 + 2] * 3);
                TV current = vi * alpha + vj * beta + vk * (1.0 - alpha - beta);
                // mass_point_Euclidean.segment<3>(i * 3) =  current;
                mass_point_vec.push_back(current);
                bary_vec.push_back(TV2(alpha, beta));
                // intrinsic_vertices_barycentric_coords.segment<2>(i * 2) = TV2(alpha, beta);
                gcs::Face f = mesh->face(i);
                SurfacePoint new_pt(f, Vector3{alpha, beta, 1.0 - alpha - beta});
                new_pt = new_pt.inSomeFace();
                surface_points.push_back(new_pt);
            }
        }
    }

    mass_point_Euclidean.resize(mass_point_vec.size() * 3);
    for (int i = 0; i < mass_point_vec.size(); i++)
        mass_point_Euclidean.segment<3>(i * 3) = mass_point_vec[i];
    intrinsic_vertices_barycentric_coords.resize(bary_vec.size() * 2);
    for (int i = 0; i < bary_vec.size(); i++)
        intrinsic_vertices_barycentric_coords.segment<2>(i * 2) = bary_vec[i];


    surface_points_undeformed = surface_points;
    undeformed = intrinsic_vertices_barycentric_coords;
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();
    delta_u.resize(undeformed.rows()); delta_u.setZero();

    std::cout << "add mass point" << std::endl;
    std::cout << "#dof : " << undeformed.rows() << " #points " << undeformed.rows() / 2 << std::endl;

    // dirichlet_vertices = {4, 3, 6};

    dirichlet_vertices = {0, 1, 2};

    // std::vector<int> dirichlet_vertices = {0};
    
    for (int idx : dirichlet_vertices)
    {
        dirichlet_data[idx * 2 + 0] = 0.0;
        dirichlet_data[idx * 2 + 1] = 0.0;
    }

    // for (int i = undeformed.rows() / 2; i < undeformed.rows() / 2 + 2; i++)
    // {
    //     dirichlet_data[i] = 0.0;
    // }
   
    if (!load_sites)
    {
        VectorXi triangulation;
        triangulatePointCloud(mass_point_Euclidean, triangulation);

        MatrixXi igl_tri, igl_edges;
        vectorToIGLMatrix<int, 3>(triangulation, igl_tri);
        igl::edges(igl_tri, igl_edges);

        MatrixXT igl_vertices;
        vectorToIGLMatrix<T, 3>(mass_point_Euclidean, igl_vertices);
        
        // // igl::writeOBJ("triangulation.obj", igl_vertices, igl_tri);

        triangles.resize(igl_tri.rows());
        for (int i = 0; i < igl_tri.rows(); i++)
            triangles[i] = igl_tri.row(i);
        std::cout << "#triangles " << igl_tri.rows() << std::endl;
        // formEdgesFromConnection(F, igl_edges);
        
        for (int i = 0; i < igl_edges.rows(); i++)
            edges.push_back(igl_edges.row(i));
    }
    
    triangles.clear();

    if (two_way_coupling)
    {
        int n_dof = undeformed.rows();
        shell_dof_start = n_dof;
        undeformed.conservativeResize(n_dof + extrinsic_vertices.rows());
        undeformed.segment(n_dof, extrinsic_vertices.rows()) = extrinsic_vertices;
        deformed = undeformed;
        u.resize(undeformed.rows()); u.setZero();
        delta_u.resize(undeformed.rows()); delta_u.setZero();
    }    
    add_area_term = false;
    
    add_geo_elasticity = false;
    add_volume = true && two_way_coupling;
    initializeNetworkData(edges);
    verbose = false;
    add_length_term =true;
    ref_dis = 0.1;
    we = 1.0;
}


void IntrinsicSimulation::formEdgesFromConnection( 
    const MatrixXi& F, MatrixXi& igl_edges)
{
    std::vector<std::pair<int,int>> edge_vectors;
    Eigen::SparseMatrix<int> adj;
    igl::facet_adjacency_matrix(F, adj);
    for (int k=0; k < adj.outerSize(); ++k)
        for (Eigen::SparseMatrix<int>::InnerIterator it(adj,k); it; ++it)
        {
            edge_vectors.push_back(std::make_pair(it.row(), it.col()));
        }
    igl_edges.resize(edge_vectors.size(), 2);
    for (int i = 0; i < edge_vectors.size(); i++)
    {
        igl_edges(i, 0) = edge_vectors[i].first;
        igl_edges(i, 1) = edge_vectors[i].second;
    }
}

void IntrinsicSimulation::movePointsPlotEnergy()
{
    run_diff_test = true;
    VectorXT edge_length_squared_gradient(undeformed.rows());
    edge_length_squared_gradient.setZero();
    computeResidual(edge_length_squared_gradient); 
    // edge_length_squared_gradient *= -1.0;
    // std::cout << edge_length_squared_gradient.transpose() << std::endl;

    // TV normal = toTV(geometry->faceNormals[surface_points[1].second]);
    T a = edge_length_squared_gradient[2],
        b = edge_length_squared_gradient[3];
    edge_length_squared_gradient[2] = b;
    edge_length_squared_gradient[3] = -a;
    a = edge_length_squared_gradient[0],
    b = edge_length_squared_gradient[1];
    edge_length_squared_gradient[0] = -b;
    edge_length_squared_gradient[1] = a;
    std::cout << edge_length_squared_gradient.transpose()<<std::endl;
    // edge_length_squared_gradient << 0.0260971, -0.0127193, -0.0143727, 0.00148369;
    
    T scale = 0.001;
    int n_steps = 220 / scale;
    // T gmin = 0, g_max = 800;
    // T delta = (g_max - gmin) / n_steps;
    std::ofstream out("energy.txt");
    std::vector<T> energies, step_sizes;
    // for (T du = gmin; du < g_max + delta; du+= delta)
    out << std::setprecision(16);
    for (int i = 0 / scale; i < 200 / scale; i++)
    // for (int i = 48 / scale; i < 50 / scale; i++)
    // for (int i = 138 / scale; i < 140 / scale; i++)
    // for (int i = 170 / scale; i < 172 / scale; i++)
    {
        surface_points = surface_points_undeformed;
        delta_u = (i * scale) * edge_length_squared_gradient;
        updateCurrentState();
        // T energy = computeTotalEnergy();
        energies.push_back(current_length[0]);
        step_sizes.push_back((i * scale));
        // out << energy << " ";
    }
    for (int i = 0; i < energies.size(); i++)
        out << energies[i] << " ";
    out << std::endl;
    for (int i = 0; i < step_sizes.size(); i++)
        out << step_sizes[i] << " ";
    out << std::endl;
    out.close();
    surface_points = surface_points_undeformed;
    delta_u.setZero();
    updateCurrentState();
}

void IntrinsicSimulation::saveStates(const std::string& filename)
{
    std::ofstream out(filename);
    out << surface_points.size() << std::endl;
    for (const auto& surface_point : surface_points)
        out << surface_point.faceCoords[0] << " "
            << surface_point.faceCoords[1] << " "
            << surface_point.faceCoords[2] << " "
            << surface_point.face.getIndex() << std::endl;
    out.close();
}

void IntrinsicSimulation::saveNetworkData(const std::string& filename)
{
    std::ofstream out(filename);
    out << surface_points.size() << std::endl;
    for (const auto& surface_point : surface_points)
        out << surface_point.faceCoords[0] << " "
            << surface_point.faceCoords[1] << " "
            << surface_point.faceCoords[2] << " "
            << surface_point.face.getIndex() << std::endl;
    out << spring_edges.size() << std::endl;
    for (const auto& spring : spring_edges)
    {
        out << spring[0] << " " << spring[1] << std::endl;
    }
    out << dirichlet_vertices.size() << std::endl;
    for (int idx : dirichlet_vertices)
    {
        out << idx << " ";
    }
    out.close();
}

