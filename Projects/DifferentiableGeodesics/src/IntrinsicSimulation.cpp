
#include <Eigen/CholmodSupport>

#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

#include "../include/IntrinsicSimulation.h"
#include "geometrycentral/surface/surface_centers.h"

void IntrinsicSimulation::computeExactGeodesicPath(const SurfacePoint& va, const SurfacePoint& vb, 
        std::vector<SurfacePoint>& path)
{
    int n_tri = extrinsic_indices.rows() / 3;
    std::vector<std::vector<size_t>> mesh_indices_gc(n_tri, std::vector<size_t>(3));
    for (int i = 0; i < n_tri; i++)
        for (int d = 0; d < 3; d++)
            mesh_indices_gc[i][d] = extrinsic_indices[i * 3 + d];
    int n_vtx_extrinsic = extrinsic_vertices.rows() / 3;
    std::vector<gc::Vector3> mesh_vertices_gc(n_vtx_extrinsic);
    if (two_way_coupling)
    {
        if(use_FEM)
        {
            for (int i = 0; i < n_vtx_extrinsic; i++)
                mesh_vertices_gc[i] = gc::Vector3{
                    deformed(surface_to_tet_node_map[i] * 3 + 0 + fem_dof_start), 
                    deformed(surface_to_tet_node_map[i] * 3 + 1 + fem_dof_start), 
                    deformed(surface_to_tet_node_map[i] * 3 + 2 + fem_dof_start)};
        }
        else
            for (int i = 0; i < n_vtx_extrinsic; i++)
                mesh_vertices_gc[i] = gc::Vector3{
                    deformed(i * 3 + 0 + shell_dof_start), 
                    deformed(i * 3 + 1 + shell_dof_start), 
                    deformed(i * 3 + 2 + shell_dof_start)};
    }
    else 
        for (int i = 0; i < n_vtx_extrinsic; i++)
            mesh_vertices_gc[i] = gc::Vector3{extrinsic_vertices(i * 3 + 0), 
                extrinsic_vertices(i * 3 + 1), extrinsic_vertices(i * 3 + 2)};
    
    auto lvals = gcs::makeManifoldSurfaceMeshAndGeometry(mesh_indices_gc, mesh_vertices_gc);
    
    std::unique_ptr<gcs::ManifoldSurfaceMesh> sub_mesh;
    std::unique_ptr<gcs::VertexPositionGeometry> sub_geometry;
    std::tie(sub_mesh, sub_geometry) = std::tuple<std::unique_ptr<gcs::ManifoldSurfaceMesh>,
                    std::unique_ptr<gcs::VertexPositionGeometry>>(std::move(std::get<0>(lvals)),  // mesh
                                                             std::move(std::get<1>(lvals))); // geometry
    
    gcs::GeodesicAlgorithmExact mmp(*sub_mesh, *sub_geometry);

    SurfacePoint va_sub(sub_mesh->face(va.face.getIndex()), va.faceCoords);
    SurfacePoint vb_sub(sub_mesh->face(vb.face.getIndex()), vb.faceCoords);


    mmp.propagate(va_sub);
    T dis;
    path = mmp.traceBack(vb_sub, dis);
    std::reverse(path.begin(), path.end());
}

T IntrinsicSimulation::computeExactGeodesicDistance(const SurfacePoint& va, const SurfacePoint& vb)
{
    int n_tri = extrinsic_indices.rows() / 3;
    std::vector<std::vector<size_t>> mesh_indices_gc(n_tri, std::vector<size_t>(3));
    for (int i = 0; i < n_tri; i++)
        for (int d = 0; d < 3; d++)
            mesh_indices_gc[i][d] = extrinsic_indices[i * 3 + d];
    int n_vtx_extrinsic = extrinsic_vertices.rows() / 3;
    std::vector<gc::Vector3> mesh_vertices_gc(n_vtx_extrinsic);
    if (two_way_coupling)
    {
        if(use_FEM)
        {
            for (int i = 0; i < n_vtx_extrinsic; i++)
                mesh_vertices_gc[i] = gc::Vector3{
                    deformed(surface_to_tet_node_map[i] * 3 + 0 + fem_dof_start), 
                    deformed(surface_to_tet_node_map[i] * 3 + 1 + fem_dof_start), 
                    deformed(surface_to_tet_node_map[i] * 3 + 2 + fem_dof_start)};
        }
        else
            for (int i = 0; i < n_vtx_extrinsic; i++)
                mesh_vertices_gc[i] = gc::Vector3{
                    deformed(i * 3 + 0 + shell_dof_start), 
                    deformed(i * 3 + 1 + shell_dof_start), 
                    deformed(i * 3 + 2 + shell_dof_start)};
    }
    else 
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
    return mmp.getDistance(vb_sub);
}

void IntrinsicSimulation::computeExactGeodesic(const SurfacePoint& va, const SurfacePoint& vb, 
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
    if (two_way_coupling)
    {
        if(use_FEM)
        {
            for (int i = 0; i < n_vtx_extrinsic; i++)
                mesh_vertices_gc[i] = gc::Vector3{
                    deformed(surface_to_tet_node_map[i] * 3 + 0 + fem_dof_start), 
                    deformed(surface_to_tet_node_map[i] * 3 + 1 + fem_dof_start), 
                    deformed(surface_to_tet_node_map[i] * 3 + 2 + fem_dof_start)};
        }
        else
            for (int i = 0; i < n_vtx_extrinsic; i++)
                mesh_vertices_gc[i] = gc::Vector3{
                    deformed(i * 3 + 0 + shell_dof_start), 
                    deformed(i * 3 + 1 + shell_dof_start), 
                    deformed(i * 3 + 2 + shell_dof_start)};
    }
    else 
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
                    // std::exit(0);
                }
                test_interp = pt.tEdge * start + (1.0 - pt.tEdge) * end;
                if ((ixn - test_interp).norm() > 1e-6)
                {
                    std::cout << "error in interpolation" << std::endl;
                    // std::exit(0);
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
                    // std::exit(0);
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

void IntrinsicSimulation::computeGeodesicHeatMethod(const SurfacePoint& va, const SurfacePoint& vb, 
        T& dis, std::vector<SurfacePoint>& path, 
        std::vector<IxnData>& ixn_data, 
        bool trace_path)
{
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
    
    
    SurfacePoint va_sub = SurfacePoint(sub_mesh->vertex(0));
    SurfacePoint vb_sub = SurfacePoint(sub_mesh->vertex(1));

    T h = 0.0;
    for (gcs::Edge edge : sub_mesh->edges())
        h += sub_geometry->edgeLength(edge);
    h /= T(sub_mesh->nEdges());
    
    gcs::VectorHeatMethodSolver vhmSolver(*sub_geometry, h*h);

    gcs::HeatMethodDistanceSolver hmSolver(*sub_geometry, h*h);    

    gcs::VertexData<T> geo_dis = hmSolver.computeDistance({va_sub});


    dis = geo_dis[vb_sub.vertex];
    // std::cout << dis << std::endl;
    gcs::VertexData<gc::Vector2> logmap = vhmSolver.computeLogMap(va_sub);
    
    // gc::Vector2 pointCoord = logmap[vb_sub.vertex];
    gc::Vector2 pointCoord = vb_sub.interpolate(logmap);
    dis = pointCoord.norm();
    std::cout << "(u, v) " << pointCoord << std::endl;
    std::cout << "geodesic from log map " << pointCoord.norm() << std::endl;
    // gc::Vector2 dir = pointCoord.normalize() * dis;
    // gc::Vector2 dir = 
    // gc::Vector2 pointCoord = vb_sub.interpolate(logmap);
    // dis = pointCoord.norm();

    gcs::TraceOptions options;
    options.includePath = true; 
    gcs::TraceGeodesicResult traceResult = traceGeodesic(*sub_geometry, va_sub, pointCoord, options);
    SurfacePoint candidatePoint = traceResult.endPoint.inSomeFace();
    path = traceResult.pathPoints;
    for (auto& pt : path)
    {
        T edge_t = -1.0; TV start = TV::Zero(), end = TV::Zero();
        bool is_edge_point = (pt.type == gcs::SurfacePointType::Edge);
        if (is_edge_point)
        {
            auto he = pt.edge.halfedge();
            SurfacePoint start_extrinsic = he.tailVertex();
            SurfacePoint end_extrinsic = he.tipVertex();
            start = toTV(start_extrinsic.interpolate(sub_geometry->vertexPositions));
            end = toTV(end_extrinsic.interpolate(sub_geometry->vertexPositions));
            TV ixn = toTV(pt.interpolate(sub_geometry->vertexPositions));
            TV test_interp = pt.tEdge * start + (1.0 - pt.tEdge) * end;
            if ((test_interp - ixn).norm() > 1e-6)
                std::swap(start, end);
            
            TV dir0 = (end-start).normalized();
            TV dir1 = (ixn-start).normalized();
            if ((dir0.cross(dir1)).norm() > 1e-6)
            {
                std::cout << "error in cross product" << std::endl;
                // std::exit(0);
            }
            test_interp = pt.tEdge * start + (1.0 - pt.tEdge) * end;
            if ((ixn - test_interp).norm() > 1e-6)
            {
                std::cout << "error in interpolation" << std::endl;
                // std::exit(0);
            }
            edge_t = pt.tEdge;
            // std::cout << pt.tEdge << " " << " cross " << (dir0.cross(dir1)).norm() << std::endl;
            // std::cout << ixn.transpose() << " " << (pt.tEdge * start + (1.0-pt.tEdge) * end).transpose() << std::endl;
            // std::getchar();
        }
        ixn_data.push_back(IxnData(start, end, (1.0-edge_t)));
        pt.edge = mesh->edge(pt.edge.getIndex());
        pt.vertex = mesh->vertex(pt.vertex.getIndex());
        pt.face = mesh->face(pt.face.getIndex());
        
        pt = pt.inSomeFace();
    }
}

bool IntrinsicSimulation::hasSmallSegment(const std::vector<SurfacePoint>& path)
{
    for (int i = 0; i < path.size() - 1; i++)
    {
        TV ixn0 = toTV(path[i].interpolate(geometry->vertexPositions));
        TV ixn1 = toTV(path[i+1].interpolate(geometry->vertexPositions));
        if ((ixn0 - ixn1).norm() < 1e-6)
        {
            return true;
        }
    }
    return false;
}

void IntrinsicSimulation::traceGeodesics()
{
    if (!retrace)
        return;
    int n_springs = spring_edges.size();
    current_length.resize(n_springs);
    paths.resize(n_springs);
    ixn_data_list.resize(n_springs, std::vector<IxnData>());
    // START_TIMING(traceGeodesics)
#ifdef PARALLEL_GEODESIC
    tbb::parallel_for(0, n_springs, [&](int i)
#else
	for(int i = 0; i < n_springs; i++)
#endif
    {
        SurfacePoint vA = surface_points[spring_edges[i][0]];
        SurfacePoint vB = surface_points[spring_edges[i][1]];
        
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
    // FINISH_TIMING_PRINT(traceGeodesics)
    retrace = false;
}

void IntrinsicSimulation::computeExactGeodesicEdgeFlip(const SurfacePoint& va, const SurfacePoint& vb, 
        T& dis, std::vector<SurfacePoint>& path, 
        std::vector<IxnData>& ixn_data, bool trace_path)
{
    // START_TIMING(contruct)
    ixn_data.clear();
    int n_tri = extrinsic_indices.rows() / 3;
    std::vector<std::vector<size_t>> mesh_indices_gc(n_tri, std::vector<size_t>(3));
    for (int i = 0; i < n_tri; i++)
        for (int d = 0; d < 3; d++)
            mesh_indices_gc[i][d] = extrinsic_indices[i * 3 + d];
    int n_vtx_extrinsic = extrinsic_vertices.rows() / 3;
    std::vector<gc::Vector3> mesh_vertices_gc(n_vtx_extrinsic);
    if (two_way_coupling)
    {
        if(use_FEM)
        {
            for (int i = 0; i < n_vtx_extrinsic; i++)
                mesh_vertices_gc[i] = gc::Vector3{
                    deformed(surface_to_tet_node_map[i] * 3 + 0 + fem_dof_start), 
                    deformed(surface_to_tet_node_map[i] * 3 + 1 + fem_dof_start), 
                    deformed(surface_to_tet_node_map[i] * 3 + 2 + fem_dof_start)};
        }
        else
            for (int i = 0; i < n_vtx_extrinsic; i++)
                mesh_vertices_gc[i] = gc::Vector3{
                    deformed(i * 3 + 0 + shell_dof_start), 
                    deformed(i * 3 + 1 + shell_dof_start), 
                    deformed(i * 3 + 2 + shell_dof_start)};
    }
    else 
        for (int i = 0; i < n_vtx_extrinsic; i++)
            mesh_vertices_gc[i] = gc::Vector3{extrinsic_vertices(i * 3 + 0), 
                extrinsic_vertices(i * 3 + 1), extrinsic_vertices(i * 3 + 2)};
    auto lvals = gcs::makeManifoldSurfaceMeshAndGeometry(mesh_indices_gc, mesh_vertices_gc);

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
        // std::cout << "Failed Connection" <<std::endl;
        computeExactGeodesic(va, vb, dis, path, ixn_data, trace_path);
        return;
    }
    dis = sub_edgeNetwork->tri->edgeLengths[ei];
    // START_TIMING(gather_info)
    if (trace_path)
    {
        gcs::Halfedge he = ei.halfedge();
        if (he.tailVertex() == vb_vtx)
            he = ei.halfedge().twin();
        path = sub_edgeNetwork->tri->traceIntrinsicHalfedgeAlongInput(he, false);
        for (auto& pt : path)
        {
            // std::cout << pt.inSomeFace().face.halfedge().getIndex() << std::endl;
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
                    // std::exit(0);
                }
                test_interp = pt.tEdge * start + (1.0 - pt.tEdge) * end;
                if ((ixn - test_interp).norm() > 1e-6)
                {
                    std::cout << "error in interpolation" << std::endl;
                    // std::exit(0);
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
                    // std::exit(0);
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
    }
    // FINISH_TIMING_PRINT(gather_info)
}

void IntrinsicSimulation::getMarkerPointsPosition(VectorXT& positions)
{
    if (!surface_points.size())
        return;
    std::vector<int> marker_indices = dirichlet_vertices;
    positions.resize(marker_indices.size() * 3);
    for (int i = 0; i < marker_indices.size(); i++)
    {
        // if (marker_indices[i] * 3 < undeformed.rows())
            positions.segment<3>(i * 3) = toTV(surface_points[marker_indices[i]].interpolate(geometry->vertexPositions));
    }
    // std::cout << surface_points[marker_indices[0]].first.faceCoords << std::endl;
}

void IntrinsicSimulation::getAllPointsPosition(VectorXT& positions)
{
    positions.resize(surface_points.size() * 3);
    for (int i = 0; i < surface_points.size(); i++)
    {
        positions.segment<3>(i * 3) = toTV(surface_points[i].interpolate(geometry->vertexPositions));
    }
}

bool IntrinsicSimulation::closeToIrregular(const SurfacePoint& point)
{
    SurfacePoint end_point = point.inSomeFace();
    Vector3 bc = end_point.faceCoords;
    int close_to_zero_cnt = 0;
    for (int d = 0; d < 3; d++)
    {
        if (std::abs(bc[d]) < IRREGULAR_EPSILON)
            close_to_zero_cnt++;
    }
    if (close_to_zero_cnt > 0)
    {
        return true;
    }
    return false;
}

void IntrinsicSimulation::getCurrentMassPointConfiguration(
    std::vector<SurfacePoint>& configuration)
{
    configuration = surface_points_undeformed;
    for (int i = 0; i < surface_points.size(); i++)
    {
        configuration[i] = surface_points[i];
    }
}

void IntrinsicSimulation::updateCurrentState(bool trace)
{
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            delta_u[offset] = target;
        });
    }
    
    
    if (two_way_coupling)
    {
        deformed += delta_u;
        geometry->requireVertexPositions();
        if (use_FEM)
        {
            int n_vtx_extrinsic = extrinsic_vertices.rows() / 3;
            for (int i = 0; i < n_vtx_extrinsic; i++)
            {
                geometry->vertexPositions[mesh->vertex(i)] = 
                    gc::Vector3{deformed(surface_to_tet_node_map[i] * 3 + 0 + fem_dof_start), 
                    deformed(surface_to_tet_node_map[i] * 3 + 1 + fem_dof_start), 
                    deformed(surface_to_tet_node_map[i] * 3 + 2 + fem_dof_start)};
            }
        }
        else
        {
            int n_vtx_extrinsic = extrinsic_vertices.rows() / 3;
            for (int i = 0; i < n_vtx_extrinsic; i++)
            {
                geometry->vertexPositions[mesh->vertex(i)] = 
                    gc::Vector3{deformed(i * 3 + 0 + shell_dof_start), 
                    deformed(i * 3 + 1 + shell_dof_start), 
                    deformed(i * 3 + 2 + shell_dof_start)};
            }
            
        }
        geometry->unrequireVertexPositions();
        geometry->refreshQuantities();
    }

    for (int i = 0; i < surface_points.size(); i++)
    {
        
        gcFace fi = surface_points[i].face;
        Vector3 start_bc = surface_points[i].faceCoords;
        Vector3 trace_vec{delta_u[i*2+0],delta_u[i*2+1],0.0-delta_u[i*2+0]-delta_u[i*2+1]};
        
        if (trace_vec.norm() < 1e-10)
            continue;

        gcs::TraceOptions options; 
        options.includePath = true;
        
        // trace geodesic on the extrinsic mesh
        gcs::TraceGeodesicResult result = gcs::traceGeodesic(*geometry, fi, start_bc, trace_vec, options);
        
        if (result.pathPoints.size() != 1)
        {
            // find equivalent intrinc vertex
            SurfacePoint endpoint = result.endPoint.inSomeFace();
            surface_points[i] = endpoint;

            // std::cout << "face " << surface_points[i].second.getIndex() << std::endl;
        }
        else
        {
            if( delta_u.segment<2>(i*2).norm() > 1e-8 )
            {
                std::cout << std::setprecision(12) << "trace length " << result.length << std::endl;
                std::cout << std::setprecision(12) << "node " << i << " |u| " << delta_u.segment<2>(i*2).transpose() << std::endl;
                std::cout << "can't trace it" << std::endl; 
                std::getchar();
            }
        }
    }
    if (trace)
    {
        retrace = true;
        traceGeodesics();
    }
    
}

void IntrinsicSimulation::checkHessianPD(bool save_result)
{
    int nmodes = 10;
    int n_dof_sim = deformed.rows();
    StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
    run_diff_test = true;
    buildSystemMatrix(d2edx2);
    run_diff_test = false;
    bool use_Spectra = true;

    // Eigen::PardisoLLT<StiffnessMatrix, Eigen::Lower> solver;
    Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> solver;
    solver.analyzePattern(d2edx2); 
    // std::cout << "analyzePattern" << std::endl;
    solver.factorize(d2edx2);
    // std::cout << "factorize" << std::endl;
    bool indefinite = false;
    if (solver.info() == Eigen::NumericalIssue)
    {
        std::cout << "!!!indefinite matrix!!!" << std::endl;
        indefinite = true;
        
    }
    
    if (use_Spectra)
    {
        
        Spectra::SparseSymShiftSolve<T, Eigen::Lower> op(d2edx2);
        T shift = -1e-4;
        Spectra::SymEigsShiftSolver<T, 
        Spectra::LARGEST_MAGN, 
        Spectra::SparseSymShiftSolve<T, Eigen::Lower> > 
            eigs(&op, nmodes, 2 * nmodes, shift);

        eigs.init();

        int nconv = eigs.compute();

        if (eigs.info() == Spectra::SUCCESSFUL)
        {
            Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().real();
            Eigen::VectorXd eigen_values = eigs.eigenvalues().real();
            std::cout << eigen_values.transpose() << std::endl;
            if (save_result)
            {
                std::ofstream out("eigen_vectors.txt");
                out << eigen_vectors.rows() << " " << eigen_vectors.cols() << std::endl;
                for (int i = 0; i < eigen_vectors.cols(); i++)
                    out << eigen_values[eigen_vectors.cols() - 1 - i] << " ";
                out << std::endl;
                for (int i = 0; i < eigen_vectors.rows(); i++)
                {
                    // for (int j = 0; j < eigen_vectors.cols(); j++)
                    for (int j = eigen_vectors.cols() - 1; j >-1 ; j--)
                        out << eigen_vectors(i, j) << " ";
                    out << std::endl;
                }       
                out << std::endl;
                out.close();
            }
        }
        else
        {
            std::cout << "Eigen decomposition failed" << std::endl;
        }
    }
    else
    {
        Eigen::MatrixXd A_dense = d2edx2;
        Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver;
        eigen_solver.compute(A_dense, /* computeEigenvectors = */ true);
        auto eigen_values = eigen_solver.eigenvalues();
        auto eigen_vectors = eigen_solver.eigenvectors();
        
        std::vector<T> ev_all(A_dense.cols());
        for (int i = 0; i < A_dense.cols(); i++)
        {
            ev_all[i] = eigen_values[i].real();
        }
        
        std::vector<int> indices;
        for (int i = 0; i < A_dense.cols(); i++)
        {
            indices.push_back(i);    
        }
        std::sort(indices.begin(), indices.end(), [&ev_all](int a, int b){ return ev_all[a] < ev_all[b]; } );
        // std::sort(ev_all.begin(), ev_all.end());

        for (int i = 0; i < nmodes; i++)
            std::cout << ev_all[indices[i]] << std::endl;
        
        if (save_result)
        {
            std::ofstream out("eigen_vectors.txt");
            out << nmodes << " " << A_dense.cols() << std::endl;
            for (int i = 0; i < nmodes; i++)
                out << ev_all[indices[i]] << " ";
            out << std::endl;
            for (int i = 0; i < nmodes; i++)
            {
                out << eigen_vectors.col(indices[i]).real().transpose() << std::endl;
            }
            out.close();
        }
    }
}

bool IntrinsicSimulation::linearSolveWoodbury(StiffnessMatrix& K, const MatrixXT& UV,
        const VectorXT& residual, VectorXT& du)
{
    MatrixXT UVT= UV.transpose();
    
    Eigen::CholmodSupernodalLLT<StiffnessMatrix> solver;
    
    T alpha = 10e-6;
    solver.analyzePattern(K);
    // T time_analyze = t.elapsed_sec();
    // std::cout << "\t analyzePattern takes " << time_analyze << "s" << std::endl;
    int i = 0;
    int indefinite_count_reg_cnt = 0, invalid_search_dir_cnt = 0, invalid_residual_cnt = 0;
    for (; i < 50; i++)
    {
        // std::cout << i << std::endl;
        solver.factorize(K);
        // T time_factorize = t.elapsed_sec() - time_analyze;
        // std::cout << "\t factorize takes " << time_factorize << "s" << std::endl;
        // std::cout << "-----factorization takes " << t.elapsed_sec() << "s----" << std::endl;
        if (solver.info() == Eigen::NumericalIssue)
        {
            K.diagonal().array() += alpha;
            alpha *= 10;
            indefinite_count_reg_cnt++;
            continue;
        }

        // sherman morrison
        if (UV.cols() == 1)
        {
            VectorXT v = UV.col(0);
            MatrixXT rhs(K.rows(), 2); rhs.col(0) = residual; rhs.col(1) = v;
            // VectorXT A_inv_g = solver.solve(residual);
            // VectorXT A_inv_u = solver.solve(v);
            MatrixXT A_inv_gu = solver.solve(rhs);

            T dem = 1.0 + v.dot(A_inv_gu.col(1));

            du.noalias() = A_inv_gu.col(0) - (A_inv_gu.col(0).dot(v)) * A_inv_gu.col(1) / dem;
        }
        // UV is actually only U, since UV is the same in the case
        // C is assume to be Identity
        else // Woodbury https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        {
            VectorXT A_inv_g = solver.solve(residual);

            MatrixXT A_inv_U(UV.rows(), UV.cols());
            // for (int col = 0; col < UV.cols(); col++)
                // A_inv_U.col(col) = solver.solve(UV.col(col));
            A_inv_U = solver.solve(UV);

            MatrixXT C(UV.cols(), UV.cols());
            C.setIdentity();
            C += UVT * A_inv_U;
            du = A_inv_g - A_inv_U * C.inverse() * UVT * A_inv_g;
        }
        

        T dot_dx_g = du.normalized().dot(residual.normalized());

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;

        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-3;
        if (!search_dir_correct_sign)
            invalid_search_dir_cnt++;
        
        bool solve_success = true;//(K * du + UV * UV.transpose()*du - residual).norm() < 1e-6 && solver.info() == Eigen::Success;
        
        if (!solve_success)
            invalid_residual_cnt++;
        // std::cout << "PD: " << positive_definte << " direction " 
        //     << search_dir_correct_sign << " solve " << solve_success << std::endl;

        if (positive_definte && search_dir_correct_sign && solve_success)
        {
            
            if (verbose)
            {
                std::cout << "\t===== Linear Solve ===== " << std::endl;
                std::cout << "\tnnz: " << K.nonZeros() << std::endl;
                // std::cout << "\t takes " << t.elapsed_sec() << "s" << std::endl;
                std::cout << "\t# regularization step " << i 
                    << " indefinite " << indefinite_count_reg_cnt 
                    << " invalid search dir " << invalid_search_dir_cnt
                    << " invalid solve " << invalid_residual_cnt << std::endl;
                std::cout << "\tdot(search, -gradient) " << dot_dx_g << std::endl;
                // std::cout << (K.selfadjointView<Eigen::Lower>() * du + UV * UV.transpose()*du - residual).norm() << std::endl;
                std::cout << "\t======================== " << std::endl;
            }
            return true;
        }
        else
        {
            // K = H + alpha * I;       
            // tbb::parallel_for(0, (int)K.rows(), [&](int row)    
            // {
            //     K.coeffRef(row, row) += alpha;
            // });  
            K.diagonal().array() += alpha;
            alpha *= 10;
        }
    }
    return false;
}


bool IntrinsicSimulation::linearSolve(StiffnessMatrix& K, const VectorXT& residual, VectorXT& du)
{
    // START_TIMING(LinearSolve)
    Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> solver;
    // Eigen::CholmodSupernodalLLT<StiffnessMatrix> solver;
    
    T alpha = 1e-6;
    StiffnessMatrix H(K.rows(), K.cols());
    H.setIdentity(); H.diagonal().array() = 1e-10;
    K += H;
    solver.analyzePattern(K);
    // T time_analyze = t.elapsed_sec();
    // std::cout << "\t analyzePattern takes " << time_analyze << "s" << std::endl;

    // std::cout << K << std::endl;
    
    int indefinite_count_reg_cnt = 0, invalid_search_dir_cnt = 0, invalid_residual_cnt = 0;
    int i = 0;
    T dot_dx_g = 0.0;
    for (; i < 50; i++)
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
        
        dot_dx_g = du.normalized().dot(residual.normalized());

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
            
            if (verbose)
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
                // FINISH_TIMING_PRINT(LinearSolve)
            }
            return true;
        }
        else
        {
            K.diagonal().array() += alpha;
            alpha *= 10;
        }
    }
    if (verbose)
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
        // FINISH_TIMING_PRINT(LinearSolve)
    }
    return false;
}

void IntrinsicSimulation::projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data)
{
    for (auto iter : data)
    {
        A.row(iter.first) *= 0.0;
        A.col(iter.first) *= 0.0;
        A.coeffRef(iter.first, iter.first) = 1.0;
    }
}

T IntrinsicSimulation::computeTotalEnergy()
{
    T total_energy = 0.0;

    if(add_length_term)
        addEdgeLengthEnergy(we, total_energy);
    
    T area_energy = 0.0;
    if (add_area_term)
        addTriangleAreaEnergy(wa, area_energy);
    // std::cout << area_energy << std::endl;
    total_energy += area_energy;
    if (two_way_coupling)
    {
        T elastic_energy = 0.0;
        if (use_FEM)
        {
            elastic_energy = addElastsicPotential();
        }
        else
            addShellEnergy(elastic_energy);
        total_energy += elastic_energy;
        if (use_FEM)
        {
            VectorXT dx = (deformed - undeformed).segment(fem_dof_start, num_nodes *3);
            total_energy -= external_force.dot(dx);
        }
    }

    if (add_geo_elasticity)
    {
        T geo_elastic_energy = 0.0;
        addGeodesicNHEnergy(geo_elastic_energy);
        total_energy += geo_elastic_energy;    
        // std::cout << geo_elastic_energy << std::endl;
    }

    if (add_volume)
    {
        T volume_term = 0.0;
        addVolumePreservationEnergy(wv, volume_term);
        total_energy += volume_term;
    }
    
    return total_energy;
}

T IntrinsicSimulation::computeResidual(VectorXT& residual)
{
    if(add_length_term)
        addEdgeLengthForceEntries(we, residual);
    if (add_area_term)
        addTriangleAreaForceEntries(wa, residual);
    if (add_geo_elasticity)
        addGeodesicNHForceEntry(residual);
    if (two_way_coupling)
    {
        if (use_FEM)
            addElasticForceEntries(residual);
        else
            addShellForceEntry(residual);
        if (use_FEM)
        {
            residual.segment(fem_dof_start, num_nodes*3) += external_force;
        }
    }
    if (add_volume)
        addVolumePreservationForceEntries(wv, residual);
    
    if (!run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            residual[offset] = 0;
        });

    return residual.norm();
}

void IntrinsicSimulation::buildSystemMatrixWoodbury(StiffnessMatrix& K, MatrixXT& UV)
{
    std::vector<Entry> entries;
    if(add_length_term)
        addEdgeLengthHessianEntries(we, entries);
    if (add_area_term)
        addTriangleAreaHessianEntries(wa, entries);
    if (add_geo_elasticity)
        addGeodesicNHHessianEntry(entries);
    if (two_way_coupling)
    {
        if (use_FEM)
            addElasticHessianEntries(entries);
        else
            addShellHessianEntries(entries);
    }
    if (add_volume)
    {
        addVolumePreservationHessianEntries(wv, entries, UV);
    }
    int n_dof = deformed.rows();
    K.resize(n_dof, n_dof);

    K.setFromTriplets(entries.begin(), entries.end());
    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    
    K.makeCompressed();
}

void IntrinsicSimulation::buildSystemMatrix(StiffnessMatrix& K)
{
    std::vector<Entry> entries;
    if(add_length_term)
        addEdgeLengthHessianEntries(we, entries);
    if (add_area_term)
        addTriangleAreaHessianEntries(wa, entries);
    if (add_geo_elasticity)
        addGeodesicNHHessianEntry(entries);
    if (two_way_coupling)
    {
        if (use_FEM)
            addElasticHessianEntries(entries);
        else
            addShellHessianEntries(entries);
    }
    if (add_volume)
    {
        MatrixXT Woodbury_matrix;
        addVolumePreservationHessianEntries(wv, entries, Woodbury_matrix);
    }
    int n_dof = deformed.rows();
    K.resize(n_dof, n_dof);
    
    
    K.setFromTriplets(entries.begin(), entries.end());
    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    
    K.makeCompressed();
}

void IntrinsicSimulation::moveMassPoint(int idx, int bc)
{
    
    u[idx * 2 + bc] += 0.001;
    
    deformed = undeformed + u;

    std::cout << "on face " << surface_points[idx].face.getIndex() << std::endl;
}

void IntrinsicSimulation::massPointPosition(int idx, TV& pos)
{
    pos = toTV(surface_points[idx].interpolate(geometry->vertexPositions));
}

void IntrinsicSimulation::updateVisualization()
{
    all_intrinsic_edges.resize(0);

    int n_springs = spring_edges.size();
    std::vector<std::vector<std::pair<TV, TV>>> sub_pairs(n_springs, std::vector<std::pair<TV, TV>>());
    
    if (retrace)
    {
        traceGeodesics();
    }
    
	for(int i = 0; i < n_springs; i++)
    {

        SurfacePoint vA = surface_points[spring_edges[i][0]];
        SurfacePoint vB = surface_points[spring_edges[i][1]];

        std::vector<SurfacePoint> path = paths[i];
        for(int j = 0; j < path.size() - 1; j++)
        {
            TV xj = toTV(path[j].interpolate(geometry->vertexPositions));
            TV xk = toTV(path[j+1].interpolate(geometry->vertexPositions));
            if ((xk - xj).norm() > 1e-8)
                sub_pairs[i].push_back(std::make_pair(xj, xk));
        }
    }
    for (int i = 0; i < n_springs; i++)
    {
        all_intrinsic_edges.insert(all_intrinsic_edges.end(), sub_pairs[i].begin(), sub_pairs[i].end());
    }
}

void IntrinsicSimulation::updateVisualization(std::vector<TV>& colors, bool all_edges)
{
    colors.resize(0);
    all_intrinsic_edges.resize(0);

    int n_springs = spring_edges.size();
    std::vector<std::vector<std::pair<TV, TV>>> sub_pairs(n_springs, std::vector<std::pair<TV, TV>>());
    
    if (retrace)
    {
        traceGeodesics();
    }
    
	for(int i = 0; i < n_springs; i++)
    {

        SurfacePoint vA = surface_points[spring_edges[i][0]];
        SurfacePoint vB = surface_points[spring_edges[i][1]];

        
        std::vector<SurfacePoint> path = paths[i];
        for(int j = 0; j < path.size() - 1; j++)
        {
            sub_pairs[i].push_back(std::make_pair(
                toTV(path[j].interpolate(geometry->vertexPositions)),
                toTV(path[j+1].interpolate(geometry->vertexPositions))
            ));
            if (current_length[i] < rest_length[i])
                colors.push_back(TV(0.3,1.0,0.0));
            else
                colors.push_back(TV(1.0,0.3,0.0));
        }
    }
    for (int i = 0; i < n_springs; i++)
    {
        all_intrinsic_edges.insert(all_intrinsic_edges.end(), sub_pairs[i].begin(), sub_pairs[i].end());
    }
}

bool IntrinsicSimulation::KarcherMeanOneStep(int step)
{
    std::vector<gcVertex> vertexPts = {
        mesh->vertex(591),
        mesh->vertex(2361),
        mesh->vertex(3043),
        mesh->vertex(2908),
        mesh->vertex(2659),
    };

    if (step == 0)
    {
        // gcs::VertexData<double> dist(geometry->mesh, 0.);
        // for (gcVertex v : vertexPts) {
        //     dist[v] += 1.;
        // }
        // gcs::VectorHeatMethodSolver vhmSolver(*geometry);
        // geometry->requireFaceAreas();
        // geometry->requireHalfedgeVectorsInVertex();
        // geometry->requireHalfedgeVectorsInFace();
        // //591, 2361, 3043, 2908, 2659
        // SurfacePoint initialGuess = mesh->vertex(3097);
        // gc::Vector2 thisUpdate = gc::Vector2::zero();
        // gcs::VertexData<gc::Vector2> logmap = vhmSolver.computeLogMap(initialGuess);
        // for (gcVertex v : mesh->vertices()) {
        //     gc::Vector2 pointCoord = logmap[v];
        //     thisUpdate += pointCoord;
        // }
        // thisUpdate /= T(mesh->nVertices());
        surface_points[5] = gcs::findCenter(*mesh, *geometry, vertexPts, 2).inSomeFace();
        // std::cout << toTV(center.faceCoords).transpose() << std::endl;
        // std::cout << toTV(center.interpolate(geometry->vertexPositions)).transpose() << std::endl;
        // std::cout << center.face.getIndex() << std::endl;
        // surface_points[5] = SurfacePoint(center.face, center.faceCoords).inSomeFace();
        return false;
    }
    return true;

    T h = 0.0;
    for (gcs::Edge edge : mesh->edges())
        h += geometry->edgeLength(edge);
    h /= T(mesh->nEdges());
    
    gcs::VectorHeatMethodSolver vhmSolver(*geometry, h*h);
    SurfacePoint si = surface_points[5];

    gcs::VertexData<gc::Vector2> logmap = vhmSolver.computeLogMap(si);
    gc::Vector2 v;
    for (int i = 0; i < surface_points.size() - 1; i++)
    {
        v += 1.0 / T(surface_points.size() - 1) * surface_points[i].interpolate(logmap);
    }
    gcs::TraceOptions options;
    options.includePath = true; 
    gcs::TraceGeodesicResult traceResult = traceGeodesic(*geometry, si, v, options);
    SurfacePoint candidatePoint = traceResult.endPoint.inSomeFace();
    surface_points[5] = candidatePoint;
    std::cout << v[0] << " " << v[1] << std::endl;
    if (v.norm() < newton_tol || step == max_newton_iter)
    {
        return true;
    }
    return false;
}

void IntrinsicSimulation::staticSolveLBFGS()
{
    
}

bool IntrinsicSimulation::advanceOneStep(int step)
{
    std::cout << "===================STEP " << step << "===================" << std::endl;
    // START_TIMING(NewtonStep)
    VectorXT residual; residual.resize(deformed.rows());
    residual.setZero();
    // START_TIMING(traceGeodesics)
    traceGeodesics();
    // FINISH_TIMING_PRINT(traceGeodesics)
    // START_TIMING(computeResidual)
    T residual_norm = computeResidual(residual);
    residual_norms.push_back(residual_norm);
    // FINISH_TIMING_PRINT(computeResidual)
    std::cout << "[NEWTON] iter " << step << "/" 
        << max_newton_iter << ": residual_norm " 
        << residual_norm << " tol: " << newton_tol << std::endl;
    
    if (residual_norm < newton_tol || step == max_newton_iter)
    {
        return true;
    }

    T du_norm = 1e10;
    // START_TIMING(lineSearchNewton)
    // dq_norm = lineSearchNewton(u, residual);
    du_norm = lineSearchNewton(residual);
    maximum_step_sizes.push_back(du_norm);

    if (use_lbfgs)
    {
        // sk = xk+1 - xk == delta_u
        // yk = gk+1 - gk == -residual - gi
        VectorXT g1 = VectorXT::Zero(residual.rows());
        computeResidual(g1);
        if (step == 0)
            lbfgs_solver.update(delta_u, VectorXT::Zero(residual.rows()));
        else
            lbfgs_solver.update(delta_u, -g1 - gi);
        gi = -g1;
    }
    
    // FINISH_TIMING_PRINT(lineSearchNewton)
    // FINISH_TIMING_PRINT(NewtonStep)
    if(step == max_newton_iter || du_norm > 1e10
    //  || du_norm < 1e-8
    )
    {
        // return true;
        std::cout << "ABNORMAL STOP with |du| " << du_norm << std::endl;
        std::ofstream out("search_direction.txt");
        out << std::setprecision(16);
        out << residual.rows() << std::endl;
        for (int i = 0; i < residual.rows(); i++)
            out << residual[i] << " ";
        out << std::endl;
        out << spring_edges.size() << std::endl;
        for (const Edge& edges : spring_edges)
        {
            out << edges[0] << " " << edges[1] << std::endl;
        }
        out << surface_points.size() << std::endl;
        for (int i = 0; i < surface_points.size(); i++)
        {
            out << toTV(surface_points[i].faceCoords).transpose() << std::endl;
            out << surface_points[i].face.getIndex();
            out << std::endl;
        }
        if (two_way_coupling)
        {
            out << deformed.rows() << std::endl;
            for (int i = 0; i < deformed.rows(); i++)
                out << deformed[i] << " ";
            out << std::endl;
        }
        out.close();
        return true;
    }
    return false;
}

T IntrinsicSimulation::lineSearchNewton(const VectorXT& residual)
{
    VectorXT du = residual;
    du.setZero();

    du = residual;

    // static VectorXT y0(residual.size());
    // if (lbfgs_first_step)
    // {
    //     y0 = -residual;
    //     lbfgs_first_step = true;
    // }

    StiffnessMatrix K(residual.rows(), residual.rows());
    if (use_Newton)
    {
        bool success = false;
        
        if (woodbury)
        {
            MatrixXT UV;
            // START_TIMING(build_system)
            buildSystemMatrixWoodbury(K, UV);
            // FINISH_TIMING_PRINT(build_system)
            // START_TIMING(linear_solve)
            success = linearSolveWoodbury(K, UV, residual, du);
            // FINISH_TIMING_PRINT(linear_solve)
        }
        else
        {
            // START_TIMING(build_system)
            buildSystemMatrix(K);
            // FINISH_TIMING_PRINT(build_system)
            // std::ofstream out("matrix.txt");
            // out << K << std::endl;
            // out.close();
            // std::exit(0);
            // START_TIMING(linear_solve)
            success = linearSolve(K, residual, du);
            // FINISH_TIMING_PRINT(linear_solve)
        }
        
        if (!success)
        {
            std::cout << "Linear Solve Failed" << std::endl;
            return 1e16;
        }
    }
    else if (use_lbfgs)
    {
        lbfgs_solver.apply(du);
        if (verbose)
            std::cout << "du dot -grad" << du.normalized().dot(residual.normalized()) << std::endl;
    }
    
    T norm = du.norm();
    if (verbose)
        std::cout << "\t|du | " << norm << std::endl;
    
    T E0 = computeTotalEnergy();
    std::cout << "obj: " << E0 << std::endl;

    auto lineSearchInDirection = [&](const VectorXT& direction, bool using_gradient) -> bool
    {
        int ls_max = 12;
        T alpha = computeInversionFreeStepsize();
        if (verbose)
            std::cout << "ls alpha start " << alpha  << std::endl;
        int cnt = 0;
        std::vector<SurfacePoint> current_state = surface_points;
        VectorXT current_state_x = deformed;
        while (true)
        {
            delta_u = alpha * direction;
            updateCurrentState(/*trace = */true);
            T E1 = computeTotalEnergy();
            if (verbose)
                std::cout << "\t[LS INFO] total energy: " << E1 << " #ls " << cnt << " |du| " << delta_u.norm() << std::endl;
            if (E1 - E0 < 0 || cnt > ls_max)
            {
                // std::cout << "|du| " << alpha * du.norm() << std::endl;
                // std::cout << "ls final " << E1 << " #ls " << cnt << std::endl;
                if (cnt > ls_max)
                {
                    if (!using_gradient)
                        std::cout << "-----line search max----- cnt > 15 switch to gradient" << std::endl;
                    else
                    {
                        std::cout << "-----line search max----- cnt > 15 along gradient [BUG ALERT]" << std::endl;
                    }
                    
                    // std::exit(0);
                    if (!using_gradient && use_Newton)
                    {
                        surface_points = current_state;
                        deformed = current_state_x;
                    }
                    return false;
                }
                return true;
            }
            alpha *= 0.5;
            cnt += 1;
            surface_points = current_state;
            deformed = current_state_x;
        }
    };
    bool ls_succeed = lineSearchInDirection(du, false);
    if (!ls_succeed && use_Newton)
        ls_succeed = lineSearchInDirection(residual, true);
    if (!ls_succeed)
    {
        if (!jump_out)
            return 1e16;
    }

    return std::abs(delta_u.maxCoeff());
}

void IntrinsicSimulation::reset()
{
    surface_points = surface_points_undeformed;
    delta_u.setZero();
    deformed = undeformed;
    updateCurrentState();
    residual_norms.resize(0);
}

void IntrinsicSimulation::checkInformation()
{
    
}

void IntrinsicSimulation::computeLogMap(const SurfacePoint& source_vtx, VectorXT& logmap)
{
    logmap.resize(mesh->nVertices() * 2);
    gcs::VectorHeatMethodSolver solver(*geometry, 1.0);
    gcs::VertexData<gc::Vector2> logmap_heat = solver.computeLogMap(source_vtx);
    for (gcs::Vertex v : mesh->vertices())
    {
        gc::Vector2 logmap_vi = logmap_heat[v];
        logmap[v.getIndex() * 2] = logmap_vi[0];
        logmap[v.getIndex() * 2 + 1] = logmap_vi[1];
    }
    
    // tbb::parallel_for(0, (int)mesh->nVertices(), [&](int vtx_idx){
    //     SurfacePoint va = SurfacePoint(mesh->vertex(vtx_idx)).inSomeFace();
    //     std::vector<SurfacePoint> path;
    //     T dis; std::vector<IxnData> ixn_data;
    //     computeExactGeodesic(source_vtx, va, dis, path, ixn_data, true);
        
    //     TV x0 = toTV(path[1].interpolate(geometry->vertexPositions));

    // });
}

void IntrinsicSimulation::findCenter(const std::vector<SurfacePoint>& points, SurfacePoint& center)
{
    VectorXT current_distance(points.size()); current_distance.setZero();
    std::vector<std::vector<SurfacePoint>> paths(points.size(), std::vector<SurfacePoint>());
    std::vector<std::vector<IxnData>> ixn_data(points.size(), std::vector<IxnData>());

    auto traceDistance = [&](SurfacePoint current, bool trace_path)
    {
        tbb::parallel_for(0, (int)points.size(), [&](int j)
        // for (int j = 0; j < points.size(); j++)
        {
            ixn_data[j].clear();
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

            SurfacePoint va_sub(sub_mesh->face(current.face.getIndex()), current.faceCoords);
            SurfacePoint vb_sub(sub_mesh->face(points[j].face.getIndex()), points[j].faceCoords);


            mmp.propagate(va_sub);
            if (trace_path)
            {
                paths[j] = mmp.traceBack(vb_sub, current_distance[j]);
                std::reverse(paths[j].begin(), paths[j].end());
                for (auto& pt : paths[j])
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
                            // std::exit(0);
                        }
                        test_interp = pt.tEdge * start + (1.0 - pt.tEdge) * end;
                        if ((ixn - test_interp).norm() > 1e-6)
                        {
                            std::cout << "error in interpolation" << std::endl;
                            // std::exit(0);
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
                            // std::exit(0);
                        }
                        edge_t = 1.0;
                    }
                    else
                    {
                        edge_t = 2.0;
                    }
                    
                    ixn_data[j].push_back(IxnData(start, end, (1.0-edge_t), start_end[0], start_end[1]));
                    pt.edge = mesh->edge(pt.edge.getIndex());
                    pt.vertex = mesh->vertex(pt.vertex.getIndex());
                    pt.face = mesh->face(pt.face.getIndex());
                    
                    pt = pt.inSomeFace();
                }
                TV v0 = toTV(paths[j][0].interpolate(geometry->vertexPositions));
                TV v1 = toTV(paths[j][paths[j].size() - 1].interpolate(geometry->vertexPositions));
                TV ixn0 = toTV(paths[j][1].interpolate(geometry->vertexPositions));
                TV ixn1 = toTV(paths[j][paths[j].size() - 2].interpolate(geometry->vertexPositions));
            }
            else
                current_distance[j] = mmp.getDistance(vb_sub);
        }
        );
    };

    auto computeEnergy = [&](SurfacePoint current) -> T
    {
        return 0.5 * current_distance.dot(current_distance);
    };

    auto computeResidual = [&](SurfacePoint current, TV2& grad) -> T
    {
        for (int i = 0; i < points.size(); i++)
        {
            T l = current_distance[i];
            std::vector<SurfacePoint> path = paths[i];

            int length = path.size();
    
            TV dldx0, dldx1;
            TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
            TV v1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));

            TV v10 = toTV(geometry->vertexPositions[current.face.halfedge().vertex()]);
            TV v11 = toTV(geometry->vertexPositions[current.face.halfedge().next().vertex()]);
            TV v12 = toTV(geometry->vertexPositions[current.face.halfedge().next().next().vertex()]);

            TV v20 = toTV(geometry->vertexPositions[points[i].face.halfedge().vertex()]);
            TV v21 = toTV(geometry->vertexPositions[points[i].face.halfedge().next().vertex()]);
            TV v22 = toTV(geometry->vertexPositions[points[i].face.halfedge().next().next().vertex()]);

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

                Vector<T, 6> dldx; dldx.setZero();
                dldx.segment<3>(0) = dldx0;
                dldx.segment<3>(3) = dldx1;
            }

            Vector<T, 6> dldx; dldx.setZero();
            dldx.segment<3>(0) = dldx0;
            dldx.segment<3>(3) = dldx1;

            Matrix<T, 6, 4> dxdw; dxdw.setZero();
            dxdw.block(0, 0, 3, 1) = (v10 - v12);
            dxdw.block(0, 1, 3, 1) = (v11 - v12);

            dxdw.block(3, 2, 3, 1) = (v20 - v22);
            dxdw.block(3, 3, 3, 1) = (v21 - v22);

            Vector<T, 4> dldw = dldx.transpose() * dxdw;
            grad += dldw.segment<2>(0);
        }
        return grad.norm();
    };

    

    traceDistance(center, true);
    // std::cout << current_distance.transpose() << std::endl;
    TV2 grad = TV2::Zero();
    for (int n_iter = 0; n_iter < 5; n_iter++)
    {
        T g_norm = computeResidual(center, grad);
        T E0 = computeEnergy(center);
        // std::cout << "|g| " << g_norm << " E0 " << E0 << std::endl;
        if (g_norm < 1e-6)
            break;
        // TM2 hess = TM2::Zero();
        T alpha = 1.0;
        SurfacePoint backup = center.inSomeFace();
        for (int ls_iter = 0; ls_iter < 5; ls_iter++)
        {
            center = backup;
            gcs::TraceOptions options; options.includePath = true; 
            gcFace fi = center.face;
            Vector3 start_bc = center.faceCoords;
            Vector3 trace_vec{grad[0],grad[1],0.0-grad[0]-grad[1]};
            trace_vec *= -alpha;
            gcs::TraceGeodesicResult result = gcs::traceGeodesic(*geometry, fi, start_bc, trace_vec, options);
            center = result.endPoint.inSomeFace();
            traceDistance(center, false);
            T E1 = computeEnergy(center);
            if (E1 < E0)
                break;
            else
                alpha *= 0.5;
        }
    }
    
}