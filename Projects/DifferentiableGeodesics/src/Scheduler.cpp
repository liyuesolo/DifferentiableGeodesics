
#include "geometrycentral/surface/surface_centers.h"
#include "../include/Scheduler.h"
#include "../include/Util.h"

void Scheduler::saveSurfacePoints(const std::string& filename, VoronoiCells& voronoi_cells)
{
    FILE *stream;
    if ((stream = fopen(filename.c_str(), "wb")) != NULL)
    {
        int len = voronoi_cells.voronoi_sites.rows();
        size_t temp;
        temp = fwrite(&len, sizeof(int), 1, stream);
        temp = fwrite(voronoi_cells.voronoi_sites.data(), sizeof(double), len, stream);
    }
    else 
    {
        std::cout << "Unable to write into file" << std::endl;
    }
    fclose(stream);
}

void Scheduler::saveSurfaceMesh(const std::string& filename, VoronoiCells& voronoi_cells)
{
    FILE *stream;
    if ((stream = fopen(filename.c_str(), "wb")) != NULL)
    {
        int len = voronoi_cells.extrinsic_vertices.rows();
        size_t temp;
        temp = fwrite(&len, sizeof(int), 1, stream);
        temp = fwrite(voronoi_cells.extrinsic_vertices.data(), sizeof(double), len, stream);
        len = voronoi_cells.extrinsic_indices.rows();
        temp = fwrite(&len, sizeof(int), 1, stream);
        temp = fwrite(voronoi_cells.extrinsic_indices.data(), sizeof(int), len, stream);
    }
    else 
    {
        std::cout << "Unable to write into file" << std::endl;
    }
    fclose(stream);
}

void Scheduler::saveVectorData(const std::string& filename, const VectorXT& data)
{
    FILE *stream;
    if ((stream = fopen(filename.c_str(), "wb")) != NULL)
    {
        int len = data.rows();
        size_t temp;
        temp = fwrite(&len, sizeof(int), 1, stream);
        temp = fwrite(data.data(), sizeof(double), len, stream);
    }
    else 
    {
        std::cout << "Unable to write into file" << std::endl;
    }
    fclose(stream);
}

void Scheduler::saveCoplanarMeshData(const std::string& filename, 
    const std::string& filename2, VoronoiCells& voronoi_cells)
{
    Eigen::MatrixXd _V,_C; Eigen::MatrixXi _F;
    voronoi_cells.generateMeshForRendering(_V, _F, _C);
    Eigen::VectorXd vertices, colors; Eigen::VectorXi faces;
    iglMatrixFatten<T, 3>(_V, vertices);
    colors = _C.col(0);
    iglMatrixFatten<int, 3>(_F, faces);
    FILE *stream;
    if ((stream = fopen(filename.c_str(), "wb")) != NULL)
    {
        int len = vertices.rows();
        size_t temp;
        temp = fwrite(&len, sizeof(int), 1, stream);
        temp = fwrite(vertices.data(), sizeof(double), len, stream);
        len = faces.rows();
        temp = fwrite(&len, sizeof(int), 1, stream);
        temp = fwrite(faces.data(), sizeof(int), len, stream);
    }
    else 
    {
        std::cout << "Unable to write into file" << std::endl;
    }
    fclose(stream);
    if ((stream = fopen(filename2.c_str(), "wb")) != NULL)
    {
        int len = colors.rows();
        size_t temp;
        temp = fwrite(&len, sizeof(int), 1, stream);
        temp = fwrite(colors.data(), sizeof(double), len, stream);
    }
    else 
    {
        std::cout << "Unable to write into file" << std::endl;
    }
    fclose(stream);
}
void Scheduler::saveNetwork(const std::string& filename, VoronoiCells& voronoi_cells)
{
    FILE *stream;
    VectorXT all_data(voronoi_cells.voronoi_edges.size() * 6);
    for (int i = 0; i < voronoi_cells.voronoi_edges.size(); i++)
    {
        all_data.segment<3>(i * 6 + 0) = voronoi_cells.voronoi_edges[i].first;
        all_data.segment<3>(i * 6 + 3) = voronoi_cells.voronoi_edges[i].second;
    }
    
    if ((stream = fopen(filename.c_str(), "wb")) != NULL)
    {
        int len = all_data.rows();
        size_t temp;
        temp = fwrite(&len, sizeof(int), 1, stream);
        temp = fwrite(all_data.data(), sizeof(double), len, stream);
    }
    else 
    {
        std::cout << "Unable to write into file" << std::endl;
    }
    fclose(stream);
}

void Scheduler::saveSurfacePoints(const std::string& filename, IntrinsicSimulation& intrinsic_simulation)
{
    FILE *stream;
    int n_surface_pt = intrinsic_simulation.surface_points.size();
    VectorXT sites(n_surface_pt*3);
    for (int j = 0; j < n_surface_pt; j++)
    {
        sites.segment<3>(j * 3) = intrinsic_simulation.toTV(intrinsic_simulation.surface_points[j].interpolate(intrinsic_simulation.geometry->vertexPositions));
    }
    if ((stream = fopen(filename.c_str(), "wb")) != NULL)
    {
        int len = sites.rows();
        size_t temp;
        temp = fwrite(&len, sizeof(int), 1, stream);
        temp = fwrite(sites.data(), sizeof(double), len, stream);
    }
    else 
    {
        std::cout << "Unable to write into file" << std::endl;
    }
    fclose(stream);
}

void Scheduler::saveSurfaceMesh(const std::string& filename, IntrinsicSimulation& intrinsic_simulation)
{
    Eigen::MatrixXd V, C; Eigen::MatrixXi F;
    intrinsic_simulation.generateMeshForRendering(V, F, C);
    VectorXT vtx_pos, face_color; Eigen::VectorXi face_idx;
    iglMatrixFatten<T, 3>(V, vtx_pos);
    iglMatrixFatten<int, 3>(F, face_idx);
    FILE *stream;
    if ((stream = fopen(filename.c_str(), "wb")) != NULL)
    {
        int len = vtx_pos.rows();
        size_t temp;
        temp = fwrite(&len, sizeof(int), 1, stream);
        temp = fwrite(vtx_pos.data(), sizeof(double), len, stream);
        len = face_idx.rows();
        temp = fwrite(&len, sizeof(int), 1, stream);
        temp = fwrite(face_idx.data(), sizeof(int), len, stream);
    }
    else 
    {
        std::cout << "Unable to write into file" << std::endl;
    }
    fclose(stream);
}

void Scheduler::saveNetwork(const std::string& filename, IntrinsicSimulation& intrinsic_simulation)
{
    std::vector<TV> colors;
    intrinsic_simulation.updateVisualization(colors);
    FILE *stream;
    VectorXT all_data(intrinsic_simulation.all_intrinsic_edges.size() * 6);
    for (int j = 0; j < intrinsic_simulation.all_intrinsic_edges.size(); j++)
    {
        all_data.segment<3>(j * 6 + 0) = intrinsic_simulation.all_intrinsic_edges[j].first;
        all_data.segment<3>(j * 6 + 3) = intrinsic_simulation.all_intrinsic_edges[j].second;
    }
    
    if ((stream = fopen(filename.c_str(), "wb")) != NULL)
    {
        int len = all_data.rows();
        size_t temp;
        temp = fwrite(&len, sizeof(int), 1, stream);
        temp = fwrite(all_data.data(), sizeof(double), len, stream);
    }
    else 
    {
        std::cout << "Unable to write into file" << std::endl;
    }
    fclose(stream);
}

void Scheduler::execute(bool server, bool save_result)
{
    if (exp == "run_karcher_mean_optimization")
    {
        bool use_lbfgs = false;
        std::vector<std::string> filename_list = { 
            "sphere642", 
            "screwdriver", 
            "linkCupTop", "ear_simplified", "protein", "ear"
        };
        std::string suffix = use_lbfgs ? "_lbfgs" : "";
        for (int i = 0; i < filename_list.size(); i++)
        {
            // if (i < 6)
            //     continue;
            std::string mesh = filename_list[i];
            std::cout << "Processing mesh " << mesh << std::endl;
            IntrinsicSimulation intrinsic_simulation;
            intrinsic_simulation.use_lbfgs = use_lbfgs;
            intrinsic_simulation.use_Newton = !use_lbfgs;
            
            intrinsic_simulation.max_newton_iter = 200;
            intrinsic_simulation.initializeKarcherMeanScene(base_folder + "/data/" + mesh + ".obj", i);
            int j = 0;
            START_TIMING(karchermean_total)
            for (; j < intrinsic_simulation.max_newton_iter; j++)
            {
                if (save_result)
                    intrinsic_simulation.saveStates(base_folder + "results/KarcherMean/"+mesh+"/iter_"+std::to_string(j)+suffix+".txt");       
                
                bool converged = intrinsic_simulation.advanceOneStep(j);
                
                if (converged)
                    break;
            }
            // intrinsic_simulation.staticSolveLBFGS();
            FINISH_TIMING_PRINT(karchermean_total)
            // std::getchar();
            if (!save_result)
                continue;
            
            std::ofstream out(base_folder + "results/KarcherMean/"+mesh+"/config"+suffix+".txt");
            out << j << std::endl;
            out.close();

            std::vector<gcs::SurfacePoint> trajectory;
            std::ifstream in(base_folder + "results/KarcherMean/"+mesh+"/config"+suffix+".txt");
            int n_iter; in >> n_iter; in.close();
            for (int j = 0; j < n_iter + 1; j++)
            {
                in.open(base_folder + "results/KarcherMean/"+mesh+"/iter_"+std::to_string(j)+suffix+".txt");
                int n_pt, face_idx; T u, v, w; 
                in >> n_pt;
                for (int k = 0; k < n_pt; k++)
                {
                    in >> u >> v >> w >> face_idx;
                    if (k == n_pt - 1)
                    {
                        trajectory.push_back(
                            gcs::SurfacePoint(intrinsic_simulation.mesh->face(face_idx),
                            gc::Vector3{u, v, w}));
                    }
                }
                in.close();
            }
            std::vector<TV> points;
            for (int j = 0; j < n_iter; j++)
            {
                std::vector<gcs::SurfacePoint> path;   
                intrinsic_simulation.computeExactGeodesicPath(trajectory[j], trajectory[j+1], path);
                for (int k = 0; k < path.size() - 1; k++)
                {
                    points.push_back(intrinsic_simulation.toTV(path[k].interpolate(intrinsic_simulation.geometry->vertexPositions)));                    
                    points.push_back(intrinsic_simulation.toTV(path[k+1].interpolate(intrinsic_simulation.geometry->vertexPositions)));                    
                }   
            }
            FILE *stream;
            VectorXT steps(trajectory.size() * 3);
            for (int j = 0; j < trajectory.size(); j++)
            {
                steps.segment<3>(j * 3) = intrinsic_simulation.toTV(trajectory[j].interpolate(intrinsic_simulation.geometry->vertexPositions));
            }
            if ((stream = fopen((base_folder + "results/KarcherMean/"+mesh+"/iteration"+suffix+".data").c_str(), "wb")) != NULL)
            {
                int len = steps.rows();
                size_t temp;
                temp = fwrite(&len, sizeof(int), 1, stream);
                temp = fwrite(steps.data(), sizeof(double), len, stream);
            }
            else 
            {
                std::cout << "Unable to write into file" << std::endl;
            }
            fclose(stream);

            VectorXT all_data(points.size() * 3);
            for (int j = 0; j < points.size(); j++)
            {
                all_data.segment<3>(j * 3) = points[j];
            }
            
            if ((stream = fopen((base_folder + "results/KarcherMean/"+mesh+"/curves"+suffix+".data").c_str(), "wb")) != NULL)
            {
                int len = all_data.rows();
                size_t temp;
                temp = fwrite(&len, sizeof(int), 1, stream);
                temp = fwrite(all_data.data(), sizeof(double), len, stream);
            }
            else 
            {
                std::cout << "Unable to write into file" << std::endl;
            }
            fclose(stream);
            // save surface mesh
            if ((stream = fopen((base_folder + "results/KarcherMean/"+mesh+"/surface_mesh"+suffix+".data").c_str(), "wb")) != NULL)
            {
                int len = intrinsic_simulation.extrinsic_vertices.rows();
                size_t temp;
                temp = fwrite(&len, sizeof(int), 1, stream);
                temp = fwrite(intrinsic_simulation.extrinsic_vertices.data(), sizeof(double), len, stream);
                len = intrinsic_simulation.extrinsic_indices.rows();
                temp = fwrite(&len, sizeof(int), 1, stream);
                temp = fwrite(intrinsic_simulation.extrinsic_indices.data(), sizeof(int), len, stream);
            }
            else 
            {
                std::cout << "Unable to write into file" << std::endl;
            }
            fclose(stream);
            VectorXT sites(15);
            for (int j = 0; j < 5; j++)
            {
                sites.segment<3>(j * 3) = intrinsic_simulation.toTV(intrinsic_simulation.surface_points[j].interpolate(intrinsic_simulation.geometry->vertexPositions));
            }
            
            if ((stream = fopen((base_folder + "results/KarcherMean/"+mesh+"/sites"+suffix+".data").c_str(), "wb")) != NULL)
            {
                int len = sites.rows();
                size_t temp;
                temp = fwrite(&len, sizeof(int), 1, stream);
                temp = fwrite(sites.data(), sizeof(double), len, stream);
            }
            else 
            {
                std::cout << "Unable to write into file" << std::endl;
            }
            fclose(stream);
        }   
    }
    else if (exp == "expanding_torus")
    {
        IntrinsicSimulation intrinsic_simulation;
        intrinsic_simulation.initializeTorusExpandingScene();
        if(server)
        {
            intrinsic_simulation.max_newton_iter = 30;
            auto saveData = [&](int iter, bool geo_tri)
            {
                std::string suffix = geo_tri ? "_geo_tri" : "";
                std::vector<TV> colors;
                intrinsic_simulation.updateVisualization(colors);
                VectorXT sites;
                intrinsic_simulation.getAllPointsPosition(sites);
                FILE *stream;
                if ((stream = fopen((base_folder + "results/torus_deformation/sites_step"+std::to_string(iter)+suffix+".data").c_str(), "wb")) != NULL)
                {
                    int len = sites.rows();
                    size_t temp;
                    temp = fwrite(&len, sizeof(int), 1, stream);
                    temp = fwrite(sites.data(), sizeof(double), len, stream);
                }
                else 
                {
                    std::cout << "Unable to write into file" << std::endl;
                }
                fclose(stream);
                if ((stream = fopen((base_folder + "results/torus_deformation/surface_mesh_step"+std::to_string(iter)+suffix+".data").c_str(), "wb")) != NULL)
                {
                    int len = intrinsic_simulation.extrinsic_vertices.rows();
                    size_t temp;
                    temp = fwrite(&len, sizeof(int), 1, stream);
                    temp = fwrite(intrinsic_simulation.extrinsic_vertices.data(), sizeof(double), len, stream);
                    len = intrinsic_simulation.extrinsic_indices.rows();
                    temp = fwrite(&len, sizeof(int), 1, stream);
                    temp = fwrite(intrinsic_simulation.extrinsic_indices.data(), sizeof(int), len, stream);
                }
                else 
                {
                    std::cout << "Unable to write into file" << std::endl;
                }
                fclose(stream);
                Eigen::MatrixXd _V, _C; Eigen::MatrixXi _F;
                VectorXT face_quantity;
                intrinsic_simulation.assignFaceColorBasedOnGeodesicTriangle(_V, _F, _C, face_quantity);
                VectorXT vtx_pos, face_color; Eigen::VectorXi face_idx;
                iglMatrixFatten<T, 3>(_V, vtx_pos); iglMatrixFatten<T, 3>(_C, face_color);
                iglMatrixFatten<int, 3>(_F, face_idx);
                if ((stream = fopen((base_folder + "results/torus_deformation/surface_mesh_step"+std::to_string(iter)+suffix+"_remeshed.data").c_str(), "wb")) != NULL)
                {
                    int len = vtx_pos.rows();
                    size_t temp;
                    temp = fwrite(&len, sizeof(int), 1, stream);
                    temp = fwrite(vtx_pos.data(), sizeof(double), len, stream);
                    len = face_idx.rows();
                    temp = fwrite(&len, sizeof(int), 1, stream);
                    temp = fwrite(face_idx.data(), sizeof(int), len, stream);
                }
                else 
                {
                    std::cout << "Unable to write into file" << std::endl;
                }
                fclose(stream);
                if ((stream = fopen((base_folder + "results/torus_deformation/surface_mesh_color_step"+std::to_string(iter)+suffix+".data").c_str(), "wb")) != NULL)
                {
                    int len = face_quantity.rows();
                    size_t temp;
                    temp = fwrite(&len, sizeof(int), 1, stream);
                    temp = fwrite(face_quantity.data(), sizeof(double), len, stream);
                }
                else 
                {
                    std::cout << "Unable to write into file" << std::endl;
                }
                fclose(stream);
                VectorXT all_data(intrinsic_simulation.all_intrinsic_edges.size() * 6);
                for (int j = 0; j < intrinsic_simulation.all_intrinsic_edges.size(); j++)
                {
                    all_data.segment<3>(j * 6 + 0) = intrinsic_simulation.all_intrinsic_edges[j].first;
                    all_data.segment<3>(j * 6 + 3) = intrinsic_simulation.all_intrinsic_edges[j].second;
                }
                
                if ((stream = fopen((base_folder + "results/torus_deformation/curves_step"+std::to_string(iter)+suffix+".data").c_str(), "wb")) != NULL)
                {
                    int len = all_data.rows();
                    size_t temp;
                    temp = fwrite(&len, sizeof(int), 1, stream);
                    temp = fwrite(all_data.data(), sizeof(double), len, stream);
                }
                else 
                {
                    std::cout << "Unable to write into file" << std::endl;
                }
                fclose(stream);
            };

            bool add_elasticity = true;
            int offset = 1;
            if (save_result)
                saveData(0, add_elasticity);
            for (int i = offset; i < 51; i++)
            {
                std::cout << "#################### iter " << i << "/" << 51 << "####################" << std::endl;
                intrinsic_simulation.expandBaseMesh(0.005);
                std::vector<TV> colors;
                intrinsic_simulation.updateVisualization(colors);
                int j = 0;
                if (add_elasticity)
                    for (; j < intrinsic_simulation.max_newton_iter; j++)
                    {
                        bool converged = intrinsic_simulation.advanceOneStep(j);
                        if (converged)
                            break;
                    }
                if (save_result)
                    saveData(i-offset, add_elasticity);
                
                // std::cout << "#iter " << j << std::endl;
                // std::getchar();
            }
        }
        else
        {
            GeodesicSimApp app(intrinsic_simulation);    
            app.initializeScene();
            app.run();
        }
        
    }
    else if (exp == "generate_intrinsic_triangulation")
    {        
        std::vector<std::string> filename_list = {
            // "bunny", 
            // "3holes_simplified", 
            // "cactus_simplified",
            //  "rocker_arm_simplified"
            "donut_duck"
            // "sphere642"
        };
        for (auto filename : filename_list)
        {
            VoronoiCells voronoi_cells;
            voronoi_cells.loadGeometry(base_folder + "/data/" + filename + ".obj");
            voronoi_cells.sampleSitesPoissonDisk(0.4, /*perturb = */false);
            // voronoi_cells.sampleSitesByFace(Vector<T, 2>(0.6,0.2), 5);
            voronoi_cells.constructVoronoiDiagram(false, false);
            std::vector<std::pair<TV, TV>> idt_edges;
            std::vector<IV> idt_indices;
            voronoi_cells.computeDualIDT(idt_edges, idt_indices);
            std::ofstream out(base_folder + "/data/" + filename +"_idt.txt");
            out << idt_indices.size() << std::endl;
            for (const IV& tri : idt_indices)
                out << tri[0] << " " << tri[1] << " " << tri[2] << " " << std::endl;
            out.close();
            voronoi_cells.saveVoronoiDiagram(base_folder + "/data/" + filename + "_samples.txt");
        }
    }
    else if (exp == "voronoi_sim")
    {
        std::vector<std::string> mesh_files = { 
            "fertility" ,
            "spot",
            "armadillo"
            };
        std::vector<int> resolution = {3,2,2};
        for (int i = 0; i < mesh_files.size(); i++)
        {
            VoronoiCells voronoi_cells;
            VoronoiApp app(voronoi_cells);    
            voronoi_cells.add_centroid = true;
            voronoi_cells.edge_weighting = false;
            voronoi_cells.use_MMA = false;
            voronoi_cells.use_Newton = false;
            voronoi_cells.use_lbfgs = true;
            voronoi_cells.max_newton_iter = 5;
        
            // voronoi_cells.loadGeometry("../../../Projects/DifferentiableGeodesics/data/armadillo.obj");
            voronoi_cells.loadGeometry(base_folder+"/data/"+mesh_files[i]+".obj");
            // voronoi_cells.loadGeometry("../../../Projects/DifferentiableGeodesics/data/dragon20k.obj");
            // voronoi_cells.sampleSitesPoissonDisk(4.0, /*perturb = */false);
            // voronoi_cells.loadGeometry("../../../Projects/DifferentiableGeodesics/data/dragon20k.obj");
            Vector<T, 2> bary(0.8, 0.1);
            voronoi_cells.sampleSitesByFace(bary, resolution[i]);
            voronoi_cells.constructVoronoiDiagram(true, false);
            
            for (int j = 0; j < voronoi_cells.max_newton_iter; j++)
            {
                if (save_result)
                    app.saveMesh(base_folder+"/results/"+mesh_files[i], j);
                bool converged = voronoi_cells.advanceOneStep(j);
                if (converged)
                    break;
            }
        }
        
    }
    else if (exp == "voronoi_sim_viewer")
    {
        VoronoiCells voronoi_cells;
        VoronoiApp app(voronoi_cells);    
        app.initializeScene();
        app.run();
    }
    else if (exp == "run_coplanar_optimization")
    {
        
        if (server)
        {
            std::vector<std::string> mesh_files = { 
                "sphere642",
                "donut_duck",
                "torus"
                };
            std::vector<T> resolution = {1.0, 1.0, 1.0};
            // std::vector<int> resolution = {3, 3, 3, 0, 0};
            for (int i = 0; i < mesh_files.size(); i++)
            {
                
                VoronoiCells voronoi_cells;
                VoronoiApp app(voronoi_cells);    
                voronoi_cells.add_coplanar = true;
                // voronoi_cells.add_length = true;
                // voronoi_cells.w_len = 0.1;
                // voronoi_cells.edge_weighting = false;
                // voronoi_cells.use_MMA = true;
                // voronoi_cells.max_mma_iter = 500;
                
                voronoi_cells.use_lbfgs = true;
                voronoi_cells.use_Newton = false;
                voronoi_cells.max_newton_iter = 5; // for timing

                // voronoi_cells.loadGeometry("../../../Projects/DifferentiableGeodesics/data/armadillo.obj");
                voronoi_cells.loadGeometry(base_folder+"/data/"+mesh_files[i]+".obj");
                voronoi_cells.sampleSitesPoissonDisk(resolution[i], false);
                // Vector<T, 2> bary(0.4, 0.3);
                // voronoi_cells.sampleSitesByFace(bary, resolution[i]);
                voronoi_cells.constructVoronoiDiagram(true, false);
                if (voronoi_cells.add_length)
                    voronoi_cells.computeReferenceLength();
                
                std::ofstream out;
                if (save_result)
                    out.open(base_folder+"/results/coplanar/"+mesh_files[i]+"/statistics.txt");
                for (int j = 0; j < voronoi_cells.max_mma_iter; j++)
                {
                    if (save_result)
                    {
                        saveSurfacePoints(base_folder + "/results/coplanar/"+mesh_files[i]+"/" + "/iter_"+std::to_string(j)+"_sites.data", voronoi_cells);
                        saveSurfaceMesh(base_folder + "/results/coplanar/"+mesh_files[i]+"/" + "/iter_"+std::to_string(j)+"_surface_mesh.data", voronoi_cells);
                        saveCoplanarMeshData(base_folder + "/results/coplanar/"+mesh_files[i]+"/" + "/iter_"+std::to_string(j)+"_planar_mesh.data", 
                            base_folder + "/results/coplanar/"+mesh_files[i]+"/" + "/iter_"+std::to_string(j)+"_planar_mesh_color.data",
                            voronoi_cells);
                        saveNetwork(base_folder + "/results/coplanar/"+mesh_files[i]+"/" + "/iter_"+std::to_string(j)+"_curves.data", voronoi_cells);
                        
                        T mean_val, max_val;
                        voronoi_cells.checkCoplanarity(mean_val, max_val);
                        out << "iter " << j << " mean " << mean_val << " max " << max_val << std::endl;
                    }
                    bool converged = voronoi_cells.advanceOneStep(j);
                    if (converged)
                        break;
                }
                if (save_result)
                    out.close();
            }
        }
        else
        {
            VoronoiCells voronoi_cells;
            VoronoiApp app(voronoi_cells);    
            voronoi_cells.add_coplanar = true;
            voronoi_cells.edge_weighting = false;
            voronoi_cells.use_MMA = true;
            app.initializeScene();
            app.run();
        }
    }
    else if (exp == "run_same_length_optimization")
    {
        if (server)
        {
            std::vector<std::string> mesh_files = { 
                "sphere642",
                "cactus_simplified",
                "donut_duck"
                };
            std::vector<T> resolution = {2.0, 2.0, 2.0};
            for (int i = 0; i < mesh_files.size(); i++)
            {
                VoronoiCells voronoi_cells;
                VoronoiApp app(voronoi_cells);    
                voronoi_cells.add_length = true;
                voronoi_cells.add_centroid = true;
                voronoi_cells.w_centroid = 0.001;
                voronoi_cells.use_MMA = false;
                voronoi_cells.use_Newton = false;
                voronoi_cells.use_lbfgs = true;
                voronoi_cells.max_newton_iter = 1;
                voronoi_cells.max_mma_iter = 500;

                
                voronoi_cells.loadGeometry(base_folder+"/data/"+mesh_files[i]+".obj");
                voronoi_cells.sampleSitesPoissonDisk(resolution[i], false);
                voronoi_cells.constructVoronoiDiagram(true, false);
                voronoi_cells.computeReferenceLength();
                
                std::ofstream out(base_folder+"/results/same_length/"+mesh_files[i]+"/statistics.txt");
                for (int j = 0; j < voronoi_cells.max_mma_iter; j++)
                {
                    if (save_result)
                    {
                        saveSurfacePoints(base_folder + "/results/same_length/"+mesh_files[i]+"/" + "/iter_"+std::to_string(j)+"_sites.data", voronoi_cells);
                        saveSurfaceMesh(base_folder + "/results/same_length/"+mesh_files[i]+"/" + "/iter_"+std::to_string(j)+"_surface_mesh.data", voronoi_cells);
                        saveNetwork(base_folder + "/results/same_length/"+mesh_files[i]+"/" + "/iter_"+std::to_string(j)+"_curves.data", voronoi_cells);
                    }
                    T mean_val, max_val;
                    voronoi_cells.checkLengthObjective(mean_val, max_val);
                    out << "iter " << j << " mean " << mean_val << " max " << max_val << std::endl;
                    bool converged = voronoi_cells.advanceOneStep(j);
                    if (converged)
                        break;
                }
                out.close();
            }
        }
        else
        {
            VoronoiCells voronoi_cells;
            VoronoiApp app(voronoi_cells);    
            voronoi_cells.add_coplanar = false;
            voronoi_cells.add_length = true;
            voronoi_cells.edge_weighting = false;
            voronoi_cells.use_MMA = true;
            app.initializeScene();
            app.run();
        }
    }
    else if (exp == "intrinsic_sim_viewer")
    {
        IntrinsicSimulation intrinsic_simulation;
        intrinsic_simulation.initializeMassSpringSceneExactGeodesic("sphere642", 5.0);
        GeodesicSimApp app(intrinsic_simulation);    
        app.initializeScene();
        app.run();
    }
    else if (exp == "KarcherMeanSharp")
    {
        std::vector<std::string> filename_list = { 
            "sphere642", 
            "screwdriver", 
            "linkCupTop",
            "ear_simplified",
            "protein",
            "ear"
        };
        for (int i = 0; i < filename_list.size(); i++)
        {
            
            std::string mesh = filename_list[i];
            std::cout << "Processing mesh " << mesh << std::endl;
            IntrinsicSimulation intrinsic_simulation;
            intrinsic_simulation.initializeKarcherMeanScene(base_folder + "/data/" + mesh + ".obj", i);
            gcs::VectorHeatMethodSolver vhmSolver(*intrinsic_simulation.geometry);
            intrinsic_simulation.geometry->requireFaceAreas();
            intrinsic_simulation.geometry->requireHalfedgeVectorsInVertex();
            intrinsic_simulation.geometry->requireHalfedgeVectorsInFace();

            intrinsic_simulation.max_newton_iter = 200;
            T convergeThresh = 1 / 3.;
            int j = 0;
            
            auto evalEnergyAndUpdate = [&](gcs::SurfacePoint aboutPoint) -> std::tuple<T, gc::Vector2> 
            {
                // Compute the current log map
                // Solve at the face point
                gcs::VertexData<gc::Vector2> logmap = vhmSolver.computeLogMap(aboutPoint);

                // Evaluate energy and update step
                T thisEnergy = 0.;
                gc::Vector2 thisUpdate = gc::Vector2::zero();
                
                for (int i = 0; i < intrinsic_simulation.surface_points.size() - 1; i++)
                {
                    gc::Vector2 pointCoord = intrinsic_simulation.surface_points[i].interpolate(logmap);
                    thisUpdate += pointCoord;
                    thisEnergy += pointCoord.norm2();
                }
                thisUpdate /= T(intrinsic_simulation.surface_points.size()-1);

                return std::make_tuple(thisEnergy, thisUpdate);
            };
            
            bool converged = false;
            START_TIMING(karchermean_total)
            std::vector<T> residual_norm;
            std::vector<T> energies;
            for (; j < intrinsic_simulation.max_newton_iter; j++)
            {
                intrinsic_simulation.saveStates(base_folder + "results/KarcherMean/"+mesh+"/iter_"+std::to_string(j)+"_Sharp.txt");    
                intrinsic_simulation.KarcherMeanOneStep(j);
                T energyBefore; gc::Vector2 updateVec;
                std::tie(energyBefore, updateVec) = evalEnergyAndUpdate(intrinsic_simulation.surface_points[5]);
                // std::cout << "|g| " <<  updateVec.norm() << std::endl;
                residual_norm.push_back(updateVec.norm());
                energies.push_back(energyBefore);
                double stepSize = 1.0;
                int lineSearchIter = 0;
                int ls_max = 8;
                for (; lineSearchIter < ls_max; lineSearchIter++) 
                {
                    // Try taking a step
                    gc::Vector2 stepVec = updateVec * stepSize;
                    gcs::TraceOptions options;
                    options.includePath = true; 
                    gcs::TraceGeodesicResult traceResult = traceGeodesic(*intrinsic_simulation.geometry, 
                        intrinsic_simulation.surface_points[5], stepVec, options);
                    gcs::SurfacePoint candidatePoint = traceResult.endPoint.inSomeFace();
                    

                    // Compute new energy
                    T newEnergy = std::get<0>(evalEnergyAndUpdate(candidatePoint));
                    // std::cout << std::setprecision(10) << energyBefore << " " << newEnergy << std::endl;
                    // std::getchar();
                    // Check for convergence
                    T faceScale = std::sqrt(intrinsic_simulation.geometry->faceAreas[intrinsic_simulation.surface_points[5].inSomeFace().face]);
                    // if (updateVec.norm() < 1e-6)
                    if (stepVec.norm() < convergeThresh * faceScale) 
                    {
                        converged = true;
                        intrinsic_simulation.surface_points[5] = candidatePoint;
                        break;
                    }

                    // Accept step if good
                    if (newEnergy < energyBefore) {
                        intrinsic_simulation.surface_points[5] = candidatePoint;
                        break;
                    }
                    else if (lineSearchIter == ls_max-1)
                    {
                        intrinsic_simulation.surface_points[5] = candidatePoint;
                        break;
                    }

                    // Otherwise decrease step size and repeat
                    stepSize *= 0.5;
                }
                
                
                if (converged) 
                {
                    break;
                }
                // if (updateVec.norm() < 1e-6)
                //     break;
            }
            FINISH_TIMING_PRINT(karchermean_total)
            std::cout << "# iter " << j << " |g| " << residual_norm[residual_norm.size()-1] 
                << " obj: " << energies[energies.size() - 1] << std::endl;
            continue;
            intrinsic_simulation.geometry->unrequireFaceAreas();
            intrinsic_simulation.geometry->unrequireHalfedgeVectorsInVertex();
            intrinsic_simulation.geometry->unrequireHalfedgeVectorsInFace();

            std::ofstream out(base_folder + "results/KarcherMean/"+mesh+"/config_Sharp.txt");
            out << j << std::endl;
            for (T norm : residual_norm)
            {
                out << std::setprecision(10) << norm << " ";
            }
            out.close();
            
            std::vector<gcs::SurfacePoint> trajectory;
            std::ifstream in(base_folder + "results/KarcherMean/"+mesh+"/config_Sharp.txt");
            int n_iter; in >> n_iter; 
            in.close();
            for (int j = 0; j < n_iter + 1; j++)
            {
                
                in.open(base_folder + "results/KarcherMean/"+mesh+"/iter_"+std::to_string(j)+"_Sharp.txt");
                int n_pt, face_idx; T u, v, w; 
                in >> n_pt;
                for (int k = 0; k < n_pt; k++)
                {
                    in >> u >> v >> w >> face_idx;
                    if (k == n_pt - 1)
                    {
                        trajectory.push_back(
                            gcs::SurfacePoint(intrinsic_simulation.mesh->face(face_idx),
                            gc::Vector3{u, v, w}));
                    }
                }
                in.close();
            }
            
            std::vector<TV> points;
            for (int j = 0; j < n_iter; j++)
            {
                std::vector<gcs::SurfacePoint> path;   
                intrinsic_simulation.computeExactGeodesicPath(trajectory[j], trajectory[j+1], path);
                
                for (int k = 0; k < path.size() - 1; k++)
                {
                    points.push_back(intrinsic_simulation.toTV(path[k].interpolate(intrinsic_simulation.geometry->vertexPositions)));                    
                    points.push_back(intrinsic_simulation.toTV(path[k+1].interpolate(intrinsic_simulation.geometry->vertexPositions)));                    
                }   
            }
            // std::cout << "#points " << points.size() << std::endl;
            FILE *stream;
            VectorXT steps(trajectory.size() * 3);
            for (int j = 0; j < trajectory.size(); j++)
            {
                steps.segment<3>(j * 3) = intrinsic_simulation.toTV(trajectory[j].interpolate(intrinsic_simulation.geometry->vertexPositions));
            }
            if ((stream = fopen((base_folder + "results/KarcherMean/"+mesh+"/iteration_Sharp.data").c_str(), "wb")) != NULL)
            {
                int len = steps.rows();
                size_t temp;
                temp = fwrite(&len, sizeof(int), 1, stream);
                temp = fwrite(steps.data(), sizeof(double), len, stream);
            }
            else 
            {
                std::cout << "Unable to write into file" << std::endl;
            }
            fclose(stream);

            VectorXT all_data(points.size() * 3);
            for (int j = 0; j < points.size(); j++)
            {
                all_data.segment<3>(j * 3) = points[j];
            }
            
            if ((stream = fopen((base_folder + "results/KarcherMean/"+mesh+"/curves_Sharp.data").c_str(), "wb")) != NULL)
            {
                int len = all_data.rows();
                size_t temp;
                temp = fwrite(&len, sizeof(int), 1, stream);
                temp = fwrite(all_data.data(), sizeof(double), len, stream);
            }
            else 
            {
                std::cout << "Unable to write into file" << std::endl;
            }
            fclose(stream);
            // save surface mesh
            if ((stream = fopen((base_folder + "results/KarcherMean/"+mesh+"/surface_mesh_Sharp.data").c_str(), "wb")) != NULL)
            {
                int len = intrinsic_simulation.extrinsic_vertices.rows();
                size_t temp;
                temp = fwrite(&len, sizeof(int), 1, stream);
                temp = fwrite(intrinsic_simulation.extrinsic_vertices.data(), sizeof(double), len, stream);
                len = intrinsic_simulation.extrinsic_indices.rows();
                temp = fwrite(&len, sizeof(int), 1, stream);
                temp = fwrite(intrinsic_simulation.extrinsic_indices.data(), sizeof(int), len, stream);
            }
            else 
            {
                std::cout << "Unable to write into file" << std::endl;
            }
            fclose(stream);
            VectorXT sites(15);
            for (int j = 0; j < 5; j++)
            {
                sites.segment<3>(j * 3) = intrinsic_simulation.toTV(intrinsic_simulation.surface_points[j].interpolate(intrinsic_simulation.geometry->vertexPositions));
            }
            
            if ((stream = fopen((base_folder + "results/KarcherMean/"+mesh+"/sites_Sharp.data").c_str(), "wb")) != NULL)
            {
                int len = sites.rows();
                size_t temp;
                temp = fwrite(&len, sizeof(int), 1, stream);
                temp = fwrite(sites.data(), sizeof(double), len, stream);
            }
            else 
            {
                std::cout << "Unable to write into file" << std::endl;
            }
            fclose(stream);
        }   
    }
    else if (exp == "two_way_bunny")
    {
        std::string exp_folder = base_folder + "results/twowaybunny/";
        IntrinsicSimulation intrinsic_simulation;
        intrinsic_simulation.initializeTwowayCouplingScene();
        if (server)
        {
            for (int j = 0; j < intrinsic_simulation.max_newton_iter; j++)
            {
                if (save_result)
                {
                    saveSurfaceMesh(exp_folder + "surface_mesh_iter"+std::to_string(j)+".data", intrinsic_simulation);
                    saveSurfacePoints(exp_folder + "sites_iter"+std::to_string(j)+".data", intrinsic_simulation);
                    saveNetwork(exp_folder + "curves_iter"+std::to_string(j)+".data", intrinsic_simulation);
                }
                bool converged = intrinsic_simulation.advanceOneStep(j);
                if (converged)
                    break;
            }
        }
        else
        {
            GeodesicSimApp app(intrinsic_simulation);    
            app.initializeScene();
            app.run();
        }
    }
    else if (exp == "shell_tightening")
    {
        if (server)
        {
            IntrinsicSimulation intrinsic_simulation;
            intrinsic_simulation.initializeShellTwowayCouplingScene();
            int j = 0;
            for (; j < intrinsic_simulation.max_newton_iter; j++)
            {
                bool converged = intrinsic_simulation.advanceOneStep(j);
                if (converged)
                    break;
            }
        }
        else
        {
            IntrinsicSimulation intrinsic_simulation;
            intrinsic_simulation.initializeShellTwowayCouplingScene();
            
            GeodesicSimApp app(intrinsic_simulation);    
            app.initializeScene();
            app.run();
        }
    }
    else if (exp == "elastic_network")
    {
        if (server)
        {
            std::vector<std::string> filename_list = { 
                "torus", 
                "donut_duck"
            };
            for (int i = 0; i < 2; i++)
            {
                IntrinsicSimulation intrinsic_simulation;
                intrinsic_simulation.use_Newton = true;
                intrinsic_simulation.use_lbfgs = false;
                intrinsic_simulation.max_newton_iter = 100;
                intrinsic_simulation.initializeElasticNetworkScene(filename_list[i], i);
                int j = 0;
                for (; j < intrinsic_simulation.max_newton_iter; j++)
                {
                    if (save_result)
                    {
                        saveSurfacePoints(base_folder + "/results/spring_network/"+filename_list[i]+"_iter_"+std::to_string(j)+"_sites.data", intrinsic_simulation);
                        saveSurfaceMesh(base_folder + "/results/spring_network/"+filename_list[i]+"_iter_"+std::to_string(j)+"_surface_mesh.data", intrinsic_simulation);
                        saveNetwork(base_folder + "/results/spring_network/"+filename_list[i]+"_iter_"+std::to_string(j)+"_curves.data", intrinsic_simulation);
                    }
                    bool converged = intrinsic_simulation.advanceOneStep(j);
                    if (converged)
                        break;
                }
            }
        }
        else
        {
            IntrinsicSimulation intrinsic_simulation;
            intrinsic_simulation.initializeElasticNetworkScene("torus", 0);

            GeodesicSimApp app(intrinsic_simulation);    
            app.initializeScene();
            app.run();
        }
    }
    else if (exp == "move_geodesic")
    {
        std::vector<std::string> mesh_files = { 
            "planar", 
            "spherical", 
            "hyperbolic",
            "bumpy-cube"
        };
        if (server)
        {   
            for (int i = 0; i < mesh_files.size(); i++)
            {
                if (i != 2)
                    continue;
                IntrinsicSimulation intrinsic_simulation;
                intrinsic_simulation.initializePlottingScene(base_folder + "/data/"+ mesh_files[i] + ".obj", i);
                VectorXT dedl(intrinsic_simulation.undeformed.rows()); dedl.setZero();
                intrinsic_simulation.computeResidual(dedl); dedl *= -1.0;
                dedl.normalize();
                if (i == 3)
                    dedl *= -1.0;
                T dx = 0.01;
                
                int n_steps = 50;
                if (i > 2)
                {
                    n_steps = 200;
                    dx = 0.05;
                }
                std::ofstream out(base_folder + "/results/spring_network/"+"energy" + mesh_files[i]+".txt");
                for (int j = 0; j < n_steps; j++)
                {
                    intrinsic_simulation.surface_points = intrinsic_simulation.surface_points_undeformed;
                    if (i > 2)
                        intrinsic_simulation.delta_u.segment<2>(2) = T(j) * dx * Vector<T, 2>(-dedl[3], dedl[2]);
                    else
                        intrinsic_simulation.delta_u.segment<2>(2) = T(j-n_steps/2) * dx * Vector<T, 2>(-dedl[3], dedl[2]);
                    intrinsic_simulation.updateCurrentState(true);
                    if (save_result)
                    {
                        saveSurfacePoints(base_folder + "/results/spring_network/"+mesh_files[i]+"_iter_"+std::to_string(j)+"_sites.data", intrinsic_simulation);
                        saveSurfaceMesh(base_folder + "/results/spring_network/"+mesh_files[i]+"_iter_"+std::to_string(j)+"_surface_mesh.data", intrinsic_simulation);
                        saveNetwork(base_folder + "/results/spring_network/"+mesh_files[i]+"_iter_"+std::to_string(j)+"_curves.data", intrinsic_simulation);
                    }
                    T ej = intrinsic_simulation.computeTotalEnergy() / intrinsic_simulation.we;
                    out << "iter " << j << " obj " << std::sqrt(ej) << std::endl;
                }
                out.close();

            }
            
        }
        else
        {
            IntrinsicSimulation intrinsic_simulation;
            intrinsic_simulation.initializePlottingScene(base_folder + "/data/"+ mesh_files[2] + ".obj", 2);
            VectorXT dedl(intrinsic_simulation.undeformed.rows()); dedl.setZero();
            intrinsic_simulation.computeResidual(dedl); dedl *= -1.0;
            dedl.normalize();
            T dx = 0.01;
            int n_steps = 50;
            // for (int i = 0; i < n_steps; i++)
            int i = 25;
            {
                intrinsic_simulation.surface_points = intrinsic_simulation.surface_points_undeformed;
                intrinsic_simulation.delta_u.segment<2>(2) = T(i-n_steps/2) * dx * Vector<T, 2>(-dedl[3], dedl[2]);
                intrinsic_simulation.updateCurrentState(true);
            }
            intrinsic_simulation.ref_dis *= 0.1;
            intrinsic_simulation.use_t_wrapper = false;
            GeodesicSimApp app(intrinsic_simulation);    
            app.initializeScene();
            app.run();
        }
    }
    else if (exp == "lbfgs_comparison")
    {
        
        if (server)
        {
            std::vector<std::string> mesh_files = { 
                "torus_dense", 
                "3holes_simplified" 
            };
            std::vector<T> resolution = {2.0, 1.0, 2.0, 2.0};
            for (int i = 0; i < mesh_files.size(); i++)
            {
                IntrinsicSimulation intrinsic_simulation;
                intrinsic_simulation.initializeMassSpringSceneExactGeodesic(mesh_files[i], resolution[i]);
            
                intrinsic_simulation.use_Newton = true;
                intrinsic_simulation.use_lbfgs = false;
                intrinsic_simulation.max_newton_iter = 1000;
                int j = 0;
                for (; j < intrinsic_simulation.max_newton_iter; j++)
                {
                    if (save_result)
                    {
                        saveSurfacePoints(base_folder + "/results/convergence/"+mesh_files[i]+"_timing_iter_"+std::to_string(j)+"_sites.data", intrinsic_simulation);
                        saveSurfaceMesh(base_folder + "/results/convergence/"+mesh_files[i]+"_timing_iter_"+std::to_string(j)+"_surface_mesh.data", intrinsic_simulation);
                        saveNetwork(base_folder + "/results/convergence/"+mesh_files[i]+"_timing_iter_"+std::to_string(j)+"_curves.data", intrinsic_simulation);
                    }
                    bool converged = intrinsic_simulation.advanceOneStep(j);
                    if (converged)
                        break;
                }
                std::ofstream out(base_folder + "/results/convergence/"+mesh_files[i]+"_Newton.txt");
                for (int i = 0; i < intrinsic_simulation.residual_norms.size(); i++)
                {
                    out << "iter " << i << " |g| " << intrinsic_simulation.residual_norms[i] << std::endl;
                }
                out.close();

                intrinsic_simulation.use_Newton = false;
                intrinsic_simulation.use_lbfgs = true;
                intrinsic_simulation.max_newton_iter = 1000;
                intrinsic_simulation.reset();
                j = 0;
                for (; j < intrinsic_simulation.max_newton_iter; j++)
                {
                    if (save_result)
                    {
                        saveSurfacePoints(base_folder + "/results/convergence/"+mesh_files[i]+"_timing_iter_"+std::to_string(j)+"_sites_LBFGS.data", intrinsic_simulation);
                        saveSurfaceMesh(base_folder + "/results/convergence/"+mesh_files[i]+"_timing_iter_"+std::to_string(j)+"_surface_mesh_LBFGS.data", intrinsic_simulation);
                        saveNetwork(base_folder + "/results/convergence/"+mesh_files[i]+"_timing_iter_"+std::to_string(j)+"_curves_LBFGS.data", intrinsic_simulation);
                    }
                    bool converged = intrinsic_simulation.advanceOneStep(j);
                    if (converged)
                        break;
                }
                out.open(base_folder + "/results/convergence/"+mesh_files[i]+"_lbfgs.txt");
                for (int i = 0; i < intrinsic_simulation.residual_norms.size(); i++)
                {
                    out << "iter " << i << " |g| " << intrinsic_simulation.residual_norms[i] << std::endl;
                }
                out.close();

                intrinsic_simulation.use_Newton = false;
                intrinsic_simulation.use_lbfgs = false;
                intrinsic_simulation.max_newton_iter = 1000;
                intrinsic_simulation.reset();
                j = 0;
                for (; j < intrinsic_simulation.max_newton_iter; j++)
                {
                    bool converged = intrinsic_simulation.advanceOneStep(j);
                    if (converged)
                        break;
                }
                out.open(base_folder + "/results/convergence/"+mesh_files[i]+"_gd.txt");
                for (int i = 0; i < intrinsic_simulation.residual_norms.size(); i++)
                {
                    out << "iter " << i << " |g| " << intrinsic_simulation.residual_norms[i] << std::endl;
                }
                out.close();
            }
            

        }
        else
        {
            IntrinsicSimulation intrinsic_simulation;
            intrinsic_simulation.initializeMassSpringSceneExactGeodesic("torus_dense", 2.0, true);
            GeodesicSimApp app(intrinsic_simulation);    
            app.initializeScene();
            app.run();
        }
    }
    else if (exp == "lbfgs_comparison_timing")
    {
        if (server)
        {
            std::vector<std::string> mesh_files = { 
                "torus_dense", 
                "3holes_simplified" 
            };
            std::vector<T> resolution = {2.0, 1.0, 2.0, 2.0};
            for (int i = 0; i < mesh_files.size(); i++)
            {
                
                IntrinsicSimulation intrinsic_simulation;
                intrinsic_simulation.initializeMassSpringSceneExactGeodesic(mesh_files[i], resolution[i], true);
            
                intrinsic_simulation.use_Newton = true;
                intrinsic_simulation.use_lbfgs = false;
                intrinsic_simulation.max_newton_iter = 1000;
                int j = 0;
                std::vector<T> runtime;
                timer.restart();
                for (; j < intrinsic_simulation.max_newton_iter; j++)
                {
                    if (save_result)
                    {
                        saveSurfacePoints(base_folder + "/results/convergence/"+mesh_files[i]+"_timing_iter_"+std::to_string(j)+"_sites.data", intrinsic_simulation);
                        saveSurfaceMesh(base_folder + "/results/convergence/"+mesh_files[i]+"_timing_iter_"+std::to_string(j)+"_surface_mesh.data", intrinsic_simulation);
                        saveNetwork(base_folder + "/results/convergence/"+mesh_files[i]+"_timing_iter_"+std::to_string(j)+"_curves.data", intrinsic_simulation);
                    }
                    bool converged = intrinsic_simulation.advanceOneStep(j);
                    runtime.push_back(timer.elapsed_sec());
                    if (converged)
                        break;
                }
                std::ofstream out(base_folder + "/results/convergence/"+mesh_files[i]+"_Newton_timing.txt");
                for (int i = 0; i < intrinsic_simulation.residual_norms.size(); i++)
                {
                    out << "iter " << i << " |g| " << intrinsic_simulation.residual_norms[i] 
                        << " timing " << runtime[i] << "s" 
                        << " |du| " << intrinsic_simulation.maximum_step_sizes[i]
                        << " bb_diag " << intrinsic_simulation.scene_bb_diag << std::endl;
                }
                out.close();
                intrinsic_simulation.use_Newton = false;
                intrinsic_simulation.use_lbfgs = true;
                intrinsic_simulation.max_newton_iter = 1000;
                intrinsic_simulation.reset();
                j = 0;
                timer.restart();
                runtime.clear();
                for (; j < intrinsic_simulation.max_newton_iter; j++)
                {
                    if (save_result)
                    {
                        saveSurfacePoints(base_folder + "/results/convergence/"+mesh_files[i]+"_timing_iter_"+std::to_string(j)+"_sites_LBFGS.data", intrinsic_simulation);
                        saveSurfaceMesh(base_folder + "/results/convergence/"+mesh_files[i]+"_timing_iter_"+std::to_string(j)+"_surface_mesh_LBFGS.data", intrinsic_simulation);
                        saveNetwork(base_folder + "/results/convergence/"+mesh_files[i]+"_timing_iter_"+std::to_string(j)+"_curves_LBFGS.data", intrinsic_simulation);
                    }
                    bool converged = intrinsic_simulation.advanceOneStep(j);
                    runtime.push_back(timer.elapsed_sec());
                    if (converged)
                        break;
                }
                out.open(base_folder + "/results/convergence/"+mesh_files[i]+"_lbfgs_timing.txt");
                for (int i = 0; i < intrinsic_simulation.residual_norms.size(); i++)
                {
                    out << "iter " << i << " |g| " << intrinsic_simulation.residual_norms[i] 
                        << " timing " << runtime[i] << "s" 
                        << " |du| " << intrinsic_simulation.maximum_step_sizes[i]
                        << " bb_diag " << intrinsic_simulation.scene_bb_diag << std::endl;
                }
                out.close();

                intrinsic_simulation.use_Newton = false;
                intrinsic_simulation.use_lbfgs = false;
                intrinsic_simulation.max_newton_iter = 1000;
                intrinsic_simulation.reset();
                j = 0;
                timer.restart();
                runtime.clear();
                for (; j < intrinsic_simulation.max_newton_iter; j++)
                {
                    bool converged = intrinsic_simulation.advanceOneStep(j);
                    runtime.push_back(timer.elapsed_sec());
                    if (converged)
                        break;
                }
                out.open(base_folder + "/results/convergence/"+mesh_files[i]+"_gd_timing.txt");
                for (int i = 0; i < intrinsic_simulation.residual_norms.size(); i++)
                {
                    out << "iter " << i << " |g| " << intrinsic_simulation.residual_norms[i] 
                        << " timing " << runtime[i] << "s" 
                        << " |du| " << intrinsic_simulation.maximum_step_sizes[i]
                        << " bb_diag " << intrinsic_simulation.scene_bb_diag << std::endl;
                }
                out.close();
                timer.stop();
            }
            

        }
        else
        {
            IntrinsicSimulation intrinsic_simulation;
            intrinsic_simulation.initializeMassSpringSceneExactGeodesic("torus_dense", 2.0, true);
            GeodesicSimApp app(intrinsic_simulation);    
            app.initializeScene();
            app.run();
        }
    }
    else if (exp == "ablation_Euclidean")
    {
        if (server)
        {
            std::vector<std::string> test_cases = { 
                "left", 
                "right", 
                "right_angle"
            };
            bool use_geodesic = true;
            std::string config = use_geodesic ? "_geo" : "";
            for (int i = 0; i < test_cases.size(); i++)
            {
                IntrinsicSimulation intrinsic_simulation;
                intrinsic_simulation.initializeSceneEuclideanDistance(2);
                intrinsic_simulation.Euclidean = !use_geodesic;
                intrinsic_simulation.use_Newton = true;
                intrinsic_simulation.use_lbfgs = false;
                intrinsic_simulation.max_newton_iter = 100;
                int j = 0;
                for (; j < intrinsic_simulation.max_newton_iter; j++)
                {
                    saveSurfacePoints(base_folder + "/results/ablation/"+test_cases[i]+config+"_iter_"+std::to_string(j)+"_sites.data", intrinsic_simulation);
                    saveSurfaceMesh(base_folder + "/results/ablation/"+test_cases[i]+config+"_iter_"+std::to_string(j)+"_surface_mesh.data", intrinsic_simulation);
                    saveNetwork(base_folder + "/results/ablation/"+test_cases[i]+config+"_iter_"+std::to_string(j)+"_curves.data", intrinsic_simulation);
                    bool converged = intrinsic_simulation.advanceOneStep(j);
                    if (converged)
                        break;
                }
            }
            
        }
        else
        {
            IntrinsicSimulation intrinsic_simulation;
            intrinsic_simulation.initializeSceneEuclideanDistance(2);
            
            GeodesicSimApp app(intrinsic_simulation);    
            app.initializeScene();
            app.run();
        }
    }
    else if (exp == "comparison_Mancinell_Puppo")
    {
        std::vector<std::vector<int>> vtx_lists(100, std::vector<int>());
        std::vector<T> timings(100);
        std::ifstream in(base_folder + "/data/data_RCM.txt");
        for (int i = 0; i < 100; i++)
        {
            std::string Exp, vtx, face, succeed;
            int exp_id; 
            in >> Exp >> exp_id >> face;
            
            for (int j = 0; j < 5; j++)
            {
                int idx;
                in >> idx;
            }
            in >> vtx;
            for (int j = 0; j < 5; j++)
            {
                int idx;
                in >> idx;
                vtx_lists[i].push_back(idx);
            }
            in >> timings[i];
            T x, y, z;
            in >> x >> y >> z;
            int face_idx; T u, v;
            in >> face_idx >> u >> v;
            in >> succeed;
            bool succeeded;
            in >> succeeded;
            // std::cout << x << " " << y << " " << z << std::endl;
        }
        in.close();
        if (server)
        {
            std::ofstream out(base_folder + "/results/KarcherMeanMancinellPuppo/data_ours.txt");
            for (int i = 0; i < 100; i++)
            {
                out << "Exp " << i << std::endl;
                IntrinsicSimulation intrinsic_simulation;
                intrinsic_simulation.use_lbfgs = false;
                intrinsic_simulation.use_Newton = true;
                
                intrinsic_simulation.max_newton_iter = 200;

                std::vector<int> vtx_list = vtx_lists[i];
                vtx_list.push_back(vtx_list[0]);
                intrinsic_simulation.initializeKarcherMeanSceneMancinellPuppo(vtx_list);
                
                int j = 0;
                
                SimpleTimer timer(true);
                for (; j < intrinsic_simulation.max_newton_iter; j++)
                {
                    // if (save_result)
                        // intrinsic_simulation.saveStates(base_folder + "results/KarcherMeanMancinellPuppo/"+mesh+"/iter_"+std::to_string(j)+suffix+".txt");       
                    
                    bool converged = intrinsic_simulation.advanceOneStep(j);
                    
                    if (converged)
                        break;
                }
                out << std::setprecision(16) << timer.elapsed_sec() << " ";
                auto solution = intrinsic_simulation.surface_points.back();
                TV pos = intrinsic_simulation.toTV(solution.interpolate(intrinsic_simulation.geometry->vertexPositions));
                out << pos[0] << " " << pos[1] << " " << pos[2] << " "
                << solution.face.getIndex() << " " << solution.faceCoords[0] << " " << solution.faceCoords[1] << std::endl;
                if (!save_result)
                    continue;
            }
            out.close();
        }
        else
        {
            IntrinsicSimulation intrinsic_simulation;
            intrinsic_simulation.use_lbfgs = false;
            intrinsic_simulation.use_Newton = true;
            
            intrinsic_simulation.max_newton_iter = 200;

            std::vector<int> vtx_list = vtx_lists[0];
            vtx_list.push_back(vtx_list[0]);
            
            intrinsic_simulation.initializeKarcherMeanSceneMancinellPuppo(vtx_list);

            GeodesicSimApp app(intrinsic_simulation);    
            app.initializeScene();
            app.run();
        }
    }
    else if (exp == "comparison_statistics_Mancinell_Puppo")
    {
        std::vector<std::vector<int>> vtx_lists(100, std::vector<int>());
        std::vector<std::pair<int, Vector<T, 2>>> surface_point_lists(100);
        std::vector<T> timings(100);
        std::vector<bool> valid(100, false);
        std::ifstream in(base_folder + "/results/KarcherMeanMancinellPuppo/data_RCM.txt");
        for (int i = 0; i < 100; i++)
        {
            std::string Exp, vtx, face, succeed;
            int exp_id; 
            in >> Exp >> exp_id >> face;
            
            for (int j = 0; j < 5; j++)
            {
                int idx;
                in >> idx;
            }
            in >> vtx;
            for (int j = 0; j < 5; j++)
            {
                int idx;
                in >> idx;
                vtx_lists[i].push_back(idx);
            }
            in >> timings[i];
            T x, y, z;
            in >> x >> y >> z;
            int face_idx; T u, v;
            in >> face_idx >> u >> v;
            surface_point_lists[i] = std::make_pair(face_idx, Vector<T, 2>(u, v));
            in >> succeed;
            bool succeeded;
            in >> succeeded;
            valid[i] = succeeded;
            // std::cout << x << " " << y << " " << z << std::endl;
        }
        in.close();
        int valid_cnt = 0;
        T timing_sum = 0.0;
        T res_norm_sum = 0.0;
        for (int i = 0; i < 100; i++)
        {
            // out << "Exp " << i << std::endl;
            if (!valid[i])
                continue;
            timing_sum += timings[i];
            IntrinsicSimulation intrinsic_simulation;
            intrinsic_simulation.use_lbfgs = false;
            intrinsic_simulation.use_Newton = true;
            
            intrinsic_simulation.max_newton_iter = 200;

            std::vector<int> vtx_list = vtx_lists[i];
            vtx_list.push_back(vtx_list[0]);
            intrinsic_simulation.initializeKarcherMeanSceneMancinellPuppo(vtx_list);

            intrinsic_simulation.surface_points.back() = 
                gcs::SurfacePoint(intrinsic_simulation.mesh->face(surface_point_lists[i].first),
                gc::Vector3{surface_point_lists[i].second[0]
                ,surface_point_lists[i].second[1],
                1.0 - surface_point_lists[i].second[0] - surface_point_lists[i].second[1]});
            intrinsic_simulation.surface_points.back() = intrinsic_simulation.surface_points.back().inSomeFace();
            intrinsic_simulation.retrace = true;
            intrinsic_simulation.traceGeodesics();
            VectorXT residual(12); residual.setZero();
            intrinsic_simulation.computeResidual(residual);
            res_norm_sum += residual.norm();
            valid_cnt++;
        }
        std::cout << timing_sum / T(valid_cnt) << " |g| " << res_norm_sum / T(valid_cnt) << std::endl;

        in.open(base_folder + "/results/KarcherMeanMancinellPuppo/data_ours.txt");
        for (int i = 0; i < 100; i++)
        {
            std::string Exp;
            int exp_id; 
            in >> Exp >> exp_id;
            in >> timings[i];
            T x, y, z;
            in >> x >> y >> z;
            int face_idx; T u, v;
            in >> face_idx >> u >> v;
            surface_point_lists[i] = std::make_pair(face_idx, Vector<T, 2>(u, v));
        }
        in.close();
        
        timing_sum = 0.0;
        res_norm_sum = 0.0;
        for (int i = 0; i < 100; i++)
        {
            timing_sum += timings[i];
            IntrinsicSimulation intrinsic_simulation;
            intrinsic_simulation.use_lbfgs = false;
            intrinsic_simulation.use_Newton = true;
            
            intrinsic_simulation.max_newton_iter = 200;

            std::vector<int> vtx_list = vtx_lists[i];
            vtx_list.push_back(vtx_list[0]);
            intrinsic_simulation.initializeKarcherMeanSceneMancinellPuppo(vtx_list);

            intrinsic_simulation.surface_points.back() = 
                gcs::SurfacePoint(intrinsic_simulation.mesh->face(surface_point_lists[i].first),
                gc::Vector3{surface_point_lists[i].second[0]
                ,surface_point_lists[i].second[1],
                1.0 - surface_point_lists[i].second[0] - surface_point_lists[i].second[1]});
            intrinsic_simulation.surface_points.back() = intrinsic_simulation.surface_points.back().inSomeFace();
            intrinsic_simulation.retrace = true;
            intrinsic_simulation.traceGeodesics();
            VectorXT residual(12); residual.setZero();
            intrinsic_simulation.computeResidual(residual);
            res_norm_sum += residual.norm();
            
        }
        std::cout << timing_sum / T(100) << " |g| " << res_norm_sum / T(100) << std::endl;
    }
   

}