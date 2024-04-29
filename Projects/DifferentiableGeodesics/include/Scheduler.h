#ifndef SCHEDULER_H
#define SCHEDULER_H

// #include <igl/opengl/glfw/Viewer.h>
// #include <igl/project.h>
// #include <igl/unproject_on_plane.h>

// #include <igl/stb/write_image.h>
// #include <igl/writeOBJ.h>
// #include <igl/readOBJ.h>
// #include <igl/jet.h>

#include "App.h"
#include "IntrinsicSimulation.h"
#include "VoronoiCells.h"
#include "SimpleTimer.h"

class IntrinsicSimulation;
class VoronoiCells;

class Scheduler
{
public:
    using CMat = Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>;
    using TV = Vector<T, 3>;
    using IV = Vector<int, 3>;
    using TV3 = Vector<T, 3>;
    using TM3 = Matrix<T, 3, 3>;
    using VectorXT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    std::string exp = "";
    std::string input_folder = "";
    std::string output_folder = "";
    std::string exp_config = "";
    std::string base_folder = "";
    SimpleTimer timer;
        
    void execute(bool server, bool save_result = false);

    void saveSurfacePoints(const std::string& filename, IntrinsicSimulation& intrinsic_simulation);
    void saveSurfaceMesh(const std::string& filename, IntrinsicSimulation& intrinsic_simulation);
    void saveNetwork(const std::string& filename, IntrinsicSimulation& intrinsic_simulation);
    void saveVectorData(const std::string& filename, const VectorXT& data);

    void saveSurfacePoints(const std::string& filename, VoronoiCells& voronoi_cells);
    void saveSurfaceMesh(const std::string& filename, VoronoiCells& voronoi_cells);
    void saveCoplanarMeshData(const std::string& filename, const std::string& filename2, VoronoiCells& voronoi_cells);
    void saveNetwork(const std::string& filename, VoronoiCells& voronoi_cells);
public:
    Scheduler() {}
    ~Scheduler() {}
};


#endif