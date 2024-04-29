#ifndef APP_H
#define APP_H

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"


#include "VoronoiCells.h"
#include "IntrinsicSimulation.h"

class SimulationApp
{
public:
    using TV = Vector<double, 3>;
    using TM = Matrix<T, 3, 3>;
    using TV3 = Vector<double, 3>;
    using VectorXT = Matrix<double, Eigen::Dynamic, 1>;
    using VectorXi = Matrix<int, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
    using IV = Vector<int, 3>;
    using IV3 = Vector<int, 3>;
public:
    Eigen::MatrixXd meshV;
    Eigen::MatrixXi meshF;

    int static_solve_step = 0;
    bool run_sim = false;

    bool save_sites = false;
    bool save_mesh = false;
    bool save_curve = false;

    polyscope::SurfaceMesh* surface_mesh;
    polyscope::CurveNetwork* geodesic_curves;
    polyscope::PointCloud* endpoints;

    glm::vec3 SURFACE_COLOR = glm::vec3(0.255, 0.514, 0.996);
    glm::vec3 CURVE_COLOR = glm::vec3(1., 0.854, 0.);

    
    void run()
    {
        polyscope::show();
    }
public:
    SimulationApp() {}
    ~SimulationApp () {}
};

class GeodesicSimApp : public SimulationApp
{
    
public:
    
    IntrinsicSimulation& simulation;
    bool all_edges = false;
    int selected = -1;
    T x0, y0;
    bool step_along_search_direction = false;
    VectorXT search_direction;

    VectorXT temporary_vector;
    bool use_temp_vec = false;
    

    
    bool save_all = false;
    bool save_network = false;

    int save_counter = 0;

public:
    void updateSimulationData();

    void initializeScene();
    
    void sceneCallback();

    void saveMesh(const std::string& folder, int iter = 0);

    GeodesicSimApp(IntrinsicSimulation& _simulation) : simulation(_simulation) {}
    ~GeodesicSimApp() {}
};

class VoronoiApp : public SimulationApp
{
public:
    VoronoiCells& simulation;

public:
    int selected = -1;
    T x0, y0;
    
    bool geodesic = true;
    bool exact = false;
    bool CGVD = false;
    bool reg = false;
    bool perimeter = false;

    bool triangulate = false;

    bool update_geodesic = false;
    bool update_exact = false;
    bool update_CGVD = false;
    bool update_perimeter = false;

    bool compute_dual = false;
    bool save_idt = false;
    bool save_site_states = false;

    T reference_length = 1.0;

    void saveMesh(const std::string& folder, int iter = 0);
    
    void updateSimulationData();

    void initializeSimulationData();

    void initializeScene();
    
    void sceneCallback();

    VoronoiApp(VoronoiCells& cells) : simulation(cells) {}
    ~VoronoiApp() {}
};

#endif