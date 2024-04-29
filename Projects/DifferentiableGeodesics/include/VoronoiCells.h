#ifndef VORONOI_CELLS_H
#define VORONOI_CELLS_H
#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <unordered_map>
#include <complex>
#include <iomanip>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "VecMatDef.h"
#include "Util.h"

#include "geometrycentral/surface/edge_length_geometry.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/mesh_graph_algorithms.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "geometrycentral/surface/polygon_soup_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/poisson_disk_sampler.h"

#include "../../../Solver/MMASolver.h"
#include "../../../Solver/LBFGS.h"

namespace gcs = geometrycentral::surface;
namespace gc = geometrycentral;

enum DistanceMetric
{
    Geodesic, Euclidean
};

enum Objective
{
    Perimeter, Centroidal, Both
};

class VoronoiCells
{
public:
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;
    using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
    using VtxList = std::vector<int>;
    using TV = Vector<T, 3>;
    using TV2 = Vector<T, 2>;
    using IV = Vector<int, 3>;
    using IV2 = Vector<int, 2>;
    using TM = Matrix<T, 3, 3>;
    using TM2 = Matrix<T, 2, 2>;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;
    using Face = Vector<int, 3>;
    using Edge = Vector<int, 2>;

    using gcEdge = geometrycentral::surface::Edge;
    using gcFace = geometrycentral::surface::Face;
    using gcVertex = geometrycentral::surface::Vertex;
    using Vector3 = geometrycentral::Vector3;
    using SurfacePoint = geometrycentral::surface::SurfacePoint;

    struct IxnData
    {
        TV start, end;
        T t;
        int start_vtx_idx, end_vtx_idx;
        IxnData(const TV& _start, const TV& _end, T _t, int idx0, int idx1) 
            : start(_start), end(_end), t(_t), start_vtx_idx(idx0),
            end_vtx_idx(idx1) {};
        IxnData(const TV& _start, const TV& _end, T _t) 
            : start(_start), end(_end), t(_t), start_vtx_idx(-1),
            end_vtx_idx(-1) {};
    };

    // use to construct voronoi diagram
    struct FaceData
    {
        std::vector<int> site_indices;
        std::vector<TV> distances;
        FaceData (const std::vector<int>& _site_indices, 
            const std::vector<TV>& _distances) : 
            site_indices(_site_indices), distances(_distances) {}
        FaceData () {}

    };

    struct VoronoiCellData
    {
        // cell nodes are used to index unique ixn points, i.e. global index
        VtxList cell_nodes; 
        // vtx nodes are used to index cell nodes, i.e. local offsets
        VtxList cell_vtx_nodes; // cell vertices
        VtxList cell_edge_nodes; // intersection points on edges
        VectorXT cell_edge_lengths; // edge lengths
        VoronoiCellData() {}
    };
    

    
    VectorXT extrinsic_vertices;
    VectorXi extrinsic_indices;

    bool use_debug_face_color = false;
    MatrixXT face_color;

    // extrinsic geometry interface for geometry central
    std::unique_ptr<gcs::ManifoldSurfaceMesh> mesh;
    std::unique_ptr<gcs::VertexPositionGeometry> geometry;

    MatrixXT surface_vtx;
    MatrixXi surface_indices;

    int n_sites = 0;
    // sites 3D positions
    VectorXT voronoi_sites;
    
    // sites
    std::vector<SurfacePoint> samples;
    std::vector<SurfacePoint> samples_rest;
    std::vector<FaceData> source_data;

    // unique intersection points between voronoi edges and the hosting mesh
    std::vector<std::pair<SurfacePoint, std::vector<int>>> unique_ixn_points;
    std::vector<VtxList> voronoi_cell_vertices;
    std::vector<Edge> valid_VD_edges;

    std::vector<VtxList> voronoi_cells;
    std::vector<std::pair<TV, TV>> voronoi_edges;
    int n_voronoi_edges = 0;

    DistanceMetric metric = Euclidean;    
    Objective objective = Perimeter;

    std::unordered_map<int, T> dirichlet_data;

    bool verbose = false;
    int max_newton_iter = 3000;
    int max_mma_iter = 300;
    int ls_max = 12;
    T newton_tol = 1e-6;
    
    T w_reg = 1e-6;
    T w_centroid = 1.0;
    T w_peri = 1.0;
    T w_co = 1.0;
    T w_len = 1.0;
    bool add_reg = false;
    bool add_centroid = false;
    bool add_coplanar = false;
    bool add_length = false;
    bool add_peri = false;
    bool run_diff_test = false;
    bool use_MMA = false;
    bool edge_weighting = false;

    MMASolver mma_solver;
    VectorXT cell_weights;
    std::vector<VoronoiCellData> voronoi_cell_data;

    bool use_Newton = true;

    LBFGS<T> lbfgs_solver;
    bool use_lbfgs = false;
    bool lbfgs_first_step = true;
    VectorXT gi;

    // unique voronoi edges
    std::vector<Edge> unique_voronoi_edges;

    // saves the surface points, x, for all the geodesics
    std::vector<std::vector<SurfacePoint>> paths;
    // saves the intersection data, t, v, for all the geodesics
	std::vector<std::vector<IxnData>> ixn_data_list;
    // recompute geodesics
    bool retrace = true;
    VectorXT current_length;
    T target_length = 0.0;

    // mollifier
    bool use_t_wrapper = true;

public:
    template <class OP>
    void iterateDirichletDoF(const OP& f) {
        for (auto dirichlet: dirichlet_data){
            f(dirichlet.first, dirichlet.second);
        } 
    }

    template <class OP>
    void iterateVoronoiCells(const OP& f) 
    {
        for (int i = 0; i < voronoi_cell_data.size(); i++)
        {
            f(voronoi_cell_data[i], i);
        }
        
    }

private:
    // ============================ HELPERS ============================
    template <class OP>
    void iterateVoronoiCellsParallel(const OP& f) 
    {
        tbb::parallel_for(0, (int)voronoi_cell_data.size(), [&](int i)
        {
            f(voronoi_cell_data[i], i);
        });        
    }
    
    
    void updateFaceColor();
    
    void edgeLengthHessian(const TV& v0, const TV& v1, Matrix<T, 6, 6>& hess)
    {
        TV dir = (v1-v0).normalized();
        hess.setZero();
        hess.block(0, 0, 3, 3) = (TM::Identity() - dir * dir.transpose())/(v1 - v0).norm();
        hess.block(3, 3, 3, 3) = hess.block(0, 0, 3, 3);
        hess.block(0, 3, 3, 3) = -hess.block(0, 0, 3, 3);
        hess.block(3, 0, 3, 3) = -hess.block(0, 0, 3, 3);
    }

    template<int dim = 2>
    void addForceEntry(VectorXT& residual, 
        const std::vector<int>& vtx_idx, 
        const VectorXT& gradient, int shift = 0)
    {
        for (int i = 0; i < vtx_idx.size(); i++)
            residual.template segment<dim>(vtx_idx[i] * dim + shift) += gradient.template segment<dim>(i * dim);
    }

    template<int dim_row=2, int dim_col=2>
    void addHessianEntry(
        std::vector<Entry>& triplets,
        const std::vector<int>& vtx_idx, 
        const MatrixXT& hessian, 
        int shift_row = 0, int shift_col=0)
    {
        
        for (int i = 0; i < vtx_idx.size(); i++)
        {
            int dof_i = vtx_idx[i];
            for (int j = 0; j < vtx_idx.size(); j++)
            {
                int dof_j = vtx_idx[j];
                for (int k = 0; k < dim_row; k++)
                    for (int l = 0; l < dim_col; l++)
                        triplets.emplace_back(
                                dof_i * dim_row + k + shift_row, 
                                dof_j * dim_col + l + shift_col, 
                                hessian(i * dim_row + k, j * dim_col + l)
                            );                
            }
        }
    }

    template<int dim_row=2, int dim_col=2>
    void addJacobianEntry(
        std::vector<Entry>& triplets,
        const std::vector<int>& vtx_idx,
        const std::vector<int>& vtx_idx2, 
        const MatrixXT& jacobian, 
        int shift_row = 0, int shift_col=0)
    {
        
        for (int i = 0; i < vtx_idx.size(); i++)
        {
            int dof_i = vtx_idx[i];
            for (int j = 0; j < vtx_idx2.size(); j++)
            {
                int dof_j = vtx_idx2[j];
                for (int k = 0; k < dim_row; k++)
                    for (int l = 0; l < dim_col; l++)
                        triplets.emplace_back(
                                dof_i * dim_row + k + shift_row, 
                                dof_j * dim_col + l + shift_col, 
                                jacobian(i * dim_row + k, j * dim_col + l)
                            );                
            }
        }
    }

    void projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data)
    {
        for (auto iter : data)
        {
            A.row(iter.first) *= 0.0;
            A.col(iter.first) *= 0.0;
            A.coeffRef(iter.first, iter.first) = 1.0;
        }
    }
    
    // ============================ MOLLIFIER ============================
    template<int order>
    T wrapper_t(T t, T eps = 1e-6)
    {
        if constexpr (order == 0)
        {
            if (t < eps && t >= 0)
            {
                return -1.0/std::pow(eps, 2) * std::pow(t, 3) + 2.0 / eps * std::pow(t, 2);
            }
            else if (t >= eps && t <= 1.0 - eps)
            {
                return t;
            }
            else 
            {
                return -1.0/std::pow(eps, 2) * std::pow(t-1, 3) - 2.0 / eps * std::pow(t-1.0, 2) + 1.0;
            }
        }
        else if constexpr (order == 1)
        {
            if (t < eps && t >= 0)
            {
                return -3.0/std::pow(eps, 2) * std::pow(t, 2) + 4.0 / eps * t;
            }
            else if (t >= eps && t <= 1.0 - eps)
            {
                return 1.0;
            }
            else 
            {
                return -3.0/std::pow(eps, 2) * std::pow(t-1, 2) - 4.0 / eps * (t-1);
            }
        }
        else if constexpr (order == 2)
        {
            if (t < eps && t >= 0)
            {
                return -6.0/std::pow(eps, 2) * (t - 1.0) + 4.0 / eps;
            }
            else if (t >= eps && t <= 1.0 - eps)
            {
                return 0.0;
            }
            else 
            {
                return -6.0/std::pow(eps, 2) * (t - 1.0) - 4.0 / eps;
            }
        }
        return 0.0;
    }

    void computeSurfacePointdxds(const SurfacePoint& pt, Matrix<T, 3, 2>& dxdw);

public:
    void traceGeodesics();
    void computeReferenceLength();
    void computeGeodesicLengthGradient(const Edge& edge, int edge_idx, Vector<T, 6>& dldw);
    void computeGeodesicLengthGradientAndHessian(const Edge& edge, int edge_idx,
        Vector<T, 6>& dldw, Matrix<T, 6, 6>& d2ldw2);
    void computeExactGeodesic(const SurfacePoint& va, const SurfacePoint& vb, 
        T& dis, std::vector<SurfacePoint>& path, 
        std::vector<IxnData>& ixn_data, 
        bool trace_path = false);
    TV toTV(const Vector3& vec) const
    {
        return TV(vec.x, vec.y, vec.z);
    }
    
    void loadGeometry(const std::string& filename);
    void sampleSitesPoissonDisk(T resolution = 1.0, bool perturb = false);
    void sampleSitesByFace(const TV2& bary, T resolution = 1.0);
    void resample(int resolution = 1.0);
    void reset();

    // ============================ SIMULATION ============================

    bool advanceOneStep(int step);
    T computeTotalEnergy();
    T computeResidual(VectorXT& residual);
    void buildSystemMatrix(StiffnessMatrix& K);
    T lineSearchNewton(const VectorXT& residual);

    // ============================ OBJECTIVES ============================
    // objectives.cpp

    T addRegEnergy(T w);
    void addRegForceEntries(VectorXT& grad, T w);
    void addRegHessianEntries(std::vector<Entry>& entries, T w);

    // Objectives.cpp
    T computeCentroidalVDEnergy(T w = 1.0);
    T computeCentroidalVDGradient(VectorXT& grad, T& energy, T w = 1.0);
    T computeCentroidalVDHessian(StiffnessMatrix& hess, VectorXT& grad, T& energy, T w = 1.0);
    
    void addCentroidalVDForceEntries(VectorXT& grad, T w = 1.0);
    void addCentroidalVDHessianEntries(std::vector<Entry>& entries, T w);

    void checkCoplanarity(T& mean_val, T& max_val);
    T addCoplanarVDEnergy(T w = 1.0);
    void addCoplanarVDForceEntries(VectorXT& grad, T w = 1.0);
    void addCoplanarVDHessianEntries(std::vector<Entry>& entries, T w = 1.0);

    void find3DVoronoiEdges();
    void checkLengthObjective(T& mean_val, T& max_val);
    T addSameLengthVDEnergy(T w = 1.0);
    void addSameLengthVDForceEntries(VectorXT& grad, T w = 1.0);
    void addSameLengthVDHessianEntries(std::vector<Entry>& entries, T w = 1.0);

    T computePerimeterMinimizationEnergy(T w = 1.0);
    T computePerimeterMinimizationGradient(VectorXT& grad, T& energy, T w = 1.0);
    T computePerimeterMinimizationHessian(StiffnessMatrix& hess, VectorXT& grad, T& energy, T w = 1.0);
    
    void addPerimeterMinimizationForceEntries(VectorXT& grad, T w = 1.0);
    void addPerimeterMinimizationHessianEntries(std::vector<Entry>& entries, T w);


    T computeDistanceMatchingEnergy(const std::vector<int>& site_indices, 
        SurfacePoint& xi_current);
    T computeDistanceMatchingGradient(const std::vector<int>& site_indices, 
        SurfacePoint& xi_current, TV2& grad, T& energy);
    void computeDistanceMatchingHessian(const std::vector<int>& site_indices, 
        SurfacePoint& xi_current, TM2& hess);
    T computeDistanceMatchingEnergyGradientHessian(const std::vector<int>& site_indices, 
        SurfacePoint& xi_current, TM2& hess, TV2& grad, T& energy);

    // ============================ VORONOI DIAGRAM CONSTRUCTION ============================
    // Voronoi.cpp      
    void constructVoronoiCellConnectivity();
    void computeDualIDT(std::vector<std::pair<TV, TV>>& idt_edge_vertices,
        std::vector<IV>& idt_indices);
    bool linearSolve(StiffnessMatrix& K, const VectorXT& residual, VectorXT& du);  
    void updateSurfacePoint(SurfacePoint& xi_current, const TV2& search_direction);

    void optimizeForExactVD();
    // void optimizeForCentroidalVD();
    void diffTestScale();
    void diffTest();
    // void perimeterMinimizationVD();
    
    void intersectPrisms(std::vector<SurfacePoint>& samples,
        const std::vector<FaceData>& source_data, 
        std::vector<std::pair<SurfacePoint, std::vector<int>>>& ixn_data);
    void propagateDistanceField(std::vector<SurfacePoint>& samples,
        std::vector<FaceData>& source_data);
    void constructVoronoiDiagram(bool exact = false, bool load_from_file = false);
    void saveVoronoiDiagram(const std::string& filename);
    void generateMeshForRendering(MatrixXT& V, MatrixXi& F, MatrixXT& C);

    void computeGeodesicDistance(const SurfacePoint& va, const SurfacePoint& vb,
        T& dis, std::vector<SurfacePoint>& path, 
        std::vector<IxnData>& ixn_data, bool trace_path = false);
    bool computeGeodesicDistanceEdgeFlip(const SurfacePoint& va, const SurfacePoint& vb,
        T& dis, std::vector<SurfacePoint>& path, 
        std::vector<IxnData>& ixn_data, bool trace_path = false);
    
    // ============================ DERIVATIVE INTERFACES ============================
    // VoronoiDerivatives.cpp
    T computeGeodesicLengthAndGradient(const SurfacePoint& vA,
        const SurfacePoint& vB, Vector<T, 4>& dldw);
    T computeGeodesicLengthAndGradient(const SurfacePoint& vA,
        const SurfacePoint& vB, Vector<T, 5>& dldw);
    T computeGeodesicLengthAndGradientAndHessian(const SurfacePoint& vA,
        const SurfacePoint& vB, Vector<T, 4>& dldw, Matrix<T, 4, 4>& d2ldw2);
    T computeGeodesicLengthAndGradientAndHessian(const SurfacePoint& vA,
        const SurfacePoint& vB, Vector<T, 5>& dldw, Matrix<T, 5, 5>& d2ldxdw);
    T computeGeodesicLengthAndGradientEdgePoint(const SurfacePoint& vA,
        const SurfacePoint& vB, Vector<T, 3>& dldw);
    void computeGeodesicLengthGradient(const SurfacePoint& vA,
        const SurfacePoint& vB, Vector<T, 4>& dldw);
    void computeGeodesicLengthGradientEdgePoint(const gcs::Halfedge& he, 
        const SurfacePoint& vA, const SurfacePoint& vB, Vector<T, 3>& dldw);
    bool computeDxDs(const SurfacePoint& x, 
        const std::vector<int>& site_indices, 
        MatrixXT& dx_tilde_ds, bool reduced = false);
    void computeDxDsAllNodes(StiffnessMatrix& dxds);

    
    VoronoiCells() {}
    ~VoronoiCells() {}
};


#endif 