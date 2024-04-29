#include "../include/Util.h"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Advancing_front_surface_reconstruction.h>
#include <CGAL/tuple.h>
#include <boost/lexical_cast.hpp>

//CGAL 
typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_3  Point_3;
typedef std::array<std::size_t,3> Facet;

void triangulatePointCloud(const Eigen::VectorXd& points, Eigen::VectorXi& triangle_indices)
{
    int n_pt = points.rows() / 3;

    std::vector<Point_3> pointsCGAL;
    std::vector<Facet> facets;

    for (int i = 0; i < n_pt; i++)
        pointsCGAL.push_back(Point_3(points[i * 3 + 0],
        points[i * 3 + 1],
        points[i * 3 + 2]));
    
    
    CGAL::advancing_front_surface_reconstruction(pointsCGAL.begin(),
                                                pointsCGAL.end(),
                                                std::back_inserter(facets));
    
    triangle_indices.resize(facets.size() * 3);
    for (int i = 0; i < facets.size(); i++)
        triangle_indices.segment<3>(i * 3) = Eigen::Vector3i(facets[i][2], facets[i][1], facets[i][0]);
}

T computeTriangleArea(const Eigen::Vector3d& v0, 
    const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, bool signed_area)
{
    Eigen::Vector3d e0 = v1 - v0;
    Eigen::Vector3d e1 = v2 - v0;
    if (signed_area)
        return 0.5 * e0.cross(e1)[2];
    return 0.5 * e0.cross(e1).norm();
}

Eigen::Vector3d computeBarycentricCoordinates(const Eigen::Vector3d& point, 
    const Eigen::Vector3d& v0, const Eigen::Vector3d& v1, const Eigen::Vector3d& v2)
{
    T area = computeTriangleArea(v0, v1, v2);
    T sub0 = computeTriangleArea(point, v1, v2);
    T sub1 = computeTriangleArea(v0, point, v2);
    T sub2 = computeTriangleArea(v0, v1, point);
    // std::cout << area << " " << sub0 << " " << sub1 << " " << sub2 << " " << std::endl;
    T w0 = sub0 < 1e-8 ? 0 : sub0 / area;
    T w1 = sub1 < 1e-8 ? 0 : sub1 / area;
    T w2 = sub2 < 1e-8 ? 0 : sub2 / area;
    return Eigen::Vector3d(w0, w1, w2);
}