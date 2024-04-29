#include "../include/VoronoiCells.h"

void VoronoiCells::computeGeodesicLengthGradientEdgePoint(const gcs::Halfedge& he, 
        const SurfacePoint& vA, const SurfacePoint& vB, Vector<T, 3>& dldw)
{
    dldw.setZero();
    T l;
    std::vector<SurfacePoint> path;
    std::vector<IxnData> ixn_data;
    computeGeodesicDistance(vA, vB, l, path, ixn_data, true);

    int length = path.size();
    
    TV dldx0, dldx1;
    TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
    TV v1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));

    TV v10 = toTV(geometry->vertexPositions[he.tailVertex()]);
    TV v11 = toTV(geometry->vertexPositions[he.tipVertex()]);

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
        // std::cout << dldx0.transpose() << " " << dldx1.transpose() << std::endl;
        // std::getchar();
    }

    Vector<T, 6> dldx; dldx.setZero();
    dldx.segment<3>(0) = dldx0;
    dldx.segment<3>(3) = dldx1;

    Matrix<T, 6, 3> dxdw; dxdw.setZero();
    dxdw.block(0, 0, 3, 1) = v11 - v10;

    dxdw.block(3, 1, 3, 1) = v20 - v22;
    dxdw.block(3, 2, 3, 1) = v21 - v22;

    dldw = dldx.transpose() * dxdw;
}

T VoronoiCells::computeGeodesicLengthAndGradientEdgePoint(const SurfacePoint& vA,
        const SurfacePoint& vB, Vector<T, 3>& dldw)
{
    dldw.setZero();
    T l;
    std::vector<SurfacePoint> path;
    std::vector<IxnData> ixn_data;
    computeGeodesicDistance(vA, vB, l, path, ixn_data, true);
    
    int length = path.size();
    
    TV dldx0, dldx1;
    TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
    TV v1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));

    int zero_idx = -1;
    TV bary = toTV(vA.faceCoords);
    std::vector<gcs::Halfedge> half_edges = {vA.face.halfedge(), 
                                            vA.face.halfedge().next(), 
                                            vA.face.halfedge().next().next()};
    for (int d = 0; d < 3; d++)
    {
        if (std::abs(bary[d]) < 1e-8)
            zero_idx = d;
    }
    if (zero_idx == -1)
        std::cout << toTV(vA.faceCoords).transpose() << std::endl;
    gcs::Halfedge he = half_edges[(zero_idx + 1) % 3];
    computeGeodesicLengthGradientEdgePoint(he, vA, vB, dldw);

    return l;
}

T VoronoiCells::computeGeodesicLengthAndGradient(const SurfacePoint& vA,
        const SurfacePoint& vB, Vector<T, 5>& dldw)
{
    dldw.setZero();
    T l;
    std::vector<SurfacePoint> path;
    std::vector<IxnData> ixn_data;
    computeGeodesicDistance(vA, vB, l, path, ixn_data, true);
    
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


    Matrix<T, 3, 2> dx1dw;
    dx1dw.col(0) = v20 - v22;
    dx1dw.col(1) = v21 - v22;

    dldw.segment<3>(0) = dldx0;
    dldw.segment<2>(3) = dldx1.transpose() * dx1dw;

    return l;
}

T VoronoiCells::computeGeodesicLengthAndGradient(const SurfacePoint& vA,
        const SurfacePoint& vB, Vector<T, 4>& dldw)
{
    dldw.setZero();
    T l;
    std::vector<SurfacePoint> path;
    std::vector<IxnData> ixn_data;
    computeGeodesicDistance(vA, vB, l, path, ixn_data, true);
    
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
        // std::cout << dldx0.transpose() << " " << dldx1.transpose() << std::endl;
        // std::getchar();
    }


    Vector<T, 6> dldx; dldx.setZero();
    dldx.segment<3>(0) = dldx0;
    dldx.segment<3>(3) = dldx1;

    Matrix<T, 6, 4> dxdw; dxdw.setZero();
    dxdw.block(0, 0, 3, 1) = v10 - v12;
    dxdw.block(0, 1, 3, 1) = v11 - v12;

    dxdw.block(3, 2, 3, 1) = v20 - v22;
    dxdw.block(3, 3, 3, 1) = v21 - v22;

    dldw = dldx.transpose() * dxdw;

    return l;
}

T VoronoiCells::computeGeodesicLengthAndGradientAndHessian(const SurfacePoint& vA,
        const SurfacePoint& vB, Vector<T, 5>& dldw, Matrix<T, 5, 5>& d2ldxdw)
{
    dldw.setZero();
    d2ldxdw.setZero();
    T l;
    std::vector<SurfacePoint> path;
    std::vector<IxnData> ixn_data;
    computeGeodesicDistance(vA, vB, l, path, ixn_data, true);
    
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
    Matrix<T, 3, 3> dx0dw0;
    dx0dw0.setIdentity();

    Matrix<T, 3, 2> dx1dw1;
    dx1dw1.col(0) = v20 - v22;
    dx1dw1.col(1) = v21 - v22;

    TV dldw0 = dldx0.transpose() * dx0dw0;
    TV2 dldw1 = dldx1.transpose() * dx1dw1;

    Matrix<T, 6, 5> dxdw; dxdw.setZero();
    dxdw.block(0, 0, 3, 3).setIdentity();

    dxdw.block(3, 2, 3, 1) = v20 - v22;
    dxdw.block(3, 3, 3, 1) = v21 - v22;
    
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

        // if ((ixn0-v0).norm() < 1e-5)
        //     std::cout << "small edge : |x0 - c0| " << (ixn0-v0).norm() << " vi " << eij[0] << " vj " << eij[1] << std::endl;
        // if ((v1-ixn1).norm() < 1e-5)
        //     std::cout << "small edge : |xn - c1| " << (v1-ixn1).norm() << " vi " << eij[0] << " vj " << eij[1] << std::endl;

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
            TV x_start = ixn_data[1+ixn_id].start;
            TV x_end = ixn_data[1+ixn_id].end;
            dxdt.block(ixn_id * 3, ixn_id, 3, 1) = (x_end - x_start);
        }
        
        TM dlndxn = (TM::Identity() - dldx1 * dldx1.transpose()) / (ixn1 - v1).norm();
        dfdx.block(ixn_dof-3, ixn_dof-3, 3, 3) += dlndxn;
        dfdc.block(ixn_dof-3, 3, 3, 3) += -dlndxn;
        d2gdcdx.block(ixn_dof-3, 3, 3, 3) += -dlndxn;

        MatrixXT dxdtd2gdxdc = dxdt.transpose() * d2gdcdx;
        MatrixXT dxdtd2gdx2dxdt = dxdt.transpose() * dfdx * dxdt;
        MatrixXT dtdc = dxdtd2gdx2dxdt.colPivHouseholderQr().solve(-dxdtd2gdxdc);


        Matrix<T, 6, 6> d2gdc2; d2gdc2.setZero();
        d2gdc2.block(0, 0, 3, 3) += dl0dx0;
        d2gdc2.block(3, 3, 3, 3) += dlndxn;

        d2ldx2 = d2gdc2 + d2gdcdx.transpose() * dxdt * dtdc;

        Matrix<T, 6, 6> d2ldx2_approx; d2ldx2_approx.setZero();
        d2ldx2_approx.block(0, 0, 3, 3) = (TM::Identity() - dldx0 * dldx0.transpose()) / l;
        d2ldx2_approx.block(3, 3, 3, 3) = (TM::Identity() - dldx1 * dldx1.transpose()) / l;
        d2ldx2_approx.block(0, 3, 3, 3) = -0.5 * (d2ldx2_approx.block(0, 0, 3, 3) + d2ldx2_approx.block(3, 3, 3, 3));
        d2ldx2_approx.block(3, 0, 3, 3) = -0.5 * (d2ldx2_approx.block(0, 0, 3, 3) + d2ldx2_approx.block(3, 3, 3, 3));


        if ((dfdx.maxCoeff() > 1e6 || dfdx.minCoeff() < -1e6) )
            d2ldx2 = d2ldx2_approx;
    }
    dldw = dldx.transpose() * dxdw;
    d2ldxdw = dxdw.transpose() * d2ldx2 * dxdw;

    return l;
}

T VoronoiCells::computeGeodesicLengthAndGradientAndHessian(const SurfacePoint& vA,
        const SurfacePoint& vB, Vector<T, 4>& dldw, Matrix<T, 4, 4>& d2ldw2)
{
    dldw.setZero();
    d2ldw2.setZero();
    T l;
    std::vector<SurfacePoint> path;
    std::vector<IxnData> ixn_data;
    computeGeodesicDistance(vA, vB, l, path, ixn_data, true);
    
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
    Matrix<T, 3, 2> dx0dw0;
    dx0dw0.col(0) = v10 - v12;
    dx0dw0.col(1) = v11 - v12;

    Matrix<T, 3, 2> dx1dw1;
    dx1dw1.col(0) = v20 - v22;
    dx1dw1.col(1) = v21 - v22;

    TV2 dldw0 = dldx0.transpose() * dx0dw0;
    TV2 dldw1 = dldx1.transpose() * dx1dw1;

    Matrix<T, 6, 4> dxdw; dxdw.setZero();
    dxdw.block(0, 0, 3, 1) = v10 - v12;
    dxdw.block(0, 1, 3, 1) = v11 - v12;

    dxdw.block(3, 2, 3, 1) = v20 - v22;
    dxdw.block(3, 3, 3, 1) = v21 - v22;
    
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

        // if ((ixn0-v0).norm() < 1e-5)
        //     std::cout << "small edge : |x0 - c0| " << (ixn0-v0).norm() << " vi " << eij[0] << " vj " << eij[1] << std::endl;
        // if ((v1-ixn1).norm() < 1e-5)
        //     std::cout << "small edge : |xn - c1| " << (v1-ixn1).norm() << " vi " << eij[0] << " vj " << eij[1] << std::endl;

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
            TV x_start = ixn_data[1+ixn_id].start;
            TV x_end = ixn_data[1+ixn_id].end;
            dxdt.block(ixn_id * 3, ixn_id, 3, 1) = (x_end - x_start);
        }
        
        TM dlndxn = (TM::Identity() - dldx1 * dldx1.transpose()) / (ixn1 - v1).norm();
        dfdx.block(ixn_dof-3, ixn_dof-3, 3, 3) += dlndxn;
        dfdc.block(ixn_dof-3, 3, 3, 3) += -dlndxn;
        d2gdcdx.block(ixn_dof-3, 3, 3, 3) += -dlndxn;

        MatrixXT dxdtd2gdxdc = dxdt.transpose() * d2gdcdx;
        MatrixXT dxdtd2gdx2dxdt = dxdt.transpose() * dfdx * dxdt;
        MatrixXT dtdc = dxdtd2gdx2dxdt.colPivHouseholderQr().solve(-dxdtd2gdxdc);


        Matrix<T, 6, 6> d2gdc2; d2gdc2.setZero();
        d2gdc2.block(0, 0, 3, 3) += dl0dx0;
        d2gdc2.block(3, 3, 3, 3) += dlndxn;

        d2ldx2 = d2gdc2 + d2gdcdx.transpose() * dxdt * dtdc;

        Matrix<T, 6, 6> d2ldx2_approx; d2ldx2_approx.setZero();
        d2ldx2_approx.block(0, 0, 3, 3) = (TM::Identity() - dldx0 * dldx0.transpose()) / l;
        d2ldx2_approx.block(3, 3, 3, 3) = (TM::Identity() - dldx1 * dldx1.transpose()) / l;
        d2ldx2_approx.block(0, 3, 3, 3) = -0.5 * (d2ldx2_approx.block(0, 0, 3, 3) + d2ldx2_approx.block(3, 3, 3, 3));
        d2ldx2_approx.block(3, 0, 3, 3) = -0.5 * (d2ldx2_approx.block(0, 0, 3, 3) + d2ldx2_approx.block(3, 3, 3, 3));


        if ((dfdx.maxCoeff() > 1e6 || dfdx.minCoeff() < -1e6) )
            d2ldx2 = d2ldx2_approx;
    }
    dldw = dldx.transpose() * dxdw;
    d2ldw2 = dxdw.transpose() * d2ldx2 * dxdw;

    return l;
}

void VoronoiCells::computeGeodesicLengthGradient(const SurfacePoint& vA,
     const SurfacePoint& vB, Vector<T, 4>& dldw)
{
    dldw.setZero();
    T l;
    std::vector<SurfacePoint> path;
    std::vector<IxnData> ixn_data;
    computeGeodesicDistance(vA, vB, l, path, ixn_data, true);
    
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
        // std::cout << dldx0.transpose() << " " << dldx1.transpose() << std::endl;
        // std::getchar();
    }


    Vector<T, 6> dldx; dldx.setZero();
    dldx.segment<3>(0) = dldx0;
    dldx.segment<3>(3) = dldx1;

    Matrix<T, 6, 4> dxdw; dxdw.setZero();
    dxdw.block(0, 0, 3, 1) = v10 - v12;
    dxdw.block(0, 1, 3, 1) = v11 - v12;

    dxdw.block(3, 2, 3, 1) = v20 - v22;
    dxdw.block(3, 3, 3, 1) = v21 - v22;

    dldw = dldx.transpose() * dxdw;
}


bool VoronoiCells::computeDxDs(const SurfacePoint& x, 
    const std::vector<int>& site_indices, MatrixXT& dx_ds, bool reduced)
{
    TV pos = toTV(x.interpolate(geometry->vertexPositions));
    TV bary = toTV(x.faceCoords);
    int zero_idx = -1;
    if (site_indices.size() == 2)
    {
        std::vector<gcs::Halfedge> half_edges = {x.face.halfedge(), x.face.halfedge().next(), x.face.halfedge().next().next()};
        for (int d = 0; d < 3; d++)
        {
            if (std::abs(bary[d]) < 1e-8)
                zero_idx = d;
        }
        if (zero_idx == -1)
        {
            // std::cout << "ERROR having to sites but not on an edge " << __FILE__ << std::endl;
            // std::cout << bary.transpose() << " " << std::endl;
            dx_ds.resize(3, 4); dx_ds.setZero();
            if (reduced)
                dx_ds.resize(2, 4); dx_ds.setZero();
            use_debug_face_color = true;
            int n_tri = mesh->nFaces();
            face_color.resize(n_tri, 3);
            face_color.setOnes();

            face_color.row(x.face.getIndex()) = Eigen::RowVector3d(0.0,0.0,1.0);

            return false;
        }
        else
        {
            gcs::Halfedge he = half_edges[(zero_idx + 1) % 3];
            TV vtx0 = toTV(geometry->vertexPositions[he.tailVertex()]);
            TV vtx1 = toTV(geometry->vertexPositions[he.tipVertex()]);
            T t = (pos - vtx0).norm() / (vtx1 - vtx0).norm();
            
            SurfacePoint si = samples[site_indices[0]];
            SurfacePoint sj = samples[site_indices[1]];
            Vector<T, 3> dDdwxa, dDdwxb;
            
            std::vector<SurfacePoint> sij = {si, sj};
            std::vector<Vector<T, 3>> dDdwxsab(2);
            
            #ifdef PARALLEL
            tbb::parallel_for(0, 2, [&](int thread_idx){
                computeGeodesicLengthGradientEdgePoint(he, x, sij[thread_idx], dDdwxsab[thread_idx]);
            });
            #else
            for (int thread_idx = 0; thread_idx < 2; thread_idx++)
                computeGeodesicLengthGradientEdgePoint(he, x, sij[thread_idx], dDdwxsab[thread_idx]);
            #endif
            dDdwxa = dDdwxsab[0];
            dDdwxb = dDdwxsab[1];
            // computeGeodesicLengthGradientEdgePoint(he, x, si, dDdwxa);
            // computeGeodesicLengthGradientEdgePoint(he, x, sj, dDdwxb);
            // std::cout << dDdwxa.transpose() << " " << dDdwxb.transpose() << std::endl;
            Vector<T, 4> dDds;
            dDds.segment<2>(0) = dDdwxa.segment<2>(1);
            dDds.segment<2>(2) = -dDdwxb.segment<2>(1);

            T dDdx_tilde = dDdwxa[0] - dDdwxb[0];
            if (dDdx_tilde < 1e-8)
                return false;
            Vector<T, 4> dx_tilde_ds = -dDds / dDdx_tilde;
            Vector<T, 3> dx_dx_tilde = (vtx1 - vtx0);
            dx_ds = dx_dx_tilde * dx_tilde_ds.transpose();
            if (reduced)
                dx_ds = dx_tilde_ds;
        }
    }
    else if(site_indices.size() > 2)
    {
        SurfacePoint si = samples[site_indices[0]];
        SurfacePoint sj = samples[site_indices[1]];
        SurfacePoint sk = samples[site_indices[2]];
        std::vector<SurfacePoint> sijk = {si, sj, sk};
        Vector<T, 4> dDdwxsi, dDdwxsj, dDdwxsk;
        std::vector<Vector<T, 4>> dDdwxsijk(3);
        #ifdef PARALLEL
        tbb::parallel_for(0, 3, [&](int thread_idx){
            computeGeodesicLengthGradient(x, sijk[thread_idx], dDdwxsijk[thread_idx]);
        });
        #else
        for (int thread_idx = 0; thread_idx < 3; thread_idx++)
            computeGeodesicLengthGradient(x, sijk[thread_idx], dDdwxsijk[thread_idx]);
        #endif
        dDdwxsi = dDdwxsijk[0];
        dDdwxsj = dDdwxsijk[1];
        dDdwxsk = dDdwxsijk[2];
        // computeGeodesicLengthGradient(x, si, dDdwxsi);
        // computeGeodesicLengthGradient(x, sj, dDdwxsj);
        // computeGeodesicLengthGradient(x, sk, dDdwxsk);

        Matrix<T, 2, 6> dgds; dgds.setZero();
        TM2 dgdx_tilde(2, 2); dgdx_tilde.setZero();

        dgdx_tilde.row(0) += dDdwxsi.segment<2>(0) - dDdwxsj.segment<2>(0);
        dgdx_tilde.row(1) += dDdwxsi.segment<2>(0) - dDdwxsk.segment<2>(0);

        dgds.block(0, 0, 1, 2) += dDdwxsi.segment<2>(2).transpose();
        dgds.block(0, 2, 1, 2) -= dDdwxsj.segment<2>(2).transpose();

        dgds.block(1, 0, 1, 2) += dDdwxsi.segment<2>(2).transpose();
        dgds.block(1, 4, 1, 2) -= dDdwxsk.segment<2>(2).transpose();

        Matrix<T, 2, 6> dx_tilde_ds = dgdx_tilde.colPivHouseholderQr().solve(-dgds);
        T error = (dgdx_tilde * dx_tilde_ds + dgds).norm();
        if (error > 1e-6)
        {
            dx_ds.resize(3, site_indices.size() * 2); dx_ds.setZero();
            if (reduced)
                dx_ds.resize(2, site_indices.size() * 2); dx_ds.setZero();
            return false;
        }
            

        Matrix<T, 3, 2> dx_dx_tilde;
        TV v10 = toTV(geometry->vertexPositions[x.face.halfedge().vertex()]);
        TV v11 = toTV(geometry->vertexPositions[x.face.halfedge().next().vertex()]);
        TV v12 = toTV(geometry->vertexPositions[x.face.halfedge().next().next().vertex()]);

        dx_dx_tilde.col(0) = v10 - v12;
        dx_dx_tilde.col(1) = v11 - v12;

        dx_ds.resize(3, site_indices.size() * 2); dx_ds.setZero();
        dx_ds.block(0, 0, 3, 6) = dx_dx_tilde * dx_tilde_ds;
        
        if (reduced)
        {
            dx_ds.resize(2, site_indices.size() * 2); dx_ds.setZero();
            dx_ds.block(0, 0, 2, 6) = dx_tilde_ds;
        }
        // std::cout << dx_ds << std::endl;
        // std::getchar();
    }
    else
    {
        std::cout << "This ixn point is only linked to " << site_indices.size() 
         << " site(s)" << std::endl;
        return false;
    }
    return true;
}