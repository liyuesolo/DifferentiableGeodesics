#include "../include/App.h"

void GeodesicSimApp::initializeScene() {
  // Initialize polyscope
  polyscope::options::autocenterStructures = false;
  polyscope::options::autoscaleStructures = false;
  polyscope::view::windowWidth = 3000;
  polyscope::view::windowHeight = 2000;
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;
  polyscope::options::groundPlaneHeightFactor = 0.6;
  polyscope::options::shadowDarkness = 0.4;
  polyscope::init();
  int n_vtx_dof = simulation.extrinsic_vertices.rows();
  if (simulation.two_way_coupling) {
    if (simulation.use_FEM) {
      meshV.resize(simulation.num_nodes, 3);
      for (int i = 0; i < simulation.num_nodes; i++) {
        meshV.row(i) = simulation.deformed.segment<3>(
            simulation.fem_dof_start +
            simulation.surface_to_tet_node_map[i] * 3);
      }
    } else
      vectorToIGLMatrix<T, 3>(
          simulation.deformed.segment(simulation.deformed.rows() - n_vtx_dof,
                                      n_vtx_dof),
          meshV);
  } else
    vectorToIGLMatrix<T, 3>(simulation.extrinsic_vertices, meshV);
  vectorToIGLMatrix<int, 3>(simulation.extrinsic_indices, meshF);
  surface_mesh = polyscope::registerSurfaceMesh("surface mesh", meshV, meshF);
  surface_mesh->setSmoothShade(false);
  surface_mesh->setSurfaceColor(SURFACE_COLOR);
  surface_mesh->setEdgeWidth(1.0);

  simulation.updateVisualization();

  int n_edges = simulation.all_intrinsic_edges.size();
  std::vector<glm::vec3> nodes(n_edges * 2);
  std::vector<std::array<size_t, 2>> edges(n_edges);
  for (size_t i = 0; i < n_edges; i++) {
    nodes[i * 2] = glm::vec3(simulation.all_intrinsic_edges[i].first[0],
                             simulation.all_intrinsic_edges[i].first[1],
                             simulation.all_intrinsic_edges[i].first[2]);
    nodes[i * 2 + 1] = glm::vec3(simulation.all_intrinsic_edges[i].second[0],
                                 simulation.all_intrinsic_edges[i].second[1],
                                 simulation.all_intrinsic_edges[i].second[2]);
    edges[i] = {i * 2, i * 2 + 1};
  }
  geodesic_curves =
      polyscope::registerCurveNetwork("geodesic_curves", nodes, edges);
  geodesic_curves->setColor(CURVE_COLOR);
  geodesic_curves->setRadius(0.004, true);

  polyscope::state::userCallback = [&]() { sceneCallback(); };
}

void GeodesicSimApp::updateSimulationData() {
  int n_vtx_dof = simulation.extrinsic_vertices.rows();
  if (simulation.two_way_coupling) {
    if (simulation.use_FEM) {
      meshV.resize(simulation.num_nodes, 3);
      for (int i = 0; i < simulation.num_nodes; i++) {
        meshV.row(i) = simulation.deformed.segment<3>(
            simulation.fem_dof_start +
            simulation.surface_to_tet_node_map[i] * 3);
      }
    } else
      vectorToIGLMatrix<T, 3>(
          simulation.deformed.segment(simulation.deformed.rows() - n_vtx_dof,
                                      n_vtx_dof),
          meshV);
  } else
    vectorToIGLMatrix<T, 3>(simulation.extrinsic_vertices, meshV);

  surface_mesh->updateVertexPositions(meshV);
  simulation.updateVisualization();

  int n_edges = simulation.all_intrinsic_edges.size();
  std::vector<glm::vec3> nodes(n_edges * 2);
  std::vector<std::array<size_t, 2>> edges(n_edges);
  for (size_t i = 0; i < n_edges; i++) {
    nodes[i * 2] = glm::vec3(simulation.all_intrinsic_edges[i].first[0],
                             simulation.all_intrinsic_edges[i].first[1],
                             simulation.all_intrinsic_edges[i].first[2]);
    nodes[i * 2 + 1] = glm::vec3(simulation.all_intrinsic_edges[i].second[0],
                                 simulation.all_intrinsic_edges[i].second[1],
                                 simulation.all_intrinsic_edges[i].second[2]);
    edges[i] = {i * 2, i * 2 + 1};
  }
  geodesic_curves =
      polyscope::registerCurveNetwork("geodesic_curves", nodes, edges);
  geodesic_curves->setColor(CURVE_COLOR);
  geodesic_curves->setRadius(0.004, true);
}

void GeodesicSimApp::sceneCallback() {
  ImGui::SetWindowFontScale(2.f);
  if (ImGui::Button("Run Simulation")) {
    run_sim = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("Stop")) {
    run_sim = false;
  }
  ImGui::SameLine();
  if (ImGui::Button("Reset")) {
    simulation.reset();
    static_solve_step = 0;
    updateSimulationData();
  }
  ImGui::Spacing();
  if (ImGui::Button("Advance One Step")) {
    simulation.advanceOneStep(static_solve_step++);
    updateSimulationData();
  }
  // if (ImGui::Button("Check Derivative"))
  // {
  //     simulation.checkTotalGradient(true);
  //     simulation.checkTotalHessian(true);
  // }
  // if (ImGui::Button("Check Derivative Scale"))
  // {
  //     simulation.checkTotalGradientScale(true);
  //     simulation.checkTotalHessianScale(true);
  // }
  if (run_sim) {
    bool finished = simulation.advanceOneStep(static_solve_step++);
    updateSimulationData();
    if (finished)
      run_sim = false;
  }
}

void GeodesicSimApp::saveMesh(const std::string &folder, int iter) {
  if (save_sites) {
    VectorXT sites;
    simulation.getAllPointsPosition(sites);
    FILE *stream;
    if ((stream =
             fopen((folder + "/sites" + std::to_string(save_counter) + ".data")
                       .c_str(),
                   "wb")) != NULL) {
      int len = sites.rows();
      size_t temp;
      temp = fwrite(&len, sizeof(int), 1, stream);
      temp = fwrite(sites.data(), sizeof(double), len, stream);
    } else {
      std::cout << "Unable to write into file" << std::endl;
    }
    fclose(stream);
  }
  if (save_mesh) {
    FILE *stream;
    if ((stream = fopen(
             (folder + "/surface_mesh" + std::to_string(save_counter) + ".data")
                 .c_str(),
             "wb")) != NULL) {
      int len = simulation.extrinsic_vertices.rows();
      size_t temp;
      temp = fwrite(&len, sizeof(int), 1, stream);
      if (simulation.two_way_coupling)
        temp = fwrite(
            simulation.deformed.segment(simulation.deformed.rows() - len, len)
                .data(),
            sizeof(double), len, stream);
      else
        temp = fwrite(simulation.extrinsic_vertices.data(), sizeof(double), len,
                      stream);
      len = simulation.extrinsic_indices.rows();
      temp = fwrite(&len, sizeof(int), 1, stream);
      temp =
          fwrite(simulation.extrinsic_indices.data(), sizeof(int), len, stream);
    } else {
      std::cout << "Unable to write into file" << std::endl;
    }
    fclose(stream);
  }
  if (save_curve) {
    VectorXT all_data(simulation.all_intrinsic_edges.size() * 6);
    for (int i = 0; i < simulation.all_intrinsic_edges.size(); i++) {
      all_data.segment<3>(i * 6 + 0) = simulation.all_intrinsic_edges[i].first;
      all_data.segment<3>(i * 6 + 3) = simulation.all_intrinsic_edges[i].second;
    }

    FILE *stream;
    if ((stream =
             fopen((folder + "/curves" + std::to_string(save_counter) + ".data")
                       .c_str(),
                   "wb")) != NULL) {
      int len = all_data.rows();
      size_t temp;
      temp = fwrite(&len, sizeof(int), 1, stream);
      temp = fwrite(all_data.data(), sizeof(double), len, stream);
    } else {
      std::cout << "Unable to write into file" << std::endl;
    }
    fclose(stream);
  }

  save_counter++;
}

void VoronoiApp::initializeScene() {
  polyscope::options::autocenterStructures = false;
  polyscope::options::autoscaleStructures = false;
  polyscope::view::windowWidth = 3000;
  polyscope::view::windowHeight = 2000;
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;
  polyscope::options::groundPlaneHeightFactor = 0.6;
  polyscope::options::shadowDarkness = 0.4;

  // Initialize polyscope
  polyscope::init();

  polyscope::state::userCallback = [&]() { sceneCallback(); };
}

void VoronoiApp::initializeSimulationData() {
  int n_vtx_dof = simulation.extrinsic_vertices.rows();
  if (simulation.add_coplanar) {
    bool use_centroid = false;
    simulation.iterateVoronoiCells([&](const auto &cell_data, int cell_idx) {
      int n_cell_node = cell_data.cell_vtx_nodes.size();
      int n_vtx_current = meshV.rows();

      MatrixXT points(n_cell_node, 3);
      int cnt = 0;

      for (int idx : cell_data.cell_vtx_nodes) {
        auto ixn = simulation.unique_ixn_points[cell_data.cell_nodes[idx]];
        // std::cout << cell_data.cell_nodes[idx] << std::endl;
        auto xi = ixn.first;
        points.row(cnt++) = simulation.toTV(
            xi.interpolate(simulation.geometry->vertexPositions));
      }

      MatrixXT A(n_cell_node, 3);
      VectorXT rhs(n_cell_node);
      A.col(2).setConstant(1.0);
      for (int j = 0; j < n_cell_node; j++) {
        for (int i = 0; i < 2; i++)
          A(j, i) = points(j, i);
        rhs[j] = points(j, 2);
      }
      TM ATA = A.transpose() * A;

      TV coeff = ATA.inverse() * (A.transpose() * rhs);

      T a = coeff[0], b = coeff[1], c = coeff[2];

      T denom = std::sqrt(a * a + b * b);

      T avg = 0.0;
      for (int i = 0; i < n_cell_node; i++) {
        T nom =
            std::abs(a * points(i, 0) + b * points(i, 1) + c - points(i, 2));
        avg += nom / denom;
      }
      avg / T(n_cell_node);

      TV face_centroid = TV::Zero();
      for (int i = 0; i < points.rows(); i++) {
        face_centroid += points.row(i);
      }
      face_centroid /= T(points.rows());

      int n_face_current = meshF.rows();
      if (use_centroid) {
        meshV.conservativeResize(n_vtx_current + n_cell_node + 1, 3);
        meshV.block(n_vtx_current, 0, n_cell_node, 3) = points;
        meshV.block(n_vtx_current + n_cell_node, 0, 1, 3) =
            face_centroid.transpose();
        meshF.conservativeResize(n_face_current + n_cell_node, 3);
        for (int j = 0; j < n_cell_node; j++)
          meshF.row(n_face_current + j) =
              IV(j, n_cell_node, (j + 1) % n_cell_node) +
              IV::Constant(n_vtx_current);
      } else {
        meshV.conservativeResize(n_vtx_current + n_cell_node, 3);
        meshV.block(n_vtx_current, 0, n_cell_node, 3) = points;
        if (n_cell_node == 3) {
          meshF.conservativeResize(n_face_current + 1, 3);
          meshF.row(n_face_current) = IV(0, 1, 2) + IV::Constant(n_vtx_current);
        } else if (n_cell_node == 4) {
          meshF.conservativeResize(n_face_current + 2, 3);
          meshF.row(n_face_current) = IV(0, 3, 1) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 1) =
              IV(1, 3, 2) + IV::Constant(n_vtx_current);
        } else if (n_cell_node == 5) {
          meshF.conservativeResize(n_face_current + 3, 3);
          meshF.row(n_face_current) = IV(0, 4, 1) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 1) =
              IV(1, 4, 3) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 2) =
              IV(1, 3, 2) + IV::Constant(n_vtx_current);
        } else if (n_cell_node == 6) {
          meshF.conservativeResize(n_face_current + 4, 3);
          meshF.row(n_face_current) = IV(0, 5, 1) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 1) =
              IV(1, 5, 4) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 2) =
              IV(1, 4, 2) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 3) =
              IV(2, 4, 3) + IV::Constant(n_vtx_current);
        } else if (n_cell_node == 7) {
          meshF.conservativeResize(n_face_current + 5, 3);
          meshF.row(n_face_current) = IV(3, 2, 4) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 1) =
              IV(2, 1, 4) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 2) =
              IV(1, 0, 4) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 3) =
              IV(0, 6, 4) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 4) =
              IV(4, 6, 5) + IV::Constant(n_vtx_current);
        } else if (n_cell_node == 8) {
          meshF.conservativeResize(n_face_current + 6, 3);
          meshF.row(n_face_current) = IV(0, 7, 1) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 1) =
              IV(1, 7, 6) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 2) =
              IV(1, 6, 2) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 3) =
              IV(2, 6, 5) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 4) =
              IV(2, 5, 3) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 5) =
              IV(3, 5, 4) + IV::Constant(n_vtx_current);
        } else if (n_cell_node == 9) {
          meshF.conservativeResize(n_face_current + 7, 3);
          meshF.row(n_face_current) = IV(1, 0, 2) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 1) =
              IV(2, 0, 3) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 2) =
              IV(3, 0, 8) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 3) =
              IV(3, 8, 7) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 4) =
              IV(3, 7, 4) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 5) =
              IV(4, 7, 6) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 6) =
              IV(4, 6, 5) + IV::Constant(n_vtx_current);
        } else {
          meshV.conservativeResize(n_vtx_current + n_cell_node + 1, 3);
          meshV.block(n_vtx_current, 0, n_cell_node, 3) = points;
          meshV.block(n_vtx_current + n_cell_node, 0, 1, 3) =
              face_centroid.transpose();
          meshF.conservativeResize(n_face_current + n_cell_node, 3);
          for (int j = 0; j < n_cell_node; j++)
            meshF.row(n_face_current + j) =
                IV(j, n_cell_node, (j + 1) % n_cell_node) +
                IV::Constant(n_vtx_current);
        }
      }
    });

  } else {
    vectorToIGLMatrix<T, 3>(simulation.extrinsic_vertices, meshV);
    vectorToIGLMatrix<int, 3>(simulation.extrinsic_indices, meshF);
  }
  surface_mesh = polyscope::registerSurfaceMesh("surface mesh", meshV, meshF);
  surface_mesh->setSmoothShade(false);
  surface_mesh->setSurfaceColor(SURFACE_COLOR);
  surface_mesh->setEdgeWidth(1.0);
  if (simulation.add_coplanar)
    surface_mesh->setMaterial("normal");

  int n_edges = simulation.voronoi_edges.size();
  std::vector<glm::vec3> nodes(n_edges * 2);
  std::vector<std::array<size_t, 2>> edges(n_edges);
  for (size_t i = 0; i < n_edges; i++) {
    nodes[i * 2] = glm::vec3(simulation.voronoi_edges[i].first[0],
                             simulation.voronoi_edges[i].first[1],
                             simulation.voronoi_edges[i].first[2]);
    nodes[i * 2 + 1] = glm::vec3(simulation.voronoi_edges[i].second[0],
                                 simulation.voronoi_edges[i].second[1],
                                 simulation.voronoi_edges[i].second[2]);
    edges[i] = {i * 2, i * 2 + 1};
  }
  geodesic_curves =
      polyscope::registerCurveNetwork("geodesic_curves", nodes, edges);
  geodesic_curves->setColor(CURVE_COLOR);
  geodesic_curves->setRadius(0.004, true);
  if (simulation.add_coplanar)
    geodesic_curves->setEnabled(false);
}

void VoronoiApp::updateSimulationData() {
  int n_vtx_dof = simulation.extrinsic_vertices.rows();

  if (simulation.add_coplanar) {
    bool use_centroid = false;
    meshV.resize(0, 0);
    meshF.resize(0, 0);
    simulation.iterateVoronoiCells([&](const auto &cell_data, int cell_idx) {
      int n_cell_node = cell_data.cell_vtx_nodes.size();
      int n_vtx_current = meshV.rows();

      MatrixXT points(n_cell_node, 3);
      int cnt = 0;

      for (int idx : cell_data.cell_vtx_nodes) {
        auto ixn = simulation.unique_ixn_points[cell_data.cell_nodes[idx]];
        // std::cout << cell_data.cell_nodes[idx] << std::endl;
        auto xi = ixn.first;
        points.row(cnt++) = simulation.toTV(
            xi.interpolate(simulation.geometry->vertexPositions));
      }

      MatrixXT A(n_cell_node, 3);
      VectorXT rhs(n_cell_node);
      A.col(2).setConstant(1.0);
      for (int j = 0; j < n_cell_node; j++) {
        for (int i = 0; i < 2; i++)
          A(j, i) = points(j, i);
        rhs[j] = points(j, 2);
      }
      TM ATA = A.transpose() * A;

      TV coeff = ATA.inverse() * (A.transpose() * rhs);

      T a = coeff[0], b = coeff[1], c = coeff[2];

      T denom = std::sqrt(a * a + b * b);

      T avg = 0.0;
      for (int i = 0; i < n_cell_node; i++) {
        T nom =
            std::abs(a * points(i, 0) + b * points(i, 1) + c - points(i, 2));
        avg += nom / denom;
      }
      avg / T(n_cell_node);

      TV face_centroid = TV::Zero();
      for (int i = 0; i < points.rows(); i++) {
        face_centroid += points.row(i);
      }
      face_centroid /= T(points.rows());

      int n_face_current = meshF.rows();
      if (use_centroid) {
        meshV.conservativeResize(n_vtx_current + n_cell_node + 1, 3);
        meshV.block(n_vtx_current, 0, n_cell_node, 3) = points;
        meshV.block(n_vtx_current + n_cell_node, 0, 1, 3) =
            face_centroid.transpose();
        meshF.conservativeResize(n_face_current + n_cell_node, 3);
        for (int j = 0; j < n_cell_node; j++)
          meshF.row(n_face_current + j) =
              IV(j, n_cell_node, (j + 1) % n_cell_node) +
              IV::Constant(n_vtx_current);
      } else {
        meshV.conservativeResize(n_vtx_current + n_cell_node, 3);
        meshV.block(n_vtx_current, 0, n_cell_node, 3) = points;
        if (n_cell_node == 3) {
          meshF.conservativeResize(n_face_current + 1, 3);
          meshF.row(n_face_current) = IV(0, 1, 2) + IV::Constant(n_vtx_current);
        } else if (n_cell_node == 4) {
          meshF.conservativeResize(n_face_current + 2, 3);
          meshF.row(n_face_current) = IV(0, 3, 1) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 1) =
              IV(1, 3, 2) + IV::Constant(n_vtx_current);
        } else if (n_cell_node == 5) {
          meshF.conservativeResize(n_face_current + 3, 3);
          meshF.row(n_face_current) = IV(0, 4, 1) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 1) =
              IV(1, 4, 3) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 2) =
              IV(1, 3, 2) + IV::Constant(n_vtx_current);
        } else if (n_cell_node == 6) {
          meshF.conservativeResize(n_face_current + 4, 3);
          meshF.row(n_face_current) = IV(0, 5, 1) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 1) =
              IV(1, 5, 4) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 2) =
              IV(1, 4, 2) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 3) =
              IV(2, 4, 3) + IV::Constant(n_vtx_current);
        } else if (n_cell_node == 7) {
          meshF.conservativeResize(n_face_current + 5, 3);
          meshF.row(n_face_current) = IV(3, 2, 4) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 1) =
              IV(2, 1, 4) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 2) =
              IV(1, 0, 4) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 3) =
              IV(0, 6, 4) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 4) =
              IV(4, 6, 5) + IV::Constant(n_vtx_current);
        } else if (n_cell_node == 8) {
          meshF.conservativeResize(n_face_current + 6, 3);
          meshF.row(n_face_current) = IV(0, 7, 1) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 1) =
              IV(1, 7, 6) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 2) =
              IV(1, 6, 2) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 3) =
              IV(2, 6, 5) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 4) =
              IV(2, 5, 3) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 5) =
              IV(3, 5, 4) + IV::Constant(n_vtx_current);
        } else if (n_cell_node == 9) {
          meshF.conservativeResize(n_face_current + 7, 3);
          meshF.row(n_face_current) = IV(1, 0, 2) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 1) =
              IV(2, 0, 3) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 2) =
              IV(3, 0, 8) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 3) =
              IV(3, 8, 7) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 4) =
              IV(3, 7, 4) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 5) =
              IV(4, 7, 6) + IV::Constant(n_vtx_current);
          meshF.row(n_face_current + 6) =
              IV(4, 6, 5) + IV::Constant(n_vtx_current);
        } else {
          meshV.conservativeResize(n_vtx_current + n_cell_node + 1, 3);
          meshV.block(n_vtx_current, 0, n_cell_node, 3) = points;
          meshV.block(n_vtx_current + n_cell_node, 0, 1, 3) =
              face_centroid.transpose();
          meshF.conservativeResize(n_face_current + n_cell_node, 3);
          for (int j = 0; j < n_cell_node; j++)
            meshF.row(n_face_current + j) =
                IV(j, n_cell_node, (j + 1) % n_cell_node) +
                IV::Constant(n_vtx_current);
        }
      }
    });
    surface_mesh = polyscope::registerSurfaceMesh("surface mesh", meshV, meshF);
    surface_mesh->setSmoothShade(false);
    surface_mesh->setSurfaceColor(SURFACE_COLOR);
    surface_mesh->setEdgeWidth(1.0);
    surface_mesh->setMaterial("normal");
  } else {
    vectorToIGLMatrix<T, 3>(simulation.extrinsic_vertices, meshV);
  }

  int n_edges = simulation.voronoi_edges.size();
  std::vector<glm::vec3> nodes(n_edges * 2);
  std::vector<std::array<size_t, 2>> edges(n_edges);
  for (size_t i = 0; i < n_edges; i++) {
    nodes[i * 2] = glm::vec3(simulation.voronoi_edges[i].first[0],
                             simulation.voronoi_edges[i].first[1],
                             simulation.voronoi_edges[i].first[2]);
    nodes[i * 2 + 1] = glm::vec3(simulation.voronoi_edges[i].second[0],
                                 simulation.voronoi_edges[i].second[1],
                                 simulation.voronoi_edges[i].second[2]);
    edges[i] = {i * 2, i * 2 + 1};
  }
  geodesic_curves =
      polyscope::registerCurveNetwork("geodesic_curves", nodes, edges);
  geodesic_curves->setColor(CURVE_COLOR);
  geodesic_curves->setRadius(0.004, true);
  if (simulation.add_coplanar)
    geodesic_curves->setEnabled(false);
}

void VoronoiApp::sceneCallback() {
  ImGui::SetWindowFontScale(2.f);
  if (ImGui::Button("RunSim")) {
    run_sim = true;
  }
  if (ImGui::Button("stop")) {
    run_sim = false;
  }
  if (ImGui::Button("advanceOneStep")) {
    simulation.advanceOneStep(static_solve_step++);
    updateSimulationData();
  }
  if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (ImGui::Checkbox("geodesic", &geodesic)) {
      update_geodesic = true;
    }
    if (ImGui::Checkbox("exact", &exact)) {
      update_exact = true;
    }
    if (ImGui::Checkbox("Perimeter", &perimeter)) {
      if (perimeter)
        simulation.objective = Perimeter;
      simulation.add_peri = perimeter;
      update_perimeter = true;
    }
    if (ImGui::Checkbox("Centroidal", &CGVD)) {
      if (CGVD) {
        simulation.objective = Centroidal;
        exact = true;
      }
      simulation.add_centroid = CGVD;
      update_CGVD = true;
    }
    if (ImGui::Checkbox("Coplanar", &simulation.add_coplanar)) {
      // simulation.add_centroid = true;
      simulation.max_mma_iter = 1000;
      // simulation.w_centroid = 0.5;
    }
    if (ImGui::Checkbox("SameLength", &simulation.add_length)) {

      simulation.max_mma_iter = 100;
    }
    if (ImGui::Checkbox("EdgeWeighting", &simulation.edge_weighting)) {
    }
    if (ImGui::Checkbox("Regularizer", &reg)) {
      simulation.add_reg = reg;
      simulation.w_reg = 1e-6;
    }
    if (ImGui::Checkbox("MMA", &simulation.use_MMA)) {
      simulation.use_lbfgs = !simulation.use_MMA;
      simulation.use_Newton = !simulation.use_MMA;
    }
    if (ImGui::Checkbox("LBFGS", &simulation.use_lbfgs)) {
      simulation.use_MMA = !simulation.use_lbfgs;
      simulation.use_Newton = !simulation.use_lbfgs;
    }
  }
  if (ImGui::CollapsingHeader("Save Data", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (ImGui::Checkbox("save_IDT", &save_idt)) {
    }
  }
  if (ImGui::Button("Generate", ImVec2(-1, 0))) {
    if (geodesic)
      simulation.metric = Geodesic;
    else
      simulation.metric = Euclidean;
    simulation.loadGeometry(
        "../../../Projects/DifferentiableGeodesics/data/sphere.obj");
    simulation.resample(1.0);
    simulation.constructVoronoiDiagram(exact, false);
    initializeSimulationData();
  }
  if (ImGui::Button("Resample", ImVec2(-1, 0))) {
    simulation.resample(1.0);
    simulation.constructVoronoiDiagram(exact, false);
    initializeSimulationData();
  }
  if (ImGui::Button("Reset", ImVec2(-1, 0))) {
    simulation.reset();
    simulation.constructVoronoiDiagram(exact, false);
    updateSimulationData();
  }
  if (ImGui::Button("check derivative")) {
    simulation.diffTest();
  }
  if (ImGui::Button("check derivative scale")) {
    simulation.diffTestScale();
  }
  if (run_sim) {
    bool finished = simulation.advanceOneStep(static_solve_step++);
    updateSimulationData();
    if (finished)
      run_sim = false;
  }
}

void VoronoiApp::saveMesh(const std::string &folder, int iter) {
  if (save_sites) {
    FILE *stream;
    if ((stream = fopen(
             (folder + "/iter_" + std::to_string(iter) + "_sites.data").c_str(),
             "wb")) != NULL) {
      int len = simulation.voronoi_sites.rows();
      size_t temp;
      temp = fwrite(&len, sizeof(int), 1, stream);
      temp =
          fwrite(simulation.voronoi_sites.data(), sizeof(double), len, stream);
      fclose(stream);
    } else {
      std::cout << "Unable to write into file" << std::endl;
    }
  }
  if (save_mesh) {
    FILE *stream;
    if ((stream = fopen(
             (folder + "/iter_" + std::to_string(iter) + "_surface_mesh.data")
                 .c_str(),
             "wb")) != NULL) {
      int len = simulation.extrinsic_vertices.rows();
      size_t temp;
      temp = fwrite(&len, sizeof(int), 1, stream);
      temp = fwrite(simulation.extrinsic_vertices.data(), sizeof(double), len,
                    stream);
      len = simulation.extrinsic_indices.rows();
      temp = fwrite(&len, sizeof(int), 1, stream);
      temp =
          fwrite(simulation.extrinsic_indices.data(), sizeof(int), len, stream);
    } else {
      std::cout << "Unable to write into file" << std::endl;
    }
    fclose(stream);
    if (simulation.add_coplanar) {
      if ((stream = fopen(
               (folder + "/iter_" + std::to_string(iter) + "_planar_mesh.data")
                   .c_str(),
               "wb")) != NULL) {
        MatrixXT _V, _C;
        MatrixXi _F;
        simulation.generateMeshForRendering(_V, _F, _C);
        VectorXT vertices;
        VectorXi faces;
        iglMatrixFatten<T, 3>(_V, vertices);
        iglMatrixFatten<int, 3>(_F, faces);
        int len = vertices.rows();
        size_t temp;
        temp = fwrite(&len, sizeof(int), 1, stream);
        temp = fwrite(vertices.data(), sizeof(double), len, stream);
        len = faces.rows();
        temp = fwrite(&len, sizeof(int), 1, stream);
        temp = fwrite(faces.data(), sizeof(int), len, stream);
      } else {
        std::cout << "Unable to write into file" << std::endl;
      }
      fclose(stream);
    }
  }
  if (save_curve) {
    VectorXT all_data(simulation.voronoi_edges.size() * 6);
    for (int i = 0; i < simulation.voronoi_edges.size(); i++) {
      all_data.segment<3>(i * 6 + 0) = simulation.voronoi_edges[i].first;
      all_data.segment<3>(i * 6 + 3) = simulation.voronoi_edges[i].second;
    }

    FILE *stream;
    if ((stream =
             fopen((folder + "/iter_" + std::to_string(iter) + "_curves.data")
                       .c_str(),
                   "wb")) != NULL) {
      int len = all_data.rows();
      size_t temp;
      temp = fwrite(&len, sizeof(int), 1, stream);
      temp = fwrite(all_data.data(), sizeof(double), len, stream);
    } else {
      std::cout << "Unable to write into file" << std::endl;
    }
    fclose(stream);
  }
}
