if (APPLE)
    set(DEPS boost_filesystem tbb 
        tbbmalloc tbbmalloc_proxy igl::core
        suitesparseconfig metis cholmod amd camd ccolamd colamd
        gmp mpfr solver lbfgs_solver
    )
else(UNIX)
    set(DEPS boost_filesystem tbb 
        tbbmalloc tbbmalloc_proxy igl::core
        mkl_intel_lp64 
        mkl_sequential mkl_core 
        suitesparseconfig metis cholmod amd camd ccolamd colamd
        gmp mpfr solver lbfgs_solver
    )
endif()
    
file(GLOB HEADERS "include/*.h" "autodiff/*.h")
file(GLOB SOURCES "src/*.cpp" "autodiff/*.cpp")


add_executable(DifferentiableGeodesics ${HEADERS} ${SOURCES})

target_include_directories(DifferentiableGeodesics PUBLIC ../../Libs/spectra/include)
target_include_directories(DifferentiableGeodesics PUBLIC ../../Deps/libigl/include)

target_link_libraries(DifferentiableGeodesics ${DEPS} geometry-central polyscope)
