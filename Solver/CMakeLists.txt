add_library(solver STATIC MMASolver.h MMASolver.cpp)
target_link_libraries(solver ${DEPS})

add_library(lbfgs_solver STATIC LBFGS.h LBFGS.cpp)
target_link_libraries(lbfgs_solver ${DEPS})