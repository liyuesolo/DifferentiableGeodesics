mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. -G Ninja
cmake --build . -j8 DifferentiableGeodesics