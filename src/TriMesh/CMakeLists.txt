file(GLOB SOURCES ./*.cu)
file(GLOB HEADERS ./*.h)

add_library(cuPMP STATIC ${SOURCES} ${HEADERS})

set_target_properties(cuPMP PROPERTIES VERSION 1.0)
set_target_properties(cuPMP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#install(TARGETS surface_mesh DESTINATION lib)
#install(FILES ${HEADERS} DESTINATION include/surface_mesh/)
