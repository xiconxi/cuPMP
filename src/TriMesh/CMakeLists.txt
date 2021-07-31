file(GLOB SOURCES ./*.cpp ./io/*.cpp)
file(GLOB HEADERS ./*.h)

if(UNIX)
  add_library(surface_mesh SHARED ${SOURCES} ${HEADERS})
elseif(WIN32)
  add_library(surface_mesh STATIC ${SOURCES} ${HEADERS} HedgeMesh.cpp HedgeMesh.h)
endif()

set_target_properties(surface_mesh PROPERTIES VERSION 1.0)

install(TARGETS surface_mesh DESTINATION lib)

install(FILES ${HEADERS} DESTINATION include/surface_mesh/)
