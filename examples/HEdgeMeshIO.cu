//
// Created by 27890 on 2021/8/3.
//

#include <HEdgeMesh/HEdgeMesh.cuh>
#include <HEdgeMesh/IO/IO.h>
#include <iostream>


int main(int argc, char **argv) {
    cuPMP::HostMesh host_mesh;
    cuPMP::read_off(host_mesh, argv[1]);

    std::cout << host_mesh.n_vertices() << ' ' << host_mesh.n_faces() << ' ' << host_mesh.n_hedges() << std::endl;
}