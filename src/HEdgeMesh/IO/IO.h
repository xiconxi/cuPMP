//
// Created by 27890 on 2021/8/3.
//

#ifndef CUPMP_IO_H
#define CUPMP_IO_H

#include "../HEdgeMesh.cuh"

namespace cuPMP {
//    bool read_mesh(HostMesh& mesh, const std::string& filename){}
    bool read_off(HostMesh& mesh, const std::string& filename);
//    bool read_obj(HostMesh& mesh, const std::string& filename){}
//    bool read_poly(HostMesh& mesh, const std::string& filename){}
//    bool read_stl(HostMesh& mesh, const std::string& filename){}
//
//    bool write_mesh(const HostMesh& mesh, const std::string& filename){}
    bool write_off(const HostMesh& mesh, const std::string& filename);
//    bool write_obj(const HostMesh& mesh, const std::string& filename){}
//    bool write_poly(const HostMesh& mesh, const std::string& filename){}
//    bool write_stl(const HostMesh& mesh, const std::string& filename){}
}

#endif //CUPMP_IO_H
