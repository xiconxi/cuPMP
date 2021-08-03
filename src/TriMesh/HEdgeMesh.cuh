//
// Created by pupa on 2021/7/29.
//

#ifndef CUTRI_HEDGEMESH_H
#define CUTRI_HEDGEMESH_H

#include <string>
#include <cuda.h>
#include <vector_types.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "Properties.h"

namespace cuPMP {
#define NO_ID 0x7fffffff

class Handle {
    bool bit_: 1;
    uint32_t idx_: 31;
public:
    explicit __device__ __host__  Handle(uint32_t _idx = 0xffffffff) : bit_(false), idx_(_idx) {}
    uint32_t __device__ __host__ operator()() const { return idx_; }
    bool __device__ __host__ is_valid() const { return idx_ & NO_ID == NO_ID; }
    bool __device__ __host__ operator==(const Handle &_rhs) const { return idx_ == _rhs.idx_; }
    bool __device__ __host__ operator!=(const Handle &_rhs) const { return idx_ != _rhs.idx_; }
    bool __device__ __host__ operator<(const Handle &_rhs) const { return idx_ < _rhs.idx_; }
};

struct Vertex : public Handle {
    using Handle::Handle;
};
struct Hedge : public Handle {
    using Handle::Handle;
};
struct Face : public Handle {
    using Handle::Handle;
};

/// outgoing half edge (it will be a boundary one for boundary vertices)
struct VertexConnectivity {
    Hedge hedge_;
};
struct FaceConnectivity {
    Hedge hedge_;
};
struct HedgeConnectivity {
    static Face __device__ __host__ face(Hedge h) { return Face{h() / 3}; }
    static Hedge __device__ __host__ prev(Hedge h) {  return Hedge{h() / 3 * 3 + (h() % 3 + 2) % 3};}
    static Hedge __device__ __host__ next(Hedge h) { return Hedge{h() / 3 * 3 + (h() % 3 + 1) % 3};}
    Vertex vertex_;
};


template<class Allocator>
struct DeviceSurfaceMesh{
    __host__ __device__ void Test();
    Allocator<float3> V;
};
//
//
//class HEdgeTriMesh{
//private:
//
//public: //-------------------------------------------- constructor / destructor
//    HEdgeTriMesh();
//
//
//    HEdgeTriMesh(const HEdgeTriMesh &rhs) { operator=(rhs); }
//
//    HEdgeTriMesh &operator=(const HEdgeTriMesh &rhs);
//
//    HEdgeTriMesh &assign(const HEdgeTriMesh &rhs);
//
//public: //------------------------------------------------------------- file IO
//    bool read(const std::string &filename) {}
//
//    bool write(const std::string &filename) const {}
//
//public: //----------------------------------------------- add new vertex / face
//
//    Vertex add_vertex(const float3 &p);
//
//    Face add_triangle(Vertex v1, Vertex v2, Vertex v3);
//public: //---------------------------------------------- low-level connectivity
//
//};
//

}


#endif // CUTRI_HEDGEMESH_H
