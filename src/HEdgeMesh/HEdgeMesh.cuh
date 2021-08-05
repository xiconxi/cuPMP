//
// Created by pupa on 2021/7/29.
//

#ifndef CUTRI_HEDGEMESH_H
#define CUTRI_HEDGEMESH_H

#include <c++/7/string>
#include <cuda.h>
#include <vector_types.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "HEdgeMesh/Backup/Properties.h"

namespace cuPMP {
#define NO_ID 0x7fffffff

class Handle {
    bool bit_: 1;
    uint32_t idx_: 31;
public:
    explicit __device__ __host__  Handle(uint32_t _idx = 0xffffffff) : bit_(false), idx_(_idx) {}
    uint32_t __device__ __host__ i() const { return idx_; }
    bool __device__ __host__ is_valid() const { return (idx_ & NO_ID) == NO_ID; }
    bool __device__ __host__ bit() const { return bit_; }
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
struct VConnect {
    Hedge    hedge_;
};

struct HConnect {
    __device__ __host__ static Face face(Hedge h) { return Face{h.i() / 3}; }
    __device__ __host__ static Hedge prev(Hedge h) {  return Hedge{h.i() / 3 * 3 + (h.i() % 3 + 2) % 3};}
    __device__ __host__ static Hedge next(Hedge h) { return Hedge{h.i() / 3 * 3 + (h.i() % 3 + 1) % 3};}
    __device__ __host__ bool is_border() { return twin_.bit();}
    Hedge   twin_;
    Vertex  vertex_;
};

template <template <class...> class Vector>
struct SurfaceMesh{
    Vector<float3>      v_position_;

    Vector<VConnect>    v_conn_;
    Vector<HConnect>    h_conn_;
    Vector<uint32_t>    v_flag_;
    Vector<uint32_t>    f_flag_;

    enum class FLAG{ DELETE = 0x01, DIRTY  = 0x02, LOCKED = 0x04, IS_B = 0x08};
    __host__ uint32_t n_faces() const { return f_flag_.size(); }
    __host__ uint32_t n_hedges() const { return h_conn_.size(); }
    __host__ uint32_t n_vertices() const { return v_flag_.size(); }
    __host__ void reserve(uint32_t n_v, uint32_t n_f) {
        v_conn_.reserve(n_v);
        v_position_.reserve(n_v);
        v_flag_.reserve(n_v);
        h_conn_.reserve(n_f * 3);
        f_flag_.reserve(n_f);
    }

    __host__ void clear() {
        v_conn_.clear();
        v_position_.clear();
        v_flag_.clear();
        h_conn_.clear();
        f_flag_.clear();
    }

    __host__ Vertex add_vertex(const float3& point) {
        v_position_.push_back(point);
        v_flag_.push_back(0);
        v_conn_.push_back(VConnect());
    }

    __host__ Face add_triangle(Vertex v0, Vertex v1, Vertex v2) {
        assert(v0 != v1 && v1 != v2 && v0 != v2);
        f_flag_.push_back(0);
        h_conn_.push_back(HConnect{Hedge(), v0});
        h_conn_.push_back(HConnect{Hedge(), v1});
        h_conn_.push_back(HConnect{Hedge(), v2});
    }


};

using HostMesh      = SurfaceMesh<thrust::host_vector>;
using DeviceMesh    = SurfaceMesh<thrust::device_vector>;


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
