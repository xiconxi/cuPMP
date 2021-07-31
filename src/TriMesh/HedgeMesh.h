//
// Created by pupa on 2021/7/29.
//

#ifndef CUTRI_HEDGEMESH_H
#define CUTRI_HEDGEMESH_H

#include <cstdint>
#include <string>
#include "Types.h"

namespace CuTri {
#define NO_ID 0x7fffffff

struct Handle {
    bool bit_: 1;
    uint32_t idx_: 31;

    explicit Handle(uint32_t _idx = -1) : idx_(_idx) {}

    uint32_t idx() { return idx_; }

    bool is_valid() const { return idx_ != -1; }

    bool operator==(const Handle &_rhs) const { return idx_ == _rhs.idx_; }

    bool operator!=(const Handle &_rhs) const { return idx_ != _rhs.idx_; }

    bool operator<(const Handle &_rhs) const { return idx_ < _rhs.idx_; }
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
    static Face face(Hedge hedge) { return Face{hedge.idx() / 3}; }

    static Hedge prev(Hedge hedge) { return Hedge{hedge.idx() / 3 * 3 + (hedge.idx() % 3 + 2) % 3}; }

    static Hedge next(Hedge hedge) { return Hedge{hedge.idx() / 3 * 3 + (hedge.idx() % 3 + 1) % 3}; }

    Vertex vertex_;
};

class HedgeTriMesh {
private:

public: //-------------------------------------------- constructor / destructor
    HedgeTriMesh();

    virtual ~HedgeTriMesh();

    HedgeTriMesh(const HedgeTriMesh &rhs) { operator=(rhs); }

    HedgeTriMesh &operator=(const HedgeTriMesh &rhs);

    HedgeTriMesh &assign(const HedgeTriMesh &rhs);

public: //------------------------------------------------------------- file IO
    bool read(const std::string &filename) {}

    bool write(const std::string &filename) const {}

public: //----------------------------------------------- add new vertex / face

    Vertex add_vertex(const Point &p);

    Face add_triangle(Vertex v1, Vertex v2, Vertex v3);
public: //---------------------------------------------- low-level connectivity

};


class HedgeMesh {

};

}


#endif // CUTRI_HEDGEMESH_H
