//
// Created by 27890 on 2021/8/3.
//
#include "IO.h"
#include <vector_types.h>

namespace cuPMP{

    // helper function
    template<typename T>
    int read(FILE *in, T &t) {
        int err = 0;
        err = fread(&t, 1, sizeof(t), in);
        return err;
    }

    bool read_off_ascii(HostMesh &mesh, FILE *in, bool has_normals,  bool has_texcoords, bool has_colors) {
        char line[200], *lp;
        int nc;
        unsigned int i, j, items, idx;
        unsigned int nV, nF, nE;
        float3 p, n, c;
        float2 t;
        Vertex v;
        Face   f;

        // #Vertice, #Faces, #Edges
        items = fscanf(in, "%d %d %d\n", (int *) &nV, (int *) &nF, (int *) &nE);
        mesh.clear();
        mesh.reserve(nV, nF);

        // read vertices: pos [normal] [color] [texcoord]
        for (i = 0; i < nV && !feof(in); ++i) {
            // read line
            lp = fgets(line, 200, in);
            lp = line;

            // position
            items = sscanf(lp, "%f %f %f%n", &p.x, &p.y, &p.z, &nc);
            assert(items == 3);
            v = mesh.add_vertex( p);
            lp += nc;

            // normal
//            if (has_normals) {
//                if (sscanf(lp, "%f %f %f%n", &n.x, &n.y, &n.z, &nc) == 3) {
//                    normals[v] = n;
//                }
//                lp += nc;
//            }

            // color
//            if (has_colors) {
//                if (sscanf(lp, "%f %f %f%n", &c.x, &c.y, &c.z, &nc) == 3) {
//                    if (c.x > 1.0f || c.y > 1.0f || c.z > 1.0f) c *= (1.0 / 255.0);
//                    colors[v] = c;
//                }
//                lp += nc;
//            }

            // tex coord
//            if (has_texcoords) {
//                items = sscanf(lp, "%f %f%n", &t.x, &t.y, &nc);
//                assert(items == 2);
//                texcoords[v].x = t.x;
//                texcoords[v].y = t.y;
//                lp += nc;
//            }
        }
        
        // read faces: #N v.y v.z ... v[n-1]
        std::vector<Vertex> vertices;
        for (i = 0; i < nF; ++i) {
            // read line
            lp = fgets(line, 200, in);
            lp = line;
            // #vertices
            items = sscanf(lp, "%d%n", (int *) &nV, &nc);
            assert(items == 1);
            vertices.resize(nV);
            lp += nc;
            // indices
            for (j = 0; j < nV; ++j) {
                items = sscanf(lp, "%d%n", (int *) &idx, &nc);
                assert(items == 1);
                vertices[j] = Vertex(idx);
                lp += nc;
            }
            assert(vertices.size() == 3);
            mesh.add_triangle(vertices[0], vertices[1], vertices[2]);
        }
        return true;
    }

    bool read_off_binary(HostMesh &mesh, FILE *in, bool has_normals, bool has_texcoords, bool has_colors) {
        unsigned int i, j, idx;
        unsigned int nV, nF, nE;
        float3 p, n, c;
        float2 t;
        Vertex v;


        // binary cannot (yet) read colors
//        if (has_colors) return false;


        // #Vertice, #Faces, #Edges
        read(in, nV);
        read(in, nF);
        read(in, nE);
        mesh.clear();
        mesh.reserve(nV, nF);


        // read vertices: pos [normal] [color] [texcoord]
        for (i = 0; i < nV && !feof(in); ++i) {
            // position
            read(in, p);
            v = mesh.add_vertex(p);

//            // normal
//            if (has_normals) {
//                read(in, n);
//                normals[v] = n;
//            }
//
//            // tex coord
//            if (has_texcoords) {
//                read(in, t);
//                texcoords[v].x = t.x;
//                texcoords[v].y = t.y;
//            }
        }

        // read faces: #N v.y v.z ... v[n-1]
        std::vector<Vertex> vertices;
        for (i = 0; i < nF; ++i) {
            read(in, nV);
            vertices.resize(nV);
            for (j = 0; j < nV; ++j) {
                read(in, idx);
                vertices[j] = Vertex(idx);
            }
            assert(vertices.size() == 3);
            mesh.add_triangle(vertices[0], vertices[1], vertices[2]);
        }

        return true;
    }

    bool read_off(HostMesh &mesh, const std::string &filename) {
        char line[200];
        bool has_texcoords = false;
        bool has_normals = false;
        bool has_colors = false;
        bool has_hcoords = false;
        bool has_dim = false;
        bool is_binary = false;

        // open file (in ASCII mode)
        FILE *in = fopen(filename.c_str(), "r");
        if (!in) return false;

        // read header: [ST][C][N][4][n]OFF BINARY
        char *c = fgets(line, 200, in);
        assert(c != NULL);
        c = line;
        if (c[0] == 'S' && c[1] == 'T') {
            has_texcoords = true; c += 2;
        } else if (c[0] == 'C') {
            has_colors = true; ++c;
        }else if (c[0] == 'N') {
            has_normals = true; ++c;
        }else if (c[0] == '4') {
            has_hcoords = true; ++c;
        }else if (c[0] == 'n') {
            has_dim = true; ++c;
        }else if (strncmp(c, "OFF", 3) != 0) {
            fclose(in);
            return false;
        } // no OFF
        if (strncmp(c + 4, "BINARY", 6) == 0) is_binary = true;

        // homogeneous coords, and vertex dimension != 3 are not supported
        if (has_hcoords || has_dim) {
            fclose(in);
            return false;
        }

        // if binary: reopen file in binary mode
        if (is_binary) {
            fclose(in);
            in = fopen(filename.c_str(), "rb");
            c = fgets(line, 200, in);
            assert(c != NULL);
        }

        // read as ASCII or binary
        bool ok = (is_binary ?
//                true:
                read_off_binary(mesh, in, has_normals, has_texcoords, has_colors) :
                read_off_ascii(mesh, in, has_normals, has_texcoords, has_colors));


        fclose(in);
        return ok;
    }

    bool write_off(const HostMesh& mesh, const std::string& filename)
    {
        FILE* out = fopen(filename.c_str(), "w");
        if (!out)
            return false;

        fprintf(out, "OFF\n%d %d 0\n", mesh.n_vertices(), mesh.n_faces());
        for(size_t i = 0; i < mesh.n_vertices(); i++) {
            const float3& p = mesh.v_position_[i];
            fprintf(out, "%.10f %.10f %.10f", p.x, p.y, p.z);
            fprintf(out, "\n");
        }

        auto &hcon = mesh.h_conn_;
        for(size_t i = 0; i < mesh.n_hedges(); i += 3) {
            fprintf(out, "3 %d %d %d", hcon[i].vertex_.i(), hcon[i+1].vertex_.i(), hcon[i+2].vertex_.i());
            fprintf(out, "\n");
        }

        fclose(out);
        return true;
    }

}