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
        Surface_mesh::Vertex v;

        // properties
        Surface_mesh::Vertex_property <Normal> normals;
        Surface_mesh::Vertex_property <Texture_coordinate> texcoords;
        Surface_mesh::Vertex_property <Color> colors;
        if (has_normals) normals = mesh.vertex_property<Normal>("v:normal");
        if (has_texcoords) texcoords = mesh.vertex_property<Texture_coordinate>("v:texcoord");
        if (has_colors) colors = mesh.vertex_property<Color>("v:color");


        // #Vertice, #Faces, #Edges
        items = fscanf(in, "%d %d %d\n", (int *) &nV, (int *) &nF, (int *) &nE);
        mesh.clear();
        mesh.reserve(nV, std::max(3 * nV, nE), nF);


        // read vertices: pos [normal] [color] [texcoord]
        for (i = 0; i < nV && !feof(in); ++i) {
            // read line
            lp = fgets(line, 200, in);
            lp = line;

            // position
            items = sscanf(lp, "%f %f %f%n", &p[0], &p[1], &p[2], &nc);
            assert(items == 3);
            v = mesh.add_vertex((Point) p);
            lp += nc;

            // normal
            if (has_normals) {
                if (sscanf(lp, "%f %f %f%n", &n[0], &n[1], &n[2], &nc) == 3) {
                    normals[v] = n;
                }
                lp += nc;
            }

            // color
            if (has_colors) {
                if (sscanf(lp, "%f %f %f%n", &c[0], &c[1], &c[2], &nc) == 3) {
                    if (c[0] > 1.0f || c[1] > 1.0f || c[2] > 1.0f) c *= (1.0 / 255.0);
                    colors[v] = c;
                }
                lp += nc;
            }

            // tex coord
            if (has_texcoords) {
                items = sscanf(lp, "%f %f%n", &t[0], &t[1], &nc);
                assert(items == 2);
                texcoords[v][0] = t[0];
                texcoords[v][1] = t[1];
                lp += nc;
            }
        }



        // read faces: #N v[1] v[2] ... v[n-1]
        std::vector<Surface_mesh::Vertex> vertices;
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
                vertices[j] = Surface_mesh::Vertex(idx);
                lp += nc;
            }
            mesh.add_face(vertices);
        }


        return true;
    }

    bool read_off_binary(HostMesh &mesh, FILE *in, bool has_normals, bool has_texcoords, bool has_colors) {
        unsigned int i, j, idx;
        unsigned int nV, nF, nE;
        float3 p, n, c;
        float2 t;
        Surface_mesh::Vertex v;


        // binary cannot (yet) read colors
        if (has_colors) return false;


        // properties
        Surface_mesh::Vertex_property <Normal> normals;
        Surface_mesh::Vertex_property <Texture_coordinate> texcoords;
        if (has_normals) normals = mesh.vertex_property<Normal>("v:normal");
        if (has_texcoords) texcoords = mesh.vertex_property<Texture_coordinate>("v:texcoord");


        // #Vertice, #Faces, #Edges
        read(in, nV);
        read(in, nF);
        read(in, nE);
        mesh.clear();
        mesh.reserve(nV, std::max(3 * nV, nE), nF);


        // read vertices: pos [normal] [color] [texcoord]
        for (i = 0; i < nV && !feof(in); ++i) {
            // position
            read(in, p);
            v = mesh.add_vertex((Point) p);

            // normal
            if (has_normals) {
                read(in, n);
                normals[v] = n;
            }

            // tex coord
            if (has_texcoords) {
                read(in, t);
                texcoords[v][0] = t[0];
                texcoords[v][1] = t[1];
            }
        }


        // read faces: #N v[1] v[2] ... v[n-1]
        std::vector<Surface_mesh::Vertex> vertices;
        for (i = 0; i < nF; ++i) {
            read(in, nV);
            vertices.resize(nV);
            for (j = 0; j < nV; ++j) {
                read(in, idx);
                vertices[j] = Surface_mesh::Vertex(idx);
            }
            mesh.add_face(vertices);
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
                read_off_binary(mesh, in, has_normals, has_texcoords, has_colors) :
                read_off_ascii(mesh, in, has_normals, has_texcoords, has_colors));


        fclose(in);
        return ok;
    }

}