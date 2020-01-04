#ifndef _MARCHING_CUBES_HEADER_
#define _MARCHING_CUBES_HEADER_

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <VolumetricFusion\Voxelgrid.hpp>

#include "Utils.hpp"
#include "Tables.hpp"
#include "Structs.hpp"

#include <iostream>
#include <fstream>

#include <math.h>

namespace vc::fusion {

    void exportToPly(std::vector<vc::fusion::Triangle> triangles);
        glm::vec3 VertexInterp(int isolevel, glm::vec3 p1, glm::vec3 p2, float valp1, float valp2);
    int Polygonise(vc::fusion::GridCell grid, double isolevel, vc::fusion::Triangle* triangles);

    void marchingCubes(vc::fusion::Voxelgrid* voxelgrid) {
        std::vector<vc::fusion::Triangle> triangles;

        int snx = voxelgrid->sizeNormalized.x;
        int sny = voxelgrid->sizeNormalized.y;
        int snz = voxelgrid->sizeNormalized.z;

        for (int z = 1; z < snz; z++)
        {
            for (int y = 1; y < sny; y++)
            {
                for (int x = 1; x < snx; x++)
                {
                    std::stringstream ss;

                    vc::fusion::GridCell cell = voxelgrid->getGridCell(x, y, z);
                    vc::fusion::Triangle* new_triangles = new vc::fusion::Triangle[10];
                    int num_new_triangles = Polygonise(cell, 0.0, new_triangles);

                    for (int i = 0; i < num_new_triangles; i++)
                    {
                        triangles.emplace_back(new_triangles[i]);
                    }

                    std::cout << ss.str() << std::endl;
                }
            }
        }

        exportToPly(triangles);
    }


    int Polygonise(vc::fusion::GridCell grid, double isolevel, vc::fusion::Triangle* triangles) {
        int i, ntriang;
        int cubeindex;
        glm::vec3 vertlist[12];

        /*
          Determine the index into the edge table which
          tells us which vertices are inside of the surface
       */
        cubeindex = 0;
        if (grid.val[0] < isolevel) cubeindex |= 1;
        if (grid.val[1] < isolevel) cubeindex |= 2;
        if (grid.val[2] < isolevel) cubeindex |= 4;
        if (grid.val[3] < isolevel) cubeindex |= 8;
        if (grid.val[4] < isolevel) cubeindex |= 16;
        if (grid.val[5] < isolevel) cubeindex |= 32;
        if (grid.val[6] < isolevel) cubeindex |= 64;
        if (grid.val[7] < isolevel) cubeindex |= 128;

        /* Cube is entirely in/out of the surface */
        if (edgeTable[cubeindex] == 0)
            return(0);

        /* Find the vertices where the surface intersects the cube */
        if (edgeTable[cubeindex] & 1)
            vertlist[0] =
            VertexInterp(isolevel, grid.p[0], grid.p[1], grid.val[0], grid.val[1]);
        if (edgeTable[cubeindex] & 2)
            vertlist[1] =
            VertexInterp(isolevel, grid.p[1], grid.p[2], grid.val[1], grid.val[2]);
        if (edgeTable[cubeindex] & 4)
            vertlist[2] =
            VertexInterp(isolevel, grid.p[2], grid.p[3], grid.val[2], grid.val[3]);
        if (edgeTable[cubeindex] & 8)
            vertlist[3] =
            VertexInterp(isolevel, grid.p[3], grid.p[0], grid.val[3], grid.val[0]);
        if (edgeTable[cubeindex] & 16)
            vertlist[4] =
            VertexInterp(isolevel, grid.p[4], grid.p[5], grid.val[4], grid.val[5]);
        if (edgeTable[cubeindex] & 32)
            vertlist[5] =
            VertexInterp(isolevel, grid.p[5], grid.p[6], grid.val[5], grid.val[6]);
        if (edgeTable[cubeindex] & 64)
            vertlist[6] =
            VertexInterp(isolevel, grid.p[6], grid.p[7], grid.val[6], grid.val[7]);
        if (edgeTable[cubeindex] & 128)
            vertlist[7] =
            VertexInterp(isolevel, grid.p[7], grid.p[4], grid.val[7], grid.val[4]);
        if (edgeTable[cubeindex] & 256)
            vertlist[8] =
            VertexInterp(isolevel, grid.p[0], grid.p[4], grid.val[0], grid.val[4]);
        if (edgeTable[cubeindex] & 512)
            vertlist[9] =
            VertexInterp(isolevel, grid.p[1], grid.p[5], grid.val[1], grid.val[5]);
        if (edgeTable[cubeindex] & 1024)
            vertlist[10] =
            VertexInterp(isolevel, grid.p[2], grid.p[6], grid.val[2], grid.val[6]);
        if (edgeTable[cubeindex] & 2048)
            vertlist[11] =
            VertexInterp(isolevel, grid.p[3], grid.p[7], grid.val[3], grid.val[7]);

        /* Create the triangle */
        ntriang = 0;
        for (i = 0; triTable[cubeindex][i] != -1; i += 3) {
            triangles[ntriang].p[0] = vertlist[triTable[cubeindex][i]];
            triangles[ntriang].p[1] = vertlist[triTable[cubeindex][i + 1]];
            triangles[ntriang].p[2] = vertlist[triTable[cubeindex][i + 2]];
            ntriang++;
        }

        return(ntriang);
    }

    /*
       Linearly interpolate the position where an isosurface cuts
       an edge between two vertices, each with their own scalar value
    */
    glm::vec3 VertexInterp(int isolevel, glm::vec3 p1, glm::vec3 p2, float valp1, float valp2)
    {
        double mu;
        glm::vec3 p;

        if (std::abs(isolevel - valp1) < 0.00001)
            return(p1);
        if (std::abs(isolevel - valp2) < 0.00001)
            return(p2);
        if (std::abs(valp1 - valp2) < 0.00001)
            return(p1);
        mu = (isolevel - valp1) / (valp2 - valp1);
        p.x = p1.x + mu * (p2.x - p1.x);
        p.y = p1.y + mu * (p2.y - p1.y);
        p.z = p1.z + mu * (p2.z - p1.z);

        return(p);
    }

    void testSingleCellMarchingCubes() {
        marchingCubes(new vc::fusion::SingleCellMockVoxelGrid());
    }

    void testFourCellMarchingCubes() {
        marchingCubes(new vc::fusion::FourCellMockVoxelGrid());
    }

    void exportToPly(std::vector<vc::fusion::Triangle> triangles) {
        std::ofstream ply_file;
        ply_file.open("plys/marching_cube.ply");

        ply_file << "ply\n";
        ply_file << "format ascii 1.0\n";

        ply_file << "comment Test comment\n";

        ply_file << "element vertex " << triangles.size() * 3 << "\n";
        ply_file << "property float x\n";
        ply_file << "property float y\n";
        ply_file << "property float z\n";


        ply_file << "element face " << triangles.size() << "\n";
        ply_file << "property list uchar int vertex_indices\n";
        ply_file << "end_header\n";

        for (auto triangle : triangles) {
            for (auto vertex : triangle.p) {
                ply_file << vertex.x << " " << vertex.y << " " << vertex.z << "\n";
            }
        }

        for (int i = 0; i < triangles.size(); i++) {
            ply_file << 3 << " " << i * 3 + 0 << " " << i * 3 + 1 << " " << i * 3 + 2 << "\n";
        }

        ply_file.close();
    }
}

#endif