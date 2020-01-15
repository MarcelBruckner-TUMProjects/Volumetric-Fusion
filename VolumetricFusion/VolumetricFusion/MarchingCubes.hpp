#ifndef _MARCHING_CUBES_HEADER_
#define _MARCHING_CUBES_HEADER_

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Voxelgrid.hpp"

#include "Utils.hpp"
#include "Tables.hpp"
#include "Structs.hpp"
#include "ceres/ceres.h"

#include <iostream>
#include <fstream>

#include <math.h>

namespace vc::fusion {
    class Voxelgrid;

    void exportToPly(std::vector<vc::fusion::Triangle*> triangles);
        Eigen::Vector3d VertexInterp(int isolevel, Eigen::Vector3d p1, Eigen::Vector3d p2, float valp1, float valp2);
        std::vector< vc::fusion::Triangle*> Polygonise(vc::fusion::GridCell grid, double isolevel);

    void marchingCubes(vc::fusion::Voxelgrid* voxelgrid) {
        int snx = voxelgrid->sizeNormalized[0];
        int sny = voxelgrid->sizeNormalized[1];
        int snz = voxelgrid->sizeNormalized[2];

        std::vector<vc::fusion::Triangle*> triangles(snx * sny * snz * 5);

        std::vector<std::thread> threads;
        int i = 0;
        int yy = 0;
        for (int z = 1; z < snz; z++)
        {
            for (int y = 1; y < sny; y++)
            {
                threads.emplace_back(std::thread([&, y, z]() {
                    for (int x = 1; x < snx; x++)
                    {
                        std::stringstream ss;

                        vc::fusion::GridCell cell;
                        if (!voxelgrid->getGridCell(x, y, z, &cell)) {
                            continue;
                        }
                        
                        std::vector<vc::fusion::Triangle*> new_triangles = Polygonise(cell, 0.0);

                        for (auto& triangle : new_triangles)
                        {
                            if (triangle) {
                                triangles[i++] = triangle;
                            }
                        }
                    }
                }));
                if (yy++ >= vc::utils::NUM_THREADS) {
                    for (auto& thread : threads)
                    {
                        thread.join();
                    }
                    threads = std::vector<std::thread>();
                    yy = 0;
                }
            }
            std::cout << "Calculated Marching Cubes layer " << z << std::endl;
        }

        for (auto& thread : threads)
        {
            thread.join();
        }

        std::vector<vc::fusion::Triangle*> finalTriangles(0);
        finalTriangles.insert(finalTriangles.end(), triangles.begin(), triangles.begin() + i);
        exportToPly(finalTriangles);
    }


    std::vector< vc::fusion::Triangle*> Polygonise(vc::fusion::GridCell grid, double isolevel) {
        std::vector<vc::fusion::Triangle*> triangles;

        int i, ntriang;
        int cubeindex;
        Eigen::Vector3d vertlist[12];

        /*
          Determine the index into the edge table which
          tells us which vertices are inside of the surface
       */
        cubeindex = 0;
        if (grid.tsdfs[0] < isolevel) cubeindex |= 1;
        if (grid.tsdfs[1] < isolevel) cubeindex |= 2;
        if (grid.tsdfs[2] < isolevel) cubeindex |= 4;
        if (grid.tsdfs[3] < isolevel) cubeindex |= 8;
        if (grid.tsdfs[4] < isolevel) cubeindex |= 16;
        if (grid.tsdfs[5] < isolevel) cubeindex |= 32;
        if (grid.tsdfs[6] < isolevel) cubeindex |= 64;
        if (grid.tsdfs[7] < isolevel) cubeindex |= 128;

        /* Cube is entirely in/out of the surface */
        if (edgeTable[cubeindex] == 0) {
            return triangles;
        }

        /* Find the vertices where the surface intersects the cube */
        if (edgeTable[cubeindex] & 1 && grid.tsdfs[0] != vc::fusion::INVALID_TSDF_VALUE  && grid.tsdfs[1] != vc::fusion::INVALID_TSDF_VALUE)
            vertlist[0] =
            VertexInterp(isolevel, grid.corners[0], grid.corners[1], grid.tsdfs[0], grid.tsdfs[1]);
        if (edgeTable[cubeindex] & 2 && grid.tsdfs[1] != vc::fusion::INVALID_TSDF_VALUE && grid.tsdfs[2] != vc::fusion::INVALID_TSDF_VALUE)
            vertlist[1] =
            VertexInterp(isolevel, grid.corners[1], grid.corners[2], grid.tsdfs[1], grid.tsdfs[2]);
        if (edgeTable[cubeindex] & 4 && grid.tsdfs[2] != vc::fusion::INVALID_TSDF_VALUE && grid.tsdfs[3] != vc::fusion::INVALID_TSDF_VALUE)
            vertlist[2] =
            VertexInterp(isolevel, grid.corners[2], grid.corners[3], grid.tsdfs[2], grid.tsdfs[3]);
        if (edgeTable[cubeindex] & 8 && grid.tsdfs[3] != vc::fusion::INVALID_TSDF_VALUE && grid.tsdfs[0] != vc::fusion::INVALID_TSDF_VALUE)
            vertlist[3] =
            VertexInterp(isolevel, grid.corners[3], grid.corners[0], grid.tsdfs[3], grid.tsdfs[0]);
        if (edgeTable[cubeindex] & 16 && grid.tsdfs[4] != vc::fusion::INVALID_TSDF_VALUE && grid.tsdfs[5] != vc::fusion::INVALID_TSDF_VALUE)
            vertlist[4] =
            VertexInterp(isolevel, grid.corners[4], grid.corners[5], grid.tsdfs[4], grid.tsdfs[5]);
        if (edgeTable[cubeindex] & 32 && grid.tsdfs[5] != vc::fusion::INVALID_TSDF_VALUE && grid.tsdfs[6] != vc::fusion::INVALID_TSDF_VALUE)
            vertlist[5] =
            VertexInterp(isolevel, grid.corners[5], grid.corners[6], grid.tsdfs[5], grid.tsdfs[6]);
        if (edgeTable[cubeindex] & 64 && grid.tsdfs[6] != vc::fusion::INVALID_TSDF_VALUE && grid.tsdfs[7] != vc::fusion::INVALID_TSDF_VALUE)
            vertlist[6] =
            VertexInterp(isolevel, grid.corners[6], grid.corners[7], grid.tsdfs[6], grid.tsdfs[7]);
        if (edgeTable[cubeindex] & 128 && grid.tsdfs[7] != vc::fusion::INVALID_TSDF_VALUE && grid.tsdfs[4] != vc::fusion::INVALID_TSDF_VALUE)
            vertlist[7] =
            VertexInterp(isolevel, grid.corners[7], grid.corners[4], grid.tsdfs[7], grid.tsdfs[4]);
        if (edgeTable[cubeindex] & 256 && grid.tsdfs[0] != vc::fusion::INVALID_TSDF_VALUE && grid.tsdfs[4] != vc::fusion::INVALID_TSDF_VALUE)
            vertlist[8] =
            VertexInterp(isolevel, grid.corners[0], grid.corners[4], grid.tsdfs[0], grid.tsdfs[4]);
        if (edgeTable[cubeindex] & 512 && grid.tsdfs[1] != vc::fusion::INVALID_TSDF_VALUE && grid.tsdfs[5] != vc::fusion::INVALID_TSDF_VALUE)
            vertlist[9] =
            VertexInterp(isolevel, grid.corners[1], grid.corners[5], grid.tsdfs[1], grid.tsdfs[5]);
        if (edgeTable[cubeindex] & 1024 && grid.tsdfs[2] != vc::fusion::INVALID_TSDF_VALUE && grid.tsdfs[6] != vc::fusion::INVALID_TSDF_VALUE)
            vertlist[10] =
            VertexInterp(isolevel, grid.corners[2], grid.corners[6], grid.tsdfs[2], grid.tsdfs[6]);
        if (edgeTable[cubeindex] & 2048 && grid.tsdfs[3] != vc::fusion::INVALID_TSDF_VALUE && grid.tsdfs[7] != vc::fusion::INVALID_TSDF_VALUE)
            vertlist[11] =
            VertexInterp(isolevel, grid.corners[3], grid.corners[7], grid.tsdfs[3], grid.tsdfs[7]);

        /* Create the triangle */
        for (i = 0; triTable[cubeindex][i] != -1; i += 3) {
            auto a = vertlist[triTable[cubeindex][i]];
            auto b = vertlist[triTable[cubeindex][i + 1]];
            auto c = vertlist[triTable[cubeindex][i + 2]];
            if (vc::utils::isValid(a) && vc::utils::isValid(b) && vc::utils::isValid(c)) {
                auto triangle = new vc::fusion::Triangle(a, b, c);
                if (triangle) {
                    triangles.emplace_back(triangle);
                }
            }
        }

        return triangles;
    }

    /*
       Linearly interpolate the position where an isosurface cuts
       an edge between two vertices, each with their own scalar value
    */
    Eigen::Vector3d VertexInterp(int isolevel, Eigen::Vector3d p1, Eigen::Vector3d p2, float valp1, float valp2)
    {
        double mu;
        Eigen::Vector3d p;

        if (std::abs(isolevel - valp1) < 0.00001)
            return(p1);
        if (std::abs(isolevel - valp2) < 0.00001)
            return(p2);
        if (std::abs(valp1 - valp2) < 0.00001)
            return(p1);
        mu = (isolevel - valp1) / (valp2 - valp1);
        p[0] = p1[0] + mu * (p2[0] - p1[0]);
        p[1] = p1[1] + mu * (p2[1] - p1[1]);
        p[2] = p1[2] + mu * (p2[2] - p1[2]);

        return(p);
    }

    void testSingleCellMarchingCubes() {
        marchingCubes(new vc::fusion::SingleCellMockVoxelGrid());
    }

    void testFourCellMarchingCubes() {
        marchingCubes(new vc::fusion::FourCellMockVoxelGrid());
    }

    void exportToPly(std::vector<vc::fusion::Triangle*> triangles) {
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

        int i = 0;
        for (auto triangle : triangles) {
            for (auto vertex : triangle->vertices) {
                if (vc::utils::isValid(vertex)) {
                    for (int i = 0; i < 3; i++)
                    {
                        ply_file << vertex[i] << " ";
                    }
                    ply_file << "\n";
                }
                else {
                    std::cout << "error with triangle " << i << std::endl;
                }
            }
            i++;

        }

        for (int i = 0; i < triangles.size(); i++) {
            ply_file << 3 << " " << i * 3 + 0 << " " << i * 3 + 1 << " " << i * 3 + 2 << "\n";
        }

        ply_file.close();

        std::cout << "Written ply" << std::endl;
    }
}

#endif