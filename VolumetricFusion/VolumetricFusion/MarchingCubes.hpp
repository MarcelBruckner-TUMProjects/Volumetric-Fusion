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
    glm::vec4 VertexInterp(double isolevel, vc::fusion::Vertex _p1, vc::fusion::Vertex _p2);
    std::vector< vc::fusion::Triangle*> Polygonise(vc::fusion::GridCell grid, double isolevel);

    class MarchingCubes {
    private:
        GLuint vbos[3];
        vc::fusion::Vertex* verts;
        std::vector<vc::fusion::Triangle> triangles;
        GLuint triangleCount;

        GLuint edgeTable;
        GLuint triTable;

        vc::rendering::ComputeShader* marchingCubesComputeShader;
        vc::rendering::ComputeShader* countTrianglesComputeShader;

    public:
        MarchingCubes() {
            marchingCubesComputeShader = new vc::rendering::ComputeShader("shader/marchingCubes.comp");
            countTrianglesComputeShader = new vc::rendering::ComputeShader("shader/countTriangles.comp");

            glGenBuffers(3, vbos);
            glGenBuffers(1, &edgeTable);
            glGenBuffers(1, &triTable);
        }

        void countTriangles(Eigen::Vector3d size, vc::fusion::Vertex* verts) {
            int num_verts = size[0] * size[1] * size[2];
            
            countTrianglesComputeShader->use();
            countTrianglesComputeShader->setVec3("sizeNormalized", size);
            countTrianglesComputeShader->setFloat("isolevel", 0.0f);

            glBindBuffer(GL_ARRAY_BUFFER, vbos[0]);
            glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * num_verts, verts, GL_DYNAMIC_COPY);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vbos[0]);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            GLuint tmp_numTriangles[1] = { 0 };

            glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, vbos[2]);
            glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);
            glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), tmp_numTriangles);
            glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 2, vbos[2]);
            glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

            //glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, vbos[2]);
            //glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint) * 1, tmp_numTriangles);
            //glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

            glDispatchCompute(num_verts, 1, 1);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);

            GLuint userCounters[1];
            glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, vbos[2]);
            glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint) * 1, userCounters);
            glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
            triangleCount = userCounters[0];

            //triangleCount = *tmp_numTriangles;
        }

        void compute(Eigen::Vector3d size, std::vector<vc::fusion::Vertex> verts) {

            //countTriangles(size, verts);

            int snx = size[0];
            int sny = size[1];
            int snz = size[2];

            int num_verts = snx * sny * snz;
            int num_triangles = num_verts * 5;

            triangles = std::vector<vc::fusion::Triangle>(num_triangles);

            marchingCubesComputeShader->use();
            marchingCubesComputeShader->setVec3("sizeNormalized", size);
            marchingCubesComputeShader->setFloat("isolevel", 0.0f);
            marchingCubesComputeShader->setInt("INVALID_TSDF_VALUE", vc::fusion::INVALID_TSDF_VALUE);

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, vbos[0]);
            glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Vertex) * num_verts, verts.data(), GL_DYNAMIC_COPY);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vbos[0]);

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, vbos[1]);
            glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Triangle) * num_triangles, triangles.data(), GL_DYNAMIC_COPY);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vbos[1]);

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, edgeTable);
            glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(vc::fusion::edgeTable), vc::fusion::edgeTable, GL_DYNAMIC_COPY);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, edgeTable);

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, triTable);
            glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(vc::fusion::triTable), vc::fusion::triTable, GL_DYNAMIC_COPY);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, triTable);

            glDispatchCompute(num_verts, 1, 1);

            glMemoryBarrier(GL_ALL_BARRIER_BITS);

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, vbos[1]);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(Triangle) * num_triangles, triangles.data());

            for (int i = 0; i < 4 && i < num_triangles; i++) {
                std::cout << vc::utils::toString(&triangles[i]);
            }
            std::vector<vc::fusion::Triangle*> finalTriangles(0);
            for (auto& triangle : triangles)
            {
                if (vc::utils::isValid(&triangle)) {
                    finalTriangles.push_back(&triangle);
                }
            }
            std::cout << std::endl;
            exportToPly(finalTriangles);

        }
    };

        void marchingCubes(vc::fusion::Voxelgrid * voxelgrid) {
            int snx = voxelgrid->sizeNormalized[0];
            int sny = voxelgrid->sizeNormalized[1];
            int snz = voxelgrid->sizeNormalized[2];

            std::vector<vc::fusion::Triangle*> triangles(snx * sny * snz * 5);

            std::vector<std::thread> threads;
            int i = 0;
            int yy = 0;
            for (int z = 0; z < snz - 1; z++)
            {
                for (int y = 0; y < sny - 1; y++)
                {
                    threads.emplace_back(std::thread([&, y, z]() {
                        for (int x = 0; x < snx - 1; x++)
                        {
                            std::stringstream ss;

                            vc::fusion::GridCell cell;
                            if (!voxelgrid->getGridCell(x, y, z, &cell)) {
                                continue;
                            }

                            std::cout << x + y * snx + z * snx * sny << ":: ";
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
            glm::vec4 vertlist[12];

            /*
              Determine the index into the edge table which
              tells us which vertices are inside of the surface
           */
            cubeindex = 0;
            if (grid.verts[0].tsdf[1] < isolevel) cubeindex |= 1;
            if (grid.verts[1].tsdf[1] < isolevel) cubeindex |= 2;
            if (grid.verts[2].tsdf[1] < isolevel) cubeindex |= 4;
            if (grid.verts[3].tsdf[1] < isolevel) cubeindex |= 8;
            if (grid.verts[4].tsdf[1] < isolevel) cubeindex |= 16;
            if (grid.verts[5].tsdf[1] < isolevel) cubeindex |= 32;
            if (grid.verts[6].tsdf[1] < isolevel) cubeindex |= 64;
            if (grid.verts[7].tsdf[1] < isolevel) cubeindex |= 128;

            std::cout << cubeindex << std::endl;

            /* Cube is entirely in/out of the surface */
            if (edgeTable[cubeindex] == 0) {
                return triangles;
            }

            /* Find the vertices where the surface intersects the cube */
            if (edgeTable[cubeindex] & 1 && grid.verts[0].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE && grid.verts[1].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE)
                vertlist[0] =
                VertexInterp(isolevel, grid.verts[0], grid.verts[1]);
            if (edgeTable[cubeindex] & 2 && grid.verts[1].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE && grid.verts[2].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE)
                vertlist[1] =
                VertexInterp(isolevel, grid.verts[1], grid.verts[2]);
            if (edgeTable[cubeindex] & 4 && grid.verts[2].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE && grid.verts[3].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE)
                vertlist[2] =
                VertexInterp(isolevel, grid.verts[2], grid.verts[3]);
            if (edgeTable[cubeindex] & 8 && grid.verts[3].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE && grid.verts[0].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE)
                vertlist[3] =
                VertexInterp(isolevel, grid.verts[3], grid.verts[0]);
            if (edgeTable[cubeindex] & 16 && grid.verts[4].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE && grid.verts[5].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE)
                vertlist[4] =
                VertexInterp(isolevel, grid.verts[4], grid.verts[5]);
            if (edgeTable[cubeindex] & 32 && grid.verts[5].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE && grid.verts[6].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE)
                vertlist[5] =
                VertexInterp(isolevel, grid.verts[5], grid.verts[6]);
            if (edgeTable[cubeindex] & 64 && grid.verts[6].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE && grid.verts[7].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE)
                vertlist[6] =
                VertexInterp(isolevel, grid.verts[6], grid.verts[7]);
            if (edgeTable[cubeindex] & 128 && grid.verts[7].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE && grid.verts[4].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE)
                vertlist[7] =
                VertexInterp(isolevel, grid.verts[7], grid.verts[4]);
            if (edgeTable[cubeindex] & 256 && grid.verts[0].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE && grid.verts[4].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE)
                vertlist[8] =
                VertexInterp(isolevel, grid.verts[0], grid.verts[4]);
            if (edgeTable[cubeindex] & 512 && grid.verts[1].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE && grid.verts[5].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE)
                vertlist[9] =
                VertexInterp(isolevel, grid.verts[1], grid.verts[5]);
            if (edgeTable[cubeindex] & 1024 && grid.verts[2].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE && grid.verts[6].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE)
                vertlist[10] =
                VertexInterp(isolevel, grid.verts[2], grid.verts[6]);
            if (edgeTable[cubeindex] & 2048 && grid.verts[3].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE && grid.verts[7].tsdf[1] != vc::fusion::INVALID_TSDF_VALUE)
                vertlist[11] =
                VertexInterp(isolevel, grid.verts[3], grid.verts[7]);

            for (int i = 0; i < 12; i++)
            {
                std::cout << vc::utils::toString(vertlist[i]);
            }

            /* Create the triangle */
            for (i = 0; triTable[cubeindex][i] != -1; i += 3) {
                auto a = vertlist[triTable[cubeindex][i]];
                auto b = vertlist[triTable[cubeindex][i + 1]];
                auto c = vertlist[triTable[cubeindex][i + 2]];
                auto triangle = new vc::fusion::Triangle();

                if (vc::utils::isValid(a) && vc::utils::isValid(b) && vc::utils::isValid(c)) {
                    auto triangle = new vc::fusion::Triangle();
                    triangle->pos[0] = a;
                    triangle->pos[1] = b;
                    triangle->pos[2] = c;
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
        glm::vec4 VertexInterp(double isolevel, vc::fusion::Vertex _p1, vc::fusion::Vertex _p2)
        {
            glm::vec4 p1 = glm::vec4(_p1.pos[0], _p1.pos[1], _p1.pos[2], 1.0f);
            glm::vec4 p2 = glm::vec4(_p2.pos[0], _p2.pos[1], _p2.pos[2], 1.0f);

            double mu;
            glm::vec4 p;

            if (std::abs(isolevel - _p1.tsdf[1]) < 0.00001)
                return(p1);
            if (std::abs(isolevel - _p2.tsdf[1]) < 0.00001)
                return(p2);
            if (std::abs(_p1.tsdf[1] - _p2.tsdf[1]) < 0.00001)
                return(p1);
            mu = (isolevel - _p1.tsdf[1]) / (_p2.tsdf[1] - _p1.tsdf[1]);
            p[0] = p1[0] + mu * (p2[0] - p1[0]);
            p[1] = p1[1] + mu * (p2[1] - p1[1]);
            p[2] = p1[2] + mu * (p2[2] - p1[2]);
            p[3] = 1.0f;

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
                for (auto vertex : triangle->pos) {
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