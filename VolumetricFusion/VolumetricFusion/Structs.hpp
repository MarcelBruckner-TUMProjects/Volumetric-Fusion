#pragma once
#ifndef _MARCHING_CUBES_STRUCTS
#define _MARCHING_CUBES_STRUCTS

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "ceres/ceres.h"
#include "Utils.hpp"

namespace vc::fusion {
    class Triangle {
    public:
        std::vector<Eigen::Vector3d> vertices;

        Triangle(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d c) {
            vertices.push_back(a);
            vertices.push_back(b);
            vertices.push_back(c);
        }

        operator bool () {
            return vc::utils::isValid(vertices[0]) && vc::utils::isValid(vertices[1]) && vc::utils::isValid(vertices[2]);
        }
    };

    class GridCell {
    public:
        Eigen::Vector3d corners[8];
        float tsdfs[8];
    };
}

#endif // !_MARCHING_CUBES_STRUCTS