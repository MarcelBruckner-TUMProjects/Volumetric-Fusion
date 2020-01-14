#pragma once
#ifndef _MARCHING_CUBES_STRUCTS
#define _MARCHING_CUBES_STRUCTS

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "ceres/ceres.h"

namespace vc::fusion {
    class Triangle {
    public:
        Eigen::Vector3d p[3];
    };

    class GridCell {
    public:
        Eigen::Vector3d p[8];
        float val[8];
    };
}

#endif // !_MARCHING_CUBES_STRUCTS