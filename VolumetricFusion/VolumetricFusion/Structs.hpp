#pragma once
#ifndef _MARCHING_CUBES_STRUCTS
#define _MARCHING_CUBES_STRUCTS

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "ceres/ceres.h"
#include "Utils.hpp"

namespace vc::fusion {
    struct Vertex {
        glm::vec4 pos;
        glm::vec4 tsdf;
        glm::vec4 color;
    };

    struct Triangle {
        glm::vec4 pos0;
        glm::vec4 color0;
        glm::vec4 pos1;
        glm::vec4 color1;
        glm::vec4 pos2;
        glm::vec4 color2;
    };
    
    class GridCell {
    public:
        vc::fusion::Vertex verts[8];
    };
}

#endif // !_MARCHING_CUBES_STRUCTS