#pragma once
#ifndef _MARCHING_CUBES_STRUCTS
#define _MARCHING_CUBES_STRUCTS

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace vc::fusion {
    class Triangle {
    public:
        glm::vec3 p[3];
    };

    class GridCell {
    public:
        glm::vec3 p[8];
        float val[8];
    };
}

#endif // !_MARCHING_CUBES_STRUCTS