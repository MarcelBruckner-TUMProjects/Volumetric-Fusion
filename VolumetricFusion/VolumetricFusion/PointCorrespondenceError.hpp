#pragma once

#ifndef _POINT_CORRESPONDENCE_ERROR
#define _POINT_CORRESPONDENCE_ERROR

#include "ceres/autodiff_cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"
#include "glog/logging.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace vc::optimization {

    using namespace ceres;

    struct PointCorrespondenceError {
        PointCorrespondenceError(glm::vec4 observed_point, glm::vec4 expected_point, glm::f32* baseTransformation)
            : observed_point(observed_point), expected_point(expected_point), baseTransformation(baseTransformation) {}

        template <typename T>
        bool operator()(const T* const markerToRelativeTranslation, const T* const markerToRelativeRotation,
            //const T* intrinsics, const T* const distortion, 
            T* residuals) const {

            // Column major
            T rot[9];
            ceres::AngleAxisToRotationMatrix(markerToRelativeRotation, rot);

            //baseTransformation* (markerToRelativeRotation)*glm::inverse(markerToRelativeTranslation) //######################################################################

            T p[3];
            p[0] = T(observed_point.x);
            p[1] = T(observed_point.y);
            p[2] = T(observed_point.z);

            T m2rT[3];

            for (int i = 0; i < 3; i++) {
                m2rT[i] = markerToRelativeTranslation[i];
                p[i] -= m2rT[i];
            }

            T p_rot[3];
            for (int i = 0; i < 3; i++) {
                //p_rot[i] = p[i];// rot[i] * p[0] + rot[i + 3] * p[1] + rot[i + 6] * p[2];
                p_rot[i] = rot[i] * p[0] + rot[i + 3] * p[1] + rot[i + 6] * p[2];
            }

            T base[16];
            for (int i = 0; i < 16; i++)
            {
                base[i] = T(baseTransformation[i]);
            }
            T p_base[3];
            for (int i = 0; i < 3; i++)
            {
                p_base[i] = base[i] * p_rot[0] + base[i + 4] * p_rot[1] + base[i + 8] * p_rot[2] + base[i + 12];
            }

            T expected[3];
            expected[0] = T(expected_point.x);
            expected[1] = T(expected_point.y);
            expected[2] = T(expected_point.z);

            for (int i = 0; i < 3; i++) {
                residuals[i] = p_base[i] - expected[i];
            }

            //std::stringstream ss;
            //ss << "Residuals: " << residuals[0] << ", " << residuals[1] << ", " << residuals[2] << std::endl;
            //std::cout << ss.str();

            return true;
        }

        glm::vec4 observed_point;
        glm::vec4 expected_point;
        glm::f32* baseTransformation;

        // Factory to hide the construction of the CostFunction object from
        // the client code.
        static ceres::CostFunction* Create(glm::vec4 observed_point, glm::vec4 expected_point, glm::f32* baseTransformation) {
            return (new ceres::AutoDiffCostFunction<PointCorrespondenceError, 3, 3, 3>(
                new PointCorrespondenceError(observed_point, expected_point, baseTransformation)));
        }
    };
}

#endif