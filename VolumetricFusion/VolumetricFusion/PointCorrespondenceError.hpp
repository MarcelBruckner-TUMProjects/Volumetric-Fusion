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
    
    /// <summary>
    /// Calculates the same relative transformation shader.
    /// --> baseTransformation * (markerToRelativeRotation) * glm::inverse(markerToRelativeTranslation) * point
    /// baseTransformation is baseToMarkerTranslation * glm::inverse(baseToMarkerRotation)
    ///
    /// @see Optimization.hpp::calculateRelativeTransformation(...)
    /// @see Optimization.hpp::solvePointCorrespondenceError(...)
    /// @see pointcloud_new.vert
    ///
    /// </summary>
    struct PointCorrespondenceError {
        PointCorrespondenceError(int id, int j, glm::vec4 observed_point, glm::vec4 expected_point, glm::f32* baseTransformation)
            : id(id), j(j), observed_point(observed_point), expected_point(expected_point), baseTransformation(baseTransformation) {}

        template <typename T>
        bool operator()(const T* const markerToRelativeTranslation, const T* const markerToRelativeRotation,
            //const T* intrinsics, const T* const distortion, 
            T* residuals) const {

            T point[3];
            point[0] = T(observed_point.x);
            point[1] = T(observed_point.y);
            point[2] = T(observed_point.z);

            // *********************************************************************************
            // glm::inverse(markerToRelativeTranslation) * point
                
            // markerToRelativeTranslation
            T m2rT[3];

            for (int i = 0; i < 3; i++) {
                m2rT[i] = markerToRelativeTranslation[i];
                // glm::inverse(markerToRelativeTranslation) * point
                point[i] -= m2rT[i];
            }

            // *********************************************************************************
            // (markerToRelativeRotation) * glm::inverse(markerToRelativeTranslation) * point

            // markerToRelativeRotation
            T m2rR[3];
            for (int i = 0; i < 3; i++) {
                m2rR[i] = markerToRelativeRotation[i];
            }

            // Row major
            T rot[9];
            // Rodriguez
            ceres::AngleAxisToRotationMatrix(m2rR, rot);

            T point_rotated[3];
            // (markerToRelativeRotation) * glm::inverse(markerToRelativeTranslation) * point
            for (int i = 0; i < 3; i++) {
                //p_rot[i] = p[i];// rot[i] * p[0] + rot[i + 3] * p[1] + rot[i + 6] * p[2];
                point_rotated[i] = rot[i * 3 + 0] * point[0] + rot[i * 3 + 1] * point[1] + rot[i * 3 + 2] * point[2];
            }

            // *********************************************************************************
            // baseTransformation * (markerToRelativeRotation) * glm::inverse(markerToRelativeTranslation) * point

            // baseTransformation
            T base[16];
            for (int i = 0; i < 16; i++)
            {
                base[i] = T(baseTransformation[i]);
            }

            T point_transformed[3];
            // baseTransformation * (markerToRelativeRotation) * glm::inverse(markerToRelativeTranslation) * point
            for (int i = 0; i < 3; i++)
            {
                point_transformed[i] = base[i] * point_rotated[0] + base[i + 4] * point_rotated[1] + base[i + 8] * point_rotated[2] + base[i + 12];
            }

            // *********************************************************************************
            // Final cost evaluation
            T expected[3];
            expected[0] = T(expected_point.x);
            expected[1] = T(expected_point.y);
            expected[2] = T(expected_point.z);

            for (int i = 0; i < 3; i++) {
                residuals[i] = point_transformed[i] - expected[i];
            }

            /*  std::stringstream ss;
              ss << "Residuals (" << id << ", " << j << "): " << residuals[0] << ", " << residuals[1] << ", " << residuals[2] << std::endl;
              std::cout << ss.str();*/

            return true;
        }

        int j;
        int id;
        glm::vec4 observed_point;
        glm::vec4 expected_point;
        glm::f32* baseTransformation;

        // Factory to hide the construction of the CostFunction object from
        // the client code.
        static ceres::CostFunction* Create(int id, int j, glm::vec4 observed_point, glm::vec4 expected_point, glm::f32* baseTransformation) {
            return (new ceres::AutoDiffCostFunction<PointCorrespondenceError, 3, 3, 3>(
                new PointCorrespondenceError(id, j, observed_point, expected_point, baseTransformation)));
        }
    };
}

#endif