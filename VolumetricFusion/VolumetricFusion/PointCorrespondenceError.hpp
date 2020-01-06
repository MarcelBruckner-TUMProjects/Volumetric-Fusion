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
        PointCorrespondenceError(int id, int j, glm::vec4 relative_frame_point, glm::vec4 base_frame_point, glm::f32* baseTransformation)
            : id(id), j(j), relative_frame_point(relative_frame_point), base_frame_point(base_frame_point), baseTransformation(baseTransformation) {}

        template <typename T>
        bool operator()(const T* const markerToRelativeTranslation, const T* const markerToRelativeRotation,
            //const T* intrinsics, const T* const distortion, 
            T* residuals) const {

            std::stringstream ss;

            ss << "**************************************************" << std::endl;
            T point_baseFrame[3];
            point_baseFrame[0] = T(base_frame_point.x);
            point_baseFrame[1] = T(base_frame_point.y);
            point_baseFrame[2] = T(base_frame_point.z);
            ss << base_frame_point.x << ", " << base_frame_point.y << ", " << base_frame_point.z << " --> " << relative_frame_point.x << ", " << relative_frame_point.y << ", " << relative_frame_point.z << std::endl;

            // *********************************************************************************
            // baseTransformation * (markerToRelativeRotation) * glm::inverse(markerToRelativeTranslation) * point

            // baseTransformation
            T inverseBaseTransformation[16];
            for (int i = 0; i < 16; i++)
            {
                inverseBaseTransformation[i] = T(baseTransformation[i]);
            }
            T point_noFrame[3];
            // baseTransformation * (markerToRelativeRotation) * glm::inverse(markerToRelativeTranslation) * point
            for (int i = 0; i < 3; i++)
            {
                point_noFrame[i] = point_baseFrame[i] + inverseBaseTransformation[i + 12];
            }
            ss << "T_0^-1: " << point_noFrame[0] << ", " << point_noFrame[1] << ", " << point_noFrame[2] << std::endl;

            for (int i = 0; i < 3; i++)
            {
                point_noFrame[i] = inverseBaseTransformation[i] * point_noFrame[0] + inverseBaseTransformation[i + 4] * point_noFrame[1] + inverseBaseTransformation[i + 8] * point_noFrame[2];
            }
            ss << "R_0^-1: " << point_noFrame[0] << ", " << point_noFrame[1] << ", " << point_noFrame[2] << std::endl;


            // *********************************************************************************
            // (markerToRelativeRotation) * glm::inverse(markerToRelativeTranslation) * point

            // markerToRelativeRotation
            T m2rR[3];
            for (int i = 0; i < 3; i++) {
                m2rR[i] = markerToRelativeRotation[i];
            }

            T point_relativeFrame[3];
            bool rowMajor = false;
            if (rowMajor) {
                // Row major
                T rot[9];
                // Rodriguez
                ceres::AngleAxisToRotationMatrix(m2rR, rot);

                // (markerToRelativeRotation) * glm::inverse(markerToRelativeTranslation) * point
                for (int i = 0; i < 3; i++) {
                    point_relativeFrame[i] = rot[i * 3 + 0] * point_noFrame[0] + rot[i * 3 + 1] * point_noFrame[1] + rot[i * 3 + 2] * point_noFrame[2];
                }
                ss << "R_1: " << point_relativeFrame[0] << ", " << point_relativeFrame[1] << ", " << point_relativeFrame[2] << std::endl;
            }
            else {
                // Row major
                T rot[9];
                // Rodriguez
                ceres::AngleAxisToRotationMatrix(m2rR, rot);

                // (markerToRelativeRotation) * glm::inverse(markerToRelativeTranslation) * point
                for (int i = 0; i < 3; i++) {
                    point_relativeFrame[i] = rot[i] * point_noFrame[0] + rot[i + 3] * point_noFrame[1] + rot[i + 6] * point_noFrame[2];
                }
                ss << "R_1: " << point_relativeFrame[0] << ", " << point_relativeFrame[1] << ", " << point_relativeFrame[2] << std::endl;

            }
            // *********************************************************************************
            // glm::inverse(markerToRelativeTranslation) * point

            // markerToRelativeTranslation
            T m2rT[3];

            for (int i = 0; i < 3; i++) {
                m2rT[i] = markerToRelativeTranslation[i];
                // glm::inverse(markerToRelativeTranslation) * point
                point_relativeFrame[i] += m2rT[i];
            }
            ss << "T_1: " << point_relativeFrame[0] << ", " << point_relativeFrame[1] << ", " << point_relativeFrame[2] << std::endl;

            // *********************************************************************************
            // Final cost evaluation
            T expected[3];
            expected[0] = T(relative_frame_point.x);
            expected[1] = T(relative_frame_point.y);
            expected[2] = T(relative_frame_point.z);

            ss << "Residuals:" << std::endl;

            for (int i = 0; i < 3; i++) {
                residuals[i] = point_relativeFrame[i] - expected[i];
                ss << residuals[i] << std::endl;
            }
            ss << "**************************************************" << std::endl;

            /*  std::stringstream ss;
              ss << "Residuals (" << id << ", " << j << "): " << residuals[0] << ", " << residuals[1] << ", " << residuals[2] << std::endl;
              ss << ss.str();*/

            if (base_frame_point.x == 1.0f && base_frame_point.y == 1.0f && base_frame_point.z == 1.0f) {
                ss << "";
            }
            //std::cout << ss.str();
            return true;
        }

        int j;
        int id;
        glm::vec4 relative_frame_point;
        glm::vec4 base_frame_point;
        glm::f32* baseTransformation;

        // Factory to hide the construction of the CostFunction object from
        // the client code.
        static ceres::CostFunction* Create(int id, int j, glm::vec4 relativeFramePoint, glm::vec4 baseFramePoint, glm::f32* baseTransformation) {
            return (new ceres::AutoDiffCostFunction<PointCorrespondenceError, 3, 3, 3>(
                new PointCorrespondenceError(id, j, relativeFramePoint, baseFramePoint, baseTransformation)));
        }
    };
}

#endif