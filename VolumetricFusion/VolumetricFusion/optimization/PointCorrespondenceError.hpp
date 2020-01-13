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
#include "../Utils.hpp"

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
        PointCorrespondenceError(unsigned long long hash, ceres::Vector relative_frame_point, ceres::Vector base_frame_point, ceres::Matrix inverse_base_transformation)
            : hash(hash), relative_frame_point(relative_frame_point), base_frame_point(base_frame_point), inverse_base_transformation(inverse_base_transformation) {}
               
        std::string twoVectorsAside(ceres::Vector a, ceres::Vector b) const {
            std::stringstream ss;
            for (int i = 0; i < 4; i++)
            {
                ss << a[i] << " ---> " << b[i] << std::endl;
            }
            return ss.str();
        }

        template <typename T>
        bool operator()(const T* const markerToRelativeTranslation, const T* const markerToRelativeRotation, const T* const markerToRelativeScale,
            //const T* intrinsics, const T* const distortion, 
            T* residuals) const {
                        
            std::stringstream ss;

            ss << vc::utils::asHeader("Base --> Relative");
            ss << twoVectorsAside(base_frame_point, relative_frame_point);
            
            Eigen::Matrix<T, 4, 1> transformedPoint = Eigen::Matrix<T, 4, 1>(base_frame_point.cast<T>());
            ss << vc::utils::toString("b", transformedPoint);

            //transformedPoint = Eigen::Matrix<T, 4, 1>(inverse_base_translation.cast<T>() * transformedPoint);
            //ss << vc::utils::asHeader("T0^-1 * b") << transformedPoint << std::endl;

            transformedPoint = Eigen::Matrix<T, 4, 1>(inverse_base_transformation.cast<T>() * transformedPoint);
            ss << vc::utils::toString("S0^-1 * (R0^-1 * (T0^-1 * b))", transformedPoint);
            
            Eigen::Matrix<T, 4, 4> relativeScale;
            relativeScale <<
                markerToRelativeScale[0], T(0), T(0), T(0),
                T(0), markerToRelativeScale[1], T(0), T(0),
                T(0), T(0), markerToRelativeScale[2], T(0),
                T(0), T(0), T(0), T(1);

            ss << vc::utils::toString("S1", relativeScale);

            transformedPoint = Eigen::Matrix<T, 4, 1>(relativeScale * transformedPoint);
            ss << vc::utils::toString("S1 * (S0^-1 * (R0^-1 * (T0^-1 * b)))", transformedPoint);

            // Rodriguez
            T* rot = new T[9];
            ceres::AngleAxisToRotationMatrix(markerToRelativeRotation, rot);

            Eigen::Matrix<T, 4, 4> relativeRotation;
            relativeRotation <<
                rot[0], rot[3], rot[6], T(0),
                rot[1], rot[4], rot[7], T(0),
                rot[2], rot[5], rot[8], T(0),
                T(0), T(0), T(0), T(1);

            ss << vc::utils::toString("R1", relativeRotation);

            transformedPoint = Eigen::Matrix<T, 4, 1>(relativeRotation * transformedPoint);
            ss << vc::utils::toString("R1 * (S1 * (S0^-1 * (R0^-1 * (T0^-1 * b))))", transformedPoint);

            Eigen::Matrix<T, 4, 4> relativeTranslation;
            relativeTranslation <<
                T(1), T(0), T(0), markerToRelativeTranslation[0],
                T(0), T(1), T(0), markerToRelativeTranslation[1],
                T(0), T(0), T(1), markerToRelativeTranslation[2],
                T(0), T(0), T(0), T(1);

            ss << vc::utils::toString("T1", relativeTranslation);

            transformedPoint = Eigen::Matrix<T, 4, 1>(relativeTranslation * transformedPoint);
            ss << vc::utils::toString("T1 * (R1 * (S1 * (S0^-1 * (R0^-1 * (T0^-1 * b)))))", transformedPoint);
            
            // *********************************************************************************
            // Final cost evaluation
            Eigen::Matrix<T, 4, 1> error = Eigen::Matrix<T, 4, 1>(relative_frame_point - transformedPoint);

            for (int i = 0; i < 3; i++)
            {
                residuals[i] = error[i];
            }

            ss << vc::utils::toString("Residuals", error);

            //std::cout << ss.str() << "************************************************************************************************************" << std::endl;

            return true;
        }

        unsigned long long hash;
        ceres::Vector relative_frame_point;
        ceres::Vector base_frame_point;
        ceres::Matrix inverse_base_transformation;
        ceres::Matrix inverse_base_translation;

        // Factory to hide the construction of the CostFunction object from
        // the client code.
        static ceres::CostFunction* Create(unsigned long long hash, ceres::Vector relativeFramePoint, ceres::Vector baseFramePoint, Eigen::Matrix4d inverseBaseTransformation) {
            return (new ceres::AutoDiffCostFunction<PointCorrespondenceError, 3, 3, 3, 3>(
                new PointCorrespondenceError(hash, relativeFramePoint, baseFramePoint, inverseBaseTransformation)));
        }
    };
}

#endif