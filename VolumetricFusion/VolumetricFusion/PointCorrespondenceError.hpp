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
        PointCorrespondenceError(int id, int j, ceres::Vector relative_frame_point, ceres::Vector base_frame_point, ceres::Matrix inverse_base_rotation, ceres::Matrix inverse_base_translation)
            : id(id), j(j), relative_frame_point(relative_frame_point), base_frame_point(base_frame_point), inverse_base_rotation(inverse_base_rotation), inverse_base_translation(inverse_base_translation) {}
               
        std::string asHeader(std::string name) const {
            std::stringstream ss;
            ss << "*************************************  ";
            ss << name;
            ss << "  *************************************";
            ss << std::endl;
            return ss.str();
        }

        std::string twoVectorsAside(ceres::Vector a, ceres::Vector b) const {
            std::stringstream ss;
            for (int i = 0; i < 4; i++)
            {
                ss << a[i] << " ---> " << b[i] << std::endl;
            }
            return ss.str();
        }

        template <typename T>
        bool operator()(const T* const markerToRelativeTranslation, const T* const markerToRelativeRotation,
            //const T* intrinsics, const T* const distortion, 
            T* residuals) const {
                        
            std::stringstream ss;

            ss << asHeader("Base --> Relative");
            ss << twoVectorsAside(base_frame_point, relative_frame_point);
            
            Eigen::Matrix<T, 4, 1> transformedPoint = Eigen::Matrix<T, 4, 1>(base_frame_point.cast<T>());
            ss << asHeader("b") << transformedPoint << std::endl;

            transformedPoint = Eigen::Matrix<T, 4, 1>(inverse_base_translation.cast<T>() * transformedPoint);
            ss << asHeader("T0^-1 * b") << transformedPoint << std::endl;

            transformedPoint = Eigen::Matrix<T, 4, 1>(inverse_base_rotation.cast<T>() * transformedPoint);
            ss << asHeader("R0^-1 * (T0^-1 * b)") << transformedPoint << std::endl;

            // Rodriguez
            T* rot = new T[9];
            ceres::AngleAxisToRotationMatrix(markerToRelativeRotation, rot);                        

            Eigen::Matrix<T, 4, 4> relativeRotation;
            relativeRotation <<
                rot[0], rot[3], rot[6], T(0),
                rot[1], rot[4], rot[7], T(0),
                rot[2], rot[5], rot[8], T(0),
                T(0), T(0), T(0), T(1);

            ss << asHeader("R1") << relativeRotation << std::endl;

            transformedPoint = Eigen::Matrix<T, 4, 1>(relativeRotation * transformedPoint);
            ss << asHeader("R1 * (R0^-1 * (T0^-1 * b))") << transformedPoint << std::endl;

            Eigen::Matrix<T, 4, 4> relativeTranslation;
            relativeTranslation <<
                T(1), T(0), T(0), markerToRelativeTranslation[0],
                T(0), T(1), T(0), markerToRelativeTranslation[1],
                T(0), T(0), T(1), markerToRelativeTranslation[2],
                T(0), T(0), T(0), T(1);

            ss << asHeader("T1") << relativeTranslation << std::endl;

            transformedPoint = Eigen::Matrix<T, 4, 1>(relativeTranslation * transformedPoint);
            ss << asHeader("T1 * (R1 * (R0^-1 * (T0^-1 * b)))") << transformedPoint << std::endl;
            
            // *********************************************************************************
            // Final cost evaluation
            Eigen::Matrix<T, 4, 1> error = Eigen::Matrix<T, 4, 1>(relative_frame_point - transformedPoint);

            for (int i = 0; i < 3; i++)
            {
                residuals[i] = error[i];
            }

            ss << asHeader("Residuals") << error << std::endl;

            //std::cout << ss.str() << "************************************************************************************************************" << std::endl;

            return true;
        }

        int j;
        int id;
        ceres::Vector relative_frame_point;
        ceres::Vector base_frame_point;
        ceres::Matrix inverse_base_rotation;
        ceres::Matrix inverse_base_translation;

        // Factory to hide the construction of the CostFunction object from
        // the client code.
        static ceres::CostFunction* Create(int id, int j, ceres::Vector relativeFramePoint, ceres::Vector baseFramePoint, ceres::Matrix inverse_base_rotation, ceres::Matrix inverse_base_transformation) {
            return (new ceres::AutoDiffCostFunction<PointCorrespondenceError, 3, 3, 3>(
                new PointCorrespondenceError(id, j, relativeFramePoint, baseFramePoint, inverse_base_rotation, inverse_base_transformation)));
        }
    };
}

#endif