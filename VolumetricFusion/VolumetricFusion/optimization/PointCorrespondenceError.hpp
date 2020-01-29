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
        PointCorrespondenceError(unsigned long long hash, ceres::Vector relative_frame_point, ceres::Vector base_frame_point)
            : hash(hash), relative_frame_point(relative_frame_point), base_frame_point(base_frame_point) {}
               
        std::string twoVectorsAside(ceres::Vector a, ceres::Vector b) const {
            std::stringstream ss;
            for (int i = 0; i < 4; i++)
            {
                ss << a[i] << " ---> " << b[i] << std::endl;
            }
            return ss.str();
        }

        template <typename T>
        Eigen::Matrix<T, 4, 4> buildTranslation(const T* const _translation) const {
            Eigen::Matrix<T, 4, 4> translation;
            translation <<
                T(1), T(0), T(0), _translation[0],
                T(0), T(1), T(0), _translation[1],
                T(0), T(0), T(1), _translation[2],
                T(0), T(0), T(0), T(1);
            return translation;
        }
        template <typename T>
        Eigen::Matrix<T, 4, 4> buildRotation(const T* const _rotation) const {
            // Rodriguez
            T* rot = new T[9];
            ceres::AngleAxisToRotationMatrix(_rotation, rot);

            Eigen::Matrix<T, 4, 4> relativeRotation;
            relativeRotation <<
                rot[0], rot[3], rot[6], T(0),
                rot[1], rot[4], rot[7], T(0),
                rot[2], rot[5], rot[8], T(0),
                T(0), T(0), T(0), T(1);
            return relativeRotation;
        }

        template <typename T>
        Eigen::Matrix<T, 4, 4> buildScale(const T* const _scale) const{
            Eigen::Matrix<T, 4, 4> relativeScale;
            relativeScale <<
                _scale[0], T(0), T(0), T(0),
                T(0), _scale[1], T(0), T(0),
                T(0), T(0), _scale[2], T(0),
                T(0), T(0), T(0), T(1);
            return relativeScale;
        }

        template <typename T>
        bool operator()(
            const T* const _relativeTranslation, const T* const _relativeRotation, const T* const _relativeScale,
            const T* const _baseTranslation, const T* const _baseRotation, const T* const _baseScale,
            T* residuals) const {
                        
            std::stringstream ss;

            ss << vc::utils::asHeader("Base --> Relative");
            ss << twoVectorsAside(base_frame_point, relative_frame_point);
            
            Eigen::Matrix<T, 4, 1> transformedPoint = Eigen::Matrix<T, 4, 1>(base_frame_point.cast<T>());
            ss << vc::utils::toString("b", transformedPoint);

            transformedPoint = Eigen::Matrix<T, 4, 1>(buildTranslation(_baseTranslation).inverse() * transformedPoint);
            ss << vc::utils::toString("Tb^-1 * b", transformedPoint);

            transformedPoint = Eigen::Matrix<T, 4, 1>(buildRotation(_baseRotation).inverse() * transformedPoint);
            ss << vc::utils::toString("Rb^-1 * (Tb^-1 * b)", transformedPoint);

            transformedPoint = Eigen::Matrix<T, 4, 1>(buildScale(_baseScale).inverse() * transformedPoint);
            ss << vc::utils::toString("Sb^-1 * (Rb^-1 * (Tb^-1 * b))", transformedPoint);

            transformedPoint = Eigen::Matrix<T, 4, 1>(buildScale(_relativeScale) * transformedPoint);
            ss << vc::utils::toString("Sr * (Sb^-1 * (Rb^-1 * (Tb^-1 * b)))", transformedPoint);

            transformedPoint = Eigen::Matrix<T, 4, 1>(buildRotation(_relativeRotation) * transformedPoint);
            ss << vc::utils::toString("Rr * (Sr * (Sb^-1 * (Rb^-1 * (Tb^-1 * b))))", transformedPoint);

            transformedPoint = Eigen::Matrix<T, 4, 1>(buildTranslation(_relativeTranslation) * transformedPoint);
            ss << vc::utils::toString("Tr * (Rr * (Sr * (Sb^-1 * (Rb^-1 * (Tb^-1 * b)))))", transformedPoint);

            // *********************************************************************************
            // Final cost evaluation
            Eigen::Matrix<T, 4, 1> error = Eigen::Matrix<T, 4, 1>(relative_frame_point - transformedPoint);

            for (int i = 0; i < 3; i++)
            {
                residuals[i] = error[i];
            }

            ss << vc::utils::toString("Residuals", error);

            std::cout << ss.str() << "************************************************************************************************************" << std::endl;

            return true;
        }

        unsigned long long hash;
        ceres::Vector relative_frame_point;
        ceres::Vector base_frame_point;

        // Factory to hide the construction of the CostFunction object from
        // the client code.
        static ceres::CostFunction* Create(unsigned long long hash, ceres::Vector relativeFramePoint, ceres::Vector baseFramePoint) {
            return (new ceres::AutoDiffCostFunction<PointCorrespondenceError, 3, 3, 3, 3, 3, 3, 3>(
                new PointCorrespondenceError(hash, relativeFramePoint, baseFramePoint)));
        }
    };
}

#endif