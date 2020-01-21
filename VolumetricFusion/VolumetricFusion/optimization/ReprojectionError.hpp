#pragma once

#ifndef _REPROJECTION_ERROR
#define _REPROJECTION_ERROR

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>
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
    struct ReprojectionError {
        ReprojectionError(cv::Point2f observed_pixel, glm::vec4 observed_point, double* intrinsics, double* distCoeffs)
            : observed_pixel(observed_pixel), observed_point(observed_point), intrinsics(intrinsics), distortion(distCoeffs) {}

        /// <summary>
        /// </summary>
        /// <param name="translation">The translation.</param>
        /// <param name="rotation">The rotation.</param>
        /// <param name="intrinsics">The intrinsics. (fx, fy, ppx, ppy)</param>
        /// <param name="distortion">The distortion. (k1, k2, p1, p2)</param>
        /// <param name="point">The point.</param>
        /// <param name="residuals">The residuals.</param>
        /// <returns></returns>
        template <typename T>
        bool operator()(const T* const translation, const T* const rotation,
            //const T* intrinsics, const T* const distortion, 
            T* residuals) const {

            T p[3];
            ceres::AngleAxisRotatePoint(rotation, new T[3]{ T(observed_point.x), T(observed_point.y), T(observed_point.z) }, p);

            p[0] += translation[0];
            p[1] += translation[1];
            p[2] += translation[2];

            T xd = p[0] / p[2];
            T yd = p[1] / p[2];

            T r_sq = xd * xd + yd * yd;
            T kappa = T(1.0f) + distortion[0] * r_sq + distortion[1] * r_sq * r_sq;

            T xdd = xd * kappa + T(2.0) * distortion[2] * xd * yd + distortion[3] * (r_sq + T(2.0) * xd * xd);
            T ydd = yd * kappa + T(2.0) * distortion[3] * xd * yd + distortion[2] * (r_sq + T(2.0) * yd * yd);

            T predicted_x = intrinsics[0] * xdd + intrinsics[2];
            T predicted_y = intrinsics[1] * ydd + intrinsics[3];

            // The error is the difference between the predicted and observed position.
            residuals[0] = predicted_x - T(observed_pixel.x);
            residuals[1] = predicted_y - T(observed_pixel.y);
            return true;
        }

        // Factory to hide the construction of the CostFunction object from
        // the client code.
        static ceres::CostFunction* Create(cv::Point2f observed_pixel, glm::vec4 observed_point, double* intrinsics, double* distCoeffs) {
            return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3>(
                new ReprojectionError(observed_pixel, observed_point, intrinsics, distCoeffs)));
        }

        cv::Point2f observed_pixel;
        glm::vec4 observed_point;
        double* intrinsics;
        double* distortion;
    };
}
#endif