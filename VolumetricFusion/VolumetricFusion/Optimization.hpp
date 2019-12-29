#pragma once
#ifndef _OPTIMIZATION_HEADER
#define _OPTIMIZATION_HEADER

#include "ceres/autodiff_cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"
#include "glog/logging.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace vc::optimization {

    using namespace ceres;

    // 3 translation
    // 3 rotation
    // 4 intrinsics - fx, fy, ppx, ppy
    // 4 distortion - k1, k2, p1, p2

    struct SnavelyReprojectionError {
        SnavelyReprojectionError(cv::Point2f observed_pixel, double* observed_point)
            : observed_pixel(observed_pixel), observed_point(observed_point) {}

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
            const T* intrinsics, const T* const distortion, 
            const T* const point, T* residuals) const {

            // camera[0,1,2] are the angle-axis rotation.
            T p[3];
            ceres::AngleAxisRotatePoint(rotation, point, p);

            // camera[3,4,5] are the translation.
            p[0] += translation[0];
            p[1] += translation[1];
            p[2] += translation[2];

            // Compute the center of distortion. The sign change comes from
            // the camera model that Noah Snavely's Bundler assumes, whereby
            // the camera coordinate system has a negative z axis.
            // TODO Maybe coordinate correct
            T xd = -p[0] / p[2];
            T yd = -p[1] / p[2];

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
        static ceres::CostFunction* Create(cv::Point2f observed_pixel, double* observed_point) {
            return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 3, 3, 4, 4, 3>(
                new SnavelyReprojectionError(observed_pixel, observed_point)));
        }

        cv::Point2f observed_pixel;
        double* observed_point;
    };

    class BAProblem {
    private:
        double* translations;
        double* rotations;
        double* intrinsics;
        double* distCoeffs;
        std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines;

    public:
        BAProblem(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines) {
            this->pipelines = pipelines;
            translations = new double[3 * pipelines.size()];
            rotations = new double[3 * pipelines.size()];
            intrinsics = new double[4 * pipelines.size()];
            distCoeffs = new double[5 * pipelines.size()];
            
            for (int i = 0; i < pipelines.size(); i++) {
                for (int j = 0; j < 3; j++)
                {
                    translations[i * 3 + j] = pipelines[i]->chArUco->t[j];
                }
                for (int j = 0; j < 3; j++)
                {
                    rotations[i * 3 + j] = pipelines[i]->chArUco->r[j];
                }
                intrinsics[i * 4 + 0] = pipelines[i]->depth_camera->intrinsics.fx;
                intrinsics[i * 4 + 1] = pipelines[i]->depth_camera->intrinsics.fy;
                intrinsics[i * 4 + 2] = pipelines[i]->depth_camera->intrinsics.ppx;
                intrinsics[i * 4 + 3] = pipelines[i]->depth_camera->intrinsics.ppy;
                for (int j = 0; j < 5; j++)
                {
                    distCoeffs[i * 5 + j] = pipelines[i]->depth_camera->distCoeffs[j];
                }
            }

            solve();
        }

        ~BAProblem() {
            delete[] translations;
            delete[] rotations;
            delete[] intrinsics;
            delete[] distCoeffs;
        }
        
        void solve() {
            google::InitGoogleLogging("Bundle Adjustment");

            // Create residuals for each observation in the bundle adjustment problem. The
            // parameters for cameras and points are added automatically.
            ceres::Problem problem;

            for (int i = 0; i < pipelines.size(); i++) {
                auto pipe = pipelines[i];

                rs2::depth_frame depth_frame = pipe->data->filteredDepthFrames;
                int depth_width = depth_frame.as<rs2::video_frame>().get_width();
                int depth_height = depth_frame.as<rs2::video_frame>().get_height();

                rs2::frame color_frame = pipe->data->filteredColorFrames;
                int color_width = color_frame.as<rs2::video_frame>().get_width();
                int color_height = color_frame.as<rs2::video_frame>().get_height();

                float color2DepthWidth = 1.0f * depth_width / color_width;
                float color2DepthHeight = 1.0f * depth_height / color_height;

                for (auto markerCorners : pipe->chArUco->markerCorners) {
                    for (auto observation : markerCorners) {
                        int x = observation.x * color2DepthWidth;
                        int y = observation.y * color2DepthHeight;

                        float z = depth_frame.get_distance(x, y);

                        std::cout << "";


                    }
                }
            }


            //for (int i = 0; i < num_observations(); ++i) {
            //    // Each Residual block takes a point and a camera as input and outputs a 2
            //    // dimensional residual. Internally, the cost function stores the observed
            //    // image location and compares the reprojection against the observation.
            //    ceres::CostFunction* cost_function =
            //        SnavelyReprojectionError::Create(observations_[2 * i + 0],
            //            observations_[2 * i + 1]);
            //    problem.AddResidualBlock(cost_function,
            //        NULL /* squared loss */,
            //        mutable_camera_for_observation(i),
            //        mutable_point_for_observation(i));
            //}
            // Make Ceres automatically detect the bundle structure. Note that the
            // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
            // for standard bundle adjustment problems.
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_SCHUR;
            options.minimizer_progress_to_stdout = true;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.FullReport() << "\n";
        }
    };
}

#endif