#pragma once
#ifndef _BUNDLE_ADJUSTMENT_HEADER
#define _BUNDLE_ADJUSTMENT_HEADER

#include "ceres/autodiff_cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"
#include "glog/logging.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <sstream>

#include "CharacteristicPoints.hpp"
#include "OptimizationProblem.hpp"
#include "PointCorrespondenceError.hpp"
#include "ReprojectionError.hpp"
#include "../Utils.hpp"
#include "../CaptureDevice.hpp"

namespace vc::optimization {
    class BundleAdjustment : virtual public OptimizationProblem {
    protected:
        bool needsRecalculation = true;

        const int num_translation_parameters = 3;
        const int num_rotation_parameters = 3;
        const int num_intrinsic_parameters = 4;
        const int num_distCoeff_parameters = 4;

        std::vector<std::vector<double>> translations;
        std::vector<std::vector<double>> rotations;
        std::vector<std::vector<double>> intrinsics;
        std::vector<std::vector<double>> distCoeffs;


        Eigen::Matrix4d getTransformation(int camera_index) {
            if (needsRecalculation) {
                calculateTransformations();
            }
            return transformations[camera_index];
        }

        void randomize() {
            Eigen::Vector3d randomTranslation = Eigen::Vector3d::Random();
            translations[1][0] += (double)randomTranslation[0];
            translations[1][1] += (double)randomTranslation[1];
            translations[1][2] += (double)randomTranslation[2];

            Eigen::Vector3d randomRotation = Eigen::Vector3d::Random();
            randomRotation.normalize();
            double angle = std::rand() % 360;
            randomRotation *= glm::radians(angle);
            rotations[1][0] += (double)randomRotation[0];
            rotations[1][1] += (double)randomRotation[1];
            rotations[1][2] += (double)randomRotation[2];

            needsRecalculation = true;
        }

        bool init(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines) {
            if (!OptimizationProblem::init(pipelines)) {
                return false;
            }

            for (int i = 0; i < pipelines.size(); i++) {
                std::vector<double> translation;
                for (int j = 0; j < num_translation_parameters; j++)
                {
                    translation.emplace_back(pipelines[i]->chArUco->translation[j]);
                }

                std::vector<double> rotation;
                for (int j = 0; j < num_rotation_parameters; j++)
                {
                    rotation.emplace_back(pipelines[i]->chArUco->rotation[j]);
                }

                translations[i] = (translation);
                rotations[i] = (rotation);
            }

            randomize();

            needsRecalculation = true;

            //calculateRelativeTransformations();
            return true;
        }
        Eigen::Matrix4d getRotationMatrix(int camera_index) {
            try {
                Eigen::Vector3d rotationVector(
                    rotations.at(camera_index).at(0),
                    rotations.at(camera_index).at(1),
                    rotations.at(camera_index).at(2)
                );
                return generateTransformationMatrix(0.0, 0.0, 0.0, rotationVector.norm(), rotationVector.normalized());
            }
            catch (std::out_of_range&) {
                return Eigen::Matrix4d::Identity();
            }
            catch (std::exception&) {
                return Eigen::Matrix4d::Identity();
            }
        }

        Eigen::Matrix4d getTranslationMatrix(int camera_index) {
            try {
                return generateTransformationMatrix(
                    translations.at(camera_index).at(0),
                    translations.at(camera_index).at(1),
                    translations.at(camera_index).at(2),
                    0.0, Eigen::Vector3d::Identity()
                );
            }
            catch (std::out_of_range&) {
                return Eigen::Matrix4d::Identity();
            }
            catch (std::exception&) {
                return Eigen::Matrix4d::Identity();
            }
        }

    public:
        void calculateTransformations() {
            for (int i = 0; i < translations.size(); i++) {
                transformations[i] = getTranslationMatrix(i) * getRotationMatrix(i);
            }

            needsRecalculation = false;
        }

        BundleAdjustment(bool verbose = false, bool withSleep = false) : OptimizationProblem(verbose, withSleep) {
            for (int i = 0; i < 4; i++)
            {
                translations.push_back(std::vector<double> { 0.0, 0.0, 0.0 });
                rotations.push_back(std::vector<double> { 0.0, 0.0, 0.0 });
                intrinsics.push_back(std::vector<double> { 0.0, 0.0, 0.0, 0.0 });
                distCoeffs.push_back(std::vector<double> { 0.0, 0.0, 0.0, 0.0 });
            }
        }

        void clear() {
            OptimizationProblem::clear();
            needsRecalculation = true;

            for (int i = 0; i < 4; i++)
            {
                translations[i] = (std::vector<double> { 0.0, 0.0, 0.0 });
                rotations[i] = (std::vector<double> { 0.0, 0.0, 0.0 });
                intrinsics[i] = (std::vector<double> { 0.0, 0.0, 0.0, 0.0 });
                distCoeffs[i] = (std::vector<double> { 0.0, 0.0, 0.0, 0.0 });
            }
        }
        
        void solveProblem(ceres::Problem* problem) {
            std::vector<Eigen::Matrix4d> initialTransformations = transformations;
            
            ceres::Solver::Options options;
            options.num_threads = 4;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = true;
            options.max_num_iterations = 500;
            options.update_state_every_iteration = true;
            //options.callbacks.emplace_back(new LoggingCallback(this));
            ceres::Solver::Summary summary;
            ceres::Solve(options, problem, &summary);
            std::cout << summary.FullReport() << "\n";
            
            calculateTransformations();

            std::cout << vc::utils::toString("Initial", initialTransformations);
            std::cout << vc::utils::toString("Final", transformations);

            std::cout << std::endl;
        }

        bool solvePointCorrespondenceError() {
            // Create residuals for each observation in the bundle adjustment problem. The
            // parameters for cameras and points are added automatically.
            ceres::Problem problem;
            for (int baseId = 0; baseId < characteristicPoints.size(); baseId++)
            {
                Eigen::Matrix4d inverseBaseTransformation = getTransformation(baseId).inverse();

                for (int relativeId = 1; relativeId < characteristicPoints.size(); relativeId++) {
                    if (baseId == relativeId) {
                        continue;
                    }

                    for (auto& relativeMarkerCorners : characteristicPoints[relativeId].markerCorners)
                    {
                        for (int cornerId = 0; cornerId < relativeMarkerCorners.second.size(); cornerId++)
                        {
                            if (characteristicPoints[baseId].markerCorners.count(relativeMarkerCorners.first) <= 0) {
                                continue;
                            }

                            auto baseFramePoint = characteristicPoints[baseId].markerCorners[relativeMarkerCorners.first][cornerId];
                            auto relativeFramePoint = relativeMarkerCorners.second[cornerId];

                            //std::cout << "base" << std::endl << baseFramePoint << std::endl;
                            //std::cout << "relative" << std::endl << relativeFramePoint << std::endl;

                            ceres::CostFunction* cost_function = PointCorrespondenceError::Create(
                                relativeMarkerCorners.first, cornerId,
                                relativeFramePoint,
                                baseFramePoint,
                                inverseBaseTransformation
                            );
                            problem.AddResidualBlock(cost_function, NULL,
                                translations[relativeId].data(),
                                rotations[relativeId].data()
                            );
                        }
                    }

                    for (auto& relativeCharucoCorners : characteristicPoints[relativeId].charucoCorners)
                    {
                        if (characteristicPoints[baseId].charucoCorners.count(relativeCharucoCorners.first) <= 0) {
                            continue;
                        }
                        ceres::CostFunction* cost_function = PointCorrespondenceError::Create(
                            relativeCharucoCorners.first, 0,
                            relativeCharucoCorners.second,
                            characteristicPoints[baseId].charucoCorners[relativeCharucoCorners.first],
                            inverseBaseTransformation
                        );
                        problem.AddResidualBlock(cost_function, NULL,
                            translations[relativeId].data(),
                            rotations[relativeId].data()
                        );
                    }
                }
            }

            solveProblem(&problem);
            return true;
        }

        bool vc::optimization::OptimizationProblem::specific_optimize() {
            if (!solvePointCorrespondenceError()) {
                return false;
            }

            return true;
        }
    };

    class MockBundleAdjustment : public BundleAdjustment, public MockOptimizationProblem {
    private:
        void setup() {
            for (int i = 0; i < transformations.size(); i++)
            {
                Eigen::Vector3d translation = transformations[i].block<3, 1>(0, 3);
                double angle = Eigen::AngleAxisd(transformations[i].block<3, 3>(0, 0)).angle();
                Eigen::Vector3d rotation = Eigen::AngleAxisd(transformations[i].block<3, 3>(0, 0)).axis().normalized();
                rotation *= angle;

                for (int j = 0; j < 3; j++)
                {
                    translations[i][j] = translation[j];
                    rotations[i][j] = rotation[j];
                }
            }
            needsRecalculation = true;
        }

    public:
        bool vc::optimization::OptimizationProblem::specific_optimize() {
            setupMock();
            setup();

            randomize();

            BundleAdjustment::specific_optimize();

            return true;
        }
    };
}
#endif // !_BUNDLE_ADJUSTMENT_HEADER