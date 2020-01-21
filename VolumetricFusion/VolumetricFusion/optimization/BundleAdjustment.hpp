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

#include "Procrustes.hpp"
#include "CharacteristicPoints.hpp"
#include "OptimizationProblem.hpp"
#include "PointCorrespondenceError.hpp"
#include "ReprojectionError.hpp"
#include "ICP.hpp"
#include "../Utils.hpp"
#include "../CaptureDevice.hpp"

namespace vc::optimization {
    class BundleAdjustment : virtual public OptimizationProblem {
    protected:

        bool hasProcrustesInitialization = false;
        bool needsRecalculation = true;
        int iterationsSinceImprovement = 0;
        int maxIterationsSinceImprovement = 5;

        const int num_translation_parameters = 3;
        const int num_rotation_parameters = 3;
        const int num_scale_parameters = 3;
        const int num_intrinsic_parameters = 4;
        const int num_distCoeff_parameters = 4;

        std::vector<std::vector<double>> translations;
        std::vector<std::vector<double>> rotations;
        std::vector<std::vector<double>> scales;

        std::vector<std::vector<double>> intrinsics;
        std::vector<std::vector<double>> distCoeffs;


        Eigen::Matrix4d getTransformation(int camera_index) {
            if (needsRecalculation) {
                calculateTransformations();
            }
            return OptimizationProblem::getCurrentTransformation(camera_index);
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

            //for (int i = 0; i < pipelines.size(); i++) {
            //    std::vector<double> translation;
            //    for (int j = 0; j < num_translation_parameters; j++)
            //    {
            //        translation.emplace_back(pipelines[i]->chArUco->translation[j]);
            //    }

            //    std::vector<double> rotation;
            //    for (int j = 0; j < num_rotation_parameters; j++)
            //    {
            //        rotation.emplace_back(pipelines[i]->chArUco->rotation[j]);
            //    }

            //    translations[i] = (translation);
            //    rotations[i] = (rotation);
            //    scales[i] = std::vector<double>{ 1.0, 1.0, 1.0 };
            //}

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

        Eigen::Matrix4d getScaleMatrix(int camera_index) {
            try {
                return generateScaleMatrix(
                    scales.at(camera_index).at(0),
                    scales.at(camera_index).at(1),
                    scales.at(camera_index).at(2)
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
                currentTranslations[i] = getTranslationMatrix(i);
                currentRotations[i] = getRotationMatrix(i);
                currentScales[i] = getScaleMatrix(i);
            }

            needsRecalculation = false;
        }
        
        BundleAdjustment(bool verbose = false, bool withSleep = false) : OptimizationProblem(verbose, withSleep) {
            for (int i = 0; i < 4; i++)
            {
                translations.push_back(std::vector<double> { 0.0, 0.0, 0.0 });
                rotations.push_back(std::vector<double> { 0.0, 2 * M_PI, 0.0 });
                scales.push_back(std::vector<double> {1.0, 1.0, 1.0  });

                intrinsics.push_back(std::vector<double> { 0.0, 0.0, 0.0, 0.0 });
                distCoeffs.push_back(std::vector<double> { 0.0, 0.0, 0.0, 0.0 });
            }
        }

        void clear() {
            OptimizationProblem::clear();
            needsRecalculation = true;
        }
        
        void solveProblem(ceres::Problem* problem) {
            std::vector<Eigen::Matrix4d> initialTransformations = bestTransformations;
            
            ceres::Solver::Options options;
            options.num_threads = 8;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = verbose;
            options.max_num_iterations = 500;
            options.update_state_every_iteration = true;
            //options.callbacks.emplace_back(new LoggingCallback(this));
            ceres::Solver::Summary summary;
            ceres::Solve(options, problem, &summary);
            calculateTransformations();

            //if (verbose) {
                std::cout << summary.FullReport() << "\n";
                std::cout << vc::utils::toString("Initial", initialTransformations);
                std::cout << vc::utils::toString("Final", bestTransformations);

                std::cout << std::endl;
            //}
        }

        bool solvePointCorrespondenceError() {

			std::cout << "Bundle Adjustment" << std::endl;

            // Create residuals for each observation in the bundle adjustment problem. The
            // parameters for cameras and points are added automatically.
            ceres::Problem problem;
            //for (int baseId = 0; baseId < characteristicPoints.size(); baseId++)
            {
                int baseId = 0;
                ACharacteristicPoints baseFramePoints = characteristicPoints[baseId];
                Eigen::Matrix4d inverseBaseTransformation = getTransformation(baseId).inverse();

                for (int relativeId = 1; relativeId < characteristicPoints.size(); relativeId++) {
                    if (baseId == relativeId) {
                        continue;
                    }

                    ACharacteristicPoints relativeFramePoints = characteristicPoints[relativeId];
                    std::vector<unsigned long long> matchingHashes = vc::utils::findOverlap(baseFramePoints.getHashes(verbose), relativeFramePoints.getHashes(verbose));

                    //if (matchingHashes.size() <= 4) {
                    //    std::cerr << "At least 5 points are needed for Procrustes. Provided: " << matchingHashes.size() << std::endl;
                    //    return Eigen::Matrix4d::Identity();
                    //}

                    auto& filteredBaseFramePoints = baseFramePoints.getFilteredPoints(matchingHashes, verbose);
                    auto& filteredRelativeFramePoints = relativeFramePoints.getFilteredPoints(matchingHashes, verbose);

                    for (auto& hash : matchingHashes)
                    {
                        auto relativePoint = filteredRelativeFramePoints[hash];
                        auto basePoint = filteredBaseFramePoints[hash];

                        bool valid = true;
                        for (int i = 0; i < 3; i++)
                        {
                            if (std::abs(relativePoint[i]) > 10e5) {
                                valid = false;
                                break;
                            }
                        }
                        if (!valid) {
                            continue;
                        }

                        for (int i = 0; i < 3; i++)
                        {
                            if (std::abs(basePoint[i]) > 10e5) {
                                valid = false;
                                break;
                            }
                        }

                        if (!valid) {
                            continue;
                        }

                        ceres::CostFunction* cost_function = PointCorrespondenceError::Create(
                            hash, 
                            relativePoint,
                            basePoint,
                            inverseBaseTransformation
                        );
                        problem.AddResidualBlock(cost_function, NULL,
                            translations[relativeId].data(),
                            rotations[relativeId].data(),
                            scales[relativeId].data()
                        );
                    }
                }
            }

            solveProblem(&problem);
            return true;
        }

        void initializeWithProcrustes() {
            vc::optimization::Procrustes procrustes = vc::optimization::Procrustes(verbose);
            procrustes.characteristicPoints = characteristicPoints;
            if (procrustes.optimizeOnPoints()) {
                bestTransformations = procrustes.bestTransformations;
                currentRotations = procrustes.currentRotations;
                currentTranslations = procrustes.currentTranslations;
                currentScales = procrustes.currentScales;
                hasProcrustesInitialization = true;
                setup();
            }
            else {
                hasProcrustesInitialization = false;
            }
        }

		bool solveICP() {
			
			std::vector<Eigen::Matrix4d> initialTransformations = bestTransformations;

			int baseId = 0;

			for (int relativeId = 1; relativeId < characteristicPoints.size(); relativeId++) {
				//getCurrentRelativeTransformation
				ICP icp = vc::optimization::ICP();
				bestTransformations[relativeId] = getBestRelativeTransformation(baseId, relativeId) * icp.estimatePose(characteristicPoints[relativeId], characteristicPoints[baseId], getBestRelativeTransformation(relativeId, baseId));
			}

			std::cout << vc::utils::toString("Initial", initialTransformations);
			std::cout << vc::utils::toString("Final", bestTransformations);

			std::cout << std::endl;

			return true;
		}

        void setup() {
            for (int i = 0; i < 4; i++)
            {
                Eigen::Vector3d translation = currentTranslations[i].block<3, 1>(0, 3);
                double angle = Eigen::AngleAxisd(currentRotations[i].block<3, 3>(0, 0)).angle();
                Eigen::Vector3d rotation = Eigen::AngleAxisd(currentRotations[i].block<3, 3>(0, 0)).axis().normalized();
                rotation *= angle;
                Eigen::Vector3d scale = currentScales[i].diagonal().block<3, 1>(0, 0);

                for (int j = 0; j < 3; j++)
                {
                    translations[i][j] = translation[j];
                    rotations[i][j] = rotation[j];
                    scales[i][j] = scale[j];
                }
            }
            calculateTransformations();
        }

        bool vc::optimization::OptimizationProblem::specific_optimize() {
            if (!hasProcrustesInitialization) {
                initializeWithProcrustes();
                return false;
            }

            if (!solvePointCorrespondenceError() /*|| !solveICP()*/) {
                return false;
            }

            return true;
        }
    };

    class MockBundleAdjustment : public BundleAdjustment, public MockOptimizationProblem {
    private:
        void setupMock() {
            MockOptimizationProblem::setupMock();
            setup();
        }
        
    public:
        bool vc::optimization::OptimizationProblem::specific_optimize() {
            setupMock();

            BundleAdjustment::specific_optimize();

            return true;
        }
    };
}
#endif // !_BUNDLE_ADJUSTMENT_HEADER