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
    class ABundleAdjustment : public vc::optimization::OptimizationProblem {
    public:
        void solveProblem(ceres::Problem* problem) {
            auto initialTranslations = translations;
            auto initialRotations = rotations;

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

            std::cout << vc::utils::toString("Initial", initialTranslations, initialRotations);
            std::cout << vc::utils::toString("Final", translations, rotations);

            std::cout << std::endl;
        }

        bool solvePointCorrespondenceError(std::vector<ACharacteristicPoints> characteristicPoints) {
            // Create residuals for each observation in the bundle adjustment problem. The
            // parameters for cameras and points are added automatically.
            ceres::Problem problem;
            for (int baseId = 0; baseId < characteristicPoints.size(); baseId++)
            {
                glm::mat4 baseToMarkerTranslation = getTranslationMatrix(baseId);
                glm::mat4 baseToMarkerRotation = getRotationMatrix(baseId);
                auto inverseBaseToMarkerTranslation = glmToCeresMatrix(glm::inverse(baseToMarkerTranslation));
                auto inverseBaseToMarkerRotation = glmToCeresMatrix(glm::inverse(baseToMarkerRotation));

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
                                inverseBaseToMarkerRotation, inverseBaseToMarkerTranslation
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
                            inverseBaseToMarkerRotation, inverseBaseToMarkerTranslation
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
    };

    class MockBundleAdjustment : public ABundleAdjustment {
    public:
        bool vc::optimization::OptimizationProblem::specific_optimize(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines) {
            glm::vec3 baseTranslationVector = glm::vec3(0.0f, 0.0f, 1.0f);
            translations[0] = (std::vector<double>{ baseTranslationVector.x, baseTranslationVector.y, baseTranslationVector.z });
            glm::mat4 baseTranslation = getTranslationMatrix(0);

            glm::vec3 relativeTranslationVector = glm::vec3(0.0f, 0.0f, 3.0f);
            translations[1] = (std::vector<double>{ relativeTranslationVector.x, relativeTranslationVector.y, relativeTranslationVector.z });
            glm::mat4 relativeTranslation = getTranslationMatrix(1);
            translations[1][0] -= 3;
            translations[1][1] += 1;

            glm::vec3 baseAxis = glm::vec3(0.0f, 1.0f, 0.0f);
            baseAxis = glm::normalize(baseAxis);
            float baseAngle = glm::radians(00.0f);
            baseAxis *= baseAngle;
            rotations[0] = (std::vector<double>{baseAxis.x, baseAxis.y, baseAxis.z});
            glm::mat4 baseRotation = getRotationMatrix(0);
            auto baseTransformation = glmToCeresMatrix(baseTranslation * baseRotation);

            glm::vec3 relativeAxis = glm::vec3(0.0f, 1.0f, 0.0f);
            relativeAxis = glm::normalize(relativeAxis);
            float relativeAngle = glm::radians(90.0f);
            relativeAxis *= relativeAngle;
            rotations[1] = (std::vector<double>{relativeAxis.x, relativeAxis.y, relativeAxis.z});
            glm::mat4 relativeRotation = getRotationMatrix(1);
            auto relativeTransformation = glmToCeresMatrix(relativeTranslation * relativeRotation);
            rotations[1][0] -= 1;
            rotations[1][1] += 0.5;

            std::vector<std::vector<double>> expectedTranslations;
            expectedTranslations.emplace_back(std::vector<double>{baseTranslationVector.x, baseTranslationVector.y, baseTranslationVector.z});
            expectedTranslations.emplace_back(std::vector<double>{relativeTranslationVector.x, relativeTranslationVector.y, relativeTranslationVector.z});

            std::vector<std::vector<double>> expectedRotations;
            expectedRotations.emplace_back(std::vector<double>{baseAxis.x, baseAxis.y, baseAxis.z});
            expectedRotations.emplace_back(std::vector<double>{relativeAxis.x, relativeAxis.y, relativeAxis.z});

            std::vector<ceres::Vector> points;
            points.emplace_back(Eigen::Vector4d(1.0f, 1.0f, 0.0f, 1.0f));
            points.emplace_back(Eigen::Vector4d(1.0f, 1.0f, 1.0f, 1.0f));
            points.emplace_back(Eigen::Vector4d(1.0f, -1.0f, 0.0f, 1.0f));
            points.emplace_back(Eigen::Vector4d(-1.0f, 1.0f, 0.0f, 1.0f));
            points.emplace_back(Eigen::Vector4d(-1.0f, -1.0f, 0.0f, 1.0f));
            points.emplace_back(Eigen::Vector4d(0.5f, 0.5f, 0.0f, 1.0f));
            points.emplace_back(Eigen::Vector4d(0.5f, -0.5f, 0.0f, 1.0f));
            points.emplace_back(Eigen::Vector4d(-0.5f, 0.5f, 0.0f, 1.0f));
            points.emplace_back(Eigen::Vector4d(-0.5f, -0.5f, 0.0f, 1.0f));
            
            std::vector<ceres::Vector> baseFramePoints;
            std::vector<ceres::Vector> relativeFramePoints;

            std::vector<ACharacteristicPoints> characteristicPoints;
            characteristicPoints.emplace_back(MockCharacteristicPoints());
            characteristicPoints.emplace_back(MockCharacteristicPoints());

            for (int i = 0; i < points.size(); i++)
            {
                characteristicPoints[0].markerCorners[0].emplace_back(baseTransformation * points[i]);
                characteristicPoints[1].markerCorners[0].emplace_back(relativeTransformation * points[i]);
            }

            ABundleAdjustment::solvePointCorrespondenceError(characteristicPoints);

            std::cout << vc::utils::toString("Expected", expectedTranslations, expectedRotations);

            std::cout << std::endl;

            return true;
        }
    };

    class BundleAdjustment : public ABundleAdjustment {
    public:
        bool specific_optimize(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines) {
            //vc::utils::sleepFor("Pre recalculation after optimization", 4000);

            if (!solvePointCorrespondenceError(pipelines)) {
                return false;
            }
            //vc::utils::sleepFor("After recalculation after optimization", 4000);

            return true;
        }

    private:
        std::vector<ACharacteristicPoints> getCharacteristicPoints(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines) {
            std::vector<ACharacteristicPoints> characteristicPoints;
            for (int i = 0; i < pipelines.size(); i++)
            {
                characteristicPoints.emplace_back(CharacteristicPoints(pipelines[i]));
            }
            return characteristicPoints;
        }

        bool solvePointCorrespondenceError(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines) {
            std::vector<ACharacteristicPoints> characteristicPoints = getCharacteristicPoints(pipelines);

            ABundleAdjustment::solvePointCorrespondenceError(characteristicPoints);

            return true;
        }

    };
}
#endif // !_BUNDLE_ADJUSTMENT_HEADER
