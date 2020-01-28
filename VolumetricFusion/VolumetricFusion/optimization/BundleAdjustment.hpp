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
#include "CeresOptimizationProblem.hpp"
#include "OptimizationProblem.hpp"
#include "PointCorrespondenceError.hpp"
#include "ReprojectionError.hpp"
#include "../Utils.hpp"
#include "../CaptureDevice.hpp"

namespace vc::optimization {
    class BundleAdjustment : public virtual CeresOptimizationProblem {
    
    public:

        BundleAdjustment(bool verbose = false, bool withSleep = false) : CeresOptimizationProblem(verbose, withSleep) {
            for (int i = 0; i < 4; i++)
            {
                translations.push_back(std::vector<double> { 0.0, 0.0, 0.0 });
                rotations.push_back(std::vector<double> { 0.0, 2 * M_PI, 0.0 });
                scales.push_back(std::vector<double> {1.0, 1.0, 1.0  });

                intrinsics.push_back(std::vector<double> { 0.0, 0.0, 0.0, 0.0 });
                distCoeffs.push_back(std::vector<double> { 0.0, 0.0, 0.0, 0.0 });
            }
        }
        
        void solveProblem(ceres::Problem* problem) {
            std::vector<Eigen::Matrix4d> initialTransformations(OptimizationProblem::bestTransformations);
            
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

    //        //if (verbose) {
    //            //std::cout << summary.FullReport() << "\n";
				//std::cout << "Bundle Adjustment" << std::endl;
    //            //std::cout << vc::utils::toString("Initial", initialTransformations);
    //            std::cout << vc::utils::toString("Final", bestTransformations);
    //            std::cout << std::endl;
    //        //}
        }

		//solvePointCorrespondenceError
        bool solveErrorFunction() {

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

        void initialize() {
            vc::optimization::Procrustes procrustes = vc::optimization::Procrustes(verbose);
            procrustes.characteristicPoints = characteristicPoints;
            if (procrustes.optimizeOnPoints()) {
                bestTransformations = procrustes.bestTransformations;
                currentRotations = procrustes.currentRotations;
                currentTranslations = procrustes.currentTranslations;
                currentScales = procrustes.currentScales;
                hasInitialization = true;
                setup();
            }
            else {
				hasInitialization = false;
            }
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