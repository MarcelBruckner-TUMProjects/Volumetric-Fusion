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
                for (int j = 0; j < 4; j++)
                {
                    translations[i].push_back(std::vector<double> { 0.0, 0.0, 0.0 });
                    rotations[i].push_back(std::vector<double> { 0.0, 2 * M_PI, 0.0 });
                    scales[i].push_back(std::vector<double> {1.0, 1.0, 1.0  });
                }
            }
        }
        
        void solveProblem(ceres::Problem* problem) {
            std::vector<Eigen::Matrix4d> initialTransformations(OptimizationProblem::bestTransformations);
            
            ceres::Solver::Options options;
            options.num_threads = 16;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = verbose;
            options.max_num_iterations = 20;
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
            for (int from = 0; from < characteristicPoints.size(); from++)
            {
                ACharacteristicPoints fromPoints = characteristicPoints[from];
                
                for (int to = 0; to < characteristicPoints.size(); to++) {
                    if (from == to) {
                        continue;
                    }

                    ACharacteristicPoints toPoints = characteristicPoints[to];
                    std::vector<int> matchingHashes = vc::utils::findOverlap(fromPoints.getHashes(verbose), toPoints.getHashes(verbose));
                    
                    auto& filteredFromPoints = fromPoints.getFilteredPoints(matchingHashes, verbose);
                    auto& filteredToPoints = toPoints.getFilteredPoints(matchingHashes, verbose);

                    for (auto& hash : matchingHashes)
                    {
                        auto fromPoint = filteredFromPoints[hash];
                        auto toPoint = filteredToPoints[hash];

                        if (!isValid(fromPoint) || !isValid(toPoint)) {
                            continue;
                        }

                        ceres::CostFunction* cost_function = PointCorrespondenceError::Create(
                            hash, 
                            fromPoint,
                            toPoint,
                            verbose
                        );
                        problem.AddResidualBlock(cost_function, NULL,
                            translations[from][to].data(),
                            rotations[from][to].data(),
                            scales[from][to].data()
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
                //std::cout << vc::utils::toString(currentTranslations, "\n\n", "\n\n ******************************************************************* \n\n") << std::endl;
                //std::cout << "\n\n ########################################################################## \n\n";
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
            //hasProcrustesInitialization = true;
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