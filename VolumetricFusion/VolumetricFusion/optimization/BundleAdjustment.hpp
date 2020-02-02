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

        std::vector<std::vector<std::vector<double>>> translations;
        std::vector<std::vector<std::vector<double>>> rotations;
        std::vector<std::vector<std::vector<double>>> scales;
        
        Eigen::Matrix4d getTransformation(int from , int to) {
            if (needsRecalculation) {
                calculateTransformations();
            }
            return OptimizationProblem::getCurrentTransformation(from, to);
        }

        //void randomize() {
        //    Eigen::Vector3d randomTranslation = Eigen::Vector3d::Random();
        //    translations[1][0] += (double)randomTranslation[0];
        //    translations[1][1] += (double)randomTranslation[1];
        //    translations[1][2] += (double)randomTranslation[2];

        //    Eigen::Vector3d randomRotation = Eigen::Vector3d::Random();
        //    randomRotation.normalize();
        //    double angle = std::rand() % 360;
        //    randomRotation *= glm::radians(angle);
        //    rotations[1][0] += (double)randomRotation[0];
        //    rotations[1][1] += (double)randomRotation[1];
        //    rotations[1][2] += (double)randomRotation[2];

        //    needsRecalculation = true;
        //}

        bool init(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines) {
            if (!OptimizationProblem::init(pipelines)) {
                return false;
            }
            
            needsRecalculation = true;

            return true;
        }
        Eigen::Matrix4d getRotationMatrix(int from, int to) {
            try {
                Eigen::Vector3d rotationVector(
                    rotations.at(from).at(to).at(0),
                    rotations.at(from).at(to).at(1),
                    rotations.at(from).at(to).at(2)
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

        Eigen::Matrix4d getTranslationMatrix(int from, int to) {
            try {
                return generateTransformationMatrix(
                    translations.at(from).at(to).at(0),
                    translations.at(from).at(to).at(1),
                    translations.at(from).at(to).at(2),
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

        Eigen::Matrix4d getScaleMatrix(int from, int to) {
            try {
                return generateScaleMatrix(
                    scales.at(from).at(to).at(0),
                    scales.at(from).at(to).at(1),
                    scales.at(from).at(to).at(2)
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
            for (int from = 0; from < translations.size(); from++) {
                for (int to = 0; to < translations.size(); to++)
                {
                    if (from == to || !(from < 3 && to < 3)) {
                        continue;
                    }
                    //std::stringstream ss;
                    //std::cout << vc::utils::asHeader("Pre recalculation");
                    //ss << "(" << from << ", " << to << ")";
                    //std::cout << vc::utils::asHeader("translations" + ss.str()) << vc::utils::toString(translations[from][to]);
                    //std::cout << vc::utils::asHeader("rotations" + ss.str()) << vc::utils::toString(rotations[from][to]);
                    //std::cout << vc::utils::asHeader("scales" + ss.str()) << vc::utils::toString(scales[from][to]);

                    //std::cout << vc::utils::toString("translations" + ss.str(), currentTranslations[from][to]);
                    //std::cout << vc::utils::toString("rotations" + ss.str(), currentRotations[from][to]);
                    //std::cout << vc::utils::toString("scales" + ss.str(), currentScales[from][to]);

                    currentTranslations[from][to] = getTranslationMatrix(from, to);
                    currentRotations[from][to] = getRotationMatrix(from, to);
                    currentScales[from][to] = getScaleMatrix(from, to);

                    //std::cout << vc::utils::asHeader("Post recalculation");
                    //ss = std::stringstream();
                    //ss << "(" << from << ", " << to << ")";
                    //std::cout << vc::utils::asHeader("translations" + ss.str()) << vc::utils::toString(translations[from][to]);
                    //std::cout << vc::utils::asHeader("rotations" + ss.str()) << vc::utils::toString(rotations[from][to]);
                    //std::cout << vc::utils::asHeader("scales" + ss.str()) << vc::utils::toString(scales[from][to]);

                    //std::cout << vc::utils::toString("translations" + ss.str(), currentTranslations[from][to]);
                    //std::cout << vc::utils::toString("rotations" + ss.str(), currentRotations[from][to]);
                    //std::cout << vc::utils::toString("scales" + ss.str(), currentScales[from][to]);
                 
                    //ss = std::stringstream();
                }
            }

            needsRecalculation = false;
        }
        
        BundleAdjustment(bool verbose = false, bool withSleep = false) : OptimizationProblem(verbose, withSleep) {
            translations = std::vector<std::vector<std::vector<double>>>(4);
            rotations = std::vector<std::vector<std::vector<double>>>(4);
            scales = std::vector<std::vector<std::vector<double>>>(4);
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

        void clear() {
            OptimizationProblem::clear();
            needsRecalculation = true;
        }

        void reset() {
            OptimizationProblem::reset();
            setup();
            //hasProcrustesInitialization = true;
            hasProcrustesInitialization = false;
        }
        
        void solveProblem(ceres::Problem* problem) {
            std::vector<Eigen::Matrix4d> initialTransformations = bestTransformations;
            
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

            if (verbose) {
                std::cout << summary.FullReport() << "\n";
                std::cout << vc::utils::toString("Initial", initialTransformations);
                std::cout << vc::utils::toString("Final", bestTransformations);

                std::cout << std::endl;
            }
        }

        bool isValid(Eigen::Vector4d point) {
            for (int i = 0; i < 3; i++)
            {
                if (std::abs(point[i]) > 10e5) {
                    return false;
                }
            }
            return true;
        }

        bool solvePointCorrespondenceError() {
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

        void initializeWithProcrustes() {
            vc::optimization::Procrustes procrustes = vc::optimization::Procrustes(verbose);
            procrustes.characteristicPoints = characteristicPoints;
            if (procrustes.optimizeOnPoints()) {
                bestTransformations = procrustes.bestTransformations;
                currentRotations = procrustes.currentRotations;
 
                currentTranslations = procrustes.currentTranslations;
                //std::cout << vc::utils::toString(currentTranslations, "\n\n", "\n\n ******************************************************************* \n\n") << std::endl;
                //std::cout << "\n\n ########################################################################## \n\n";
                currentScales = procrustes.currentScales;
                //std::cout << vc::utils::toString(currentScales, "\n\n", "\n\n ******************************************************************* \n\n") << std::endl;
                //std::cout << "\n\n ########################################################################## \n\n";
                hasProcrustesInitialization = true;
                setup();
            }
            else {
                hasProcrustesInitialization = false;
            }
        }

        void setup() {
            for (int from = 0; from < 4; from++)
            {
                for (int to = 0; to < 4; to++)
                {
                    Eigen::Vector3d translation = currentTranslations[from][to].block<3, 1>(0, 3);
                    double angle = Eigen::AngleAxisd(currentRotations[from][to].block<3, 3>(0, 0)).angle();
                    Eigen::Vector3d rotation = Eigen::AngleAxisd(currentRotations[from][to].block<3, 3>(0, 0)).axis().normalized();
                    rotation *= angle;
                    Eigen::Vector3d scale = currentScales[from][to].diagonal().block<3, 1>(0, 0);

                    for (int k = 0; k < 3; k++)
                    {
                        translations[from][to][k] = translation[k];
                        rotations[from][to][k] = rotation[k];
                        scales[from][to][k] = scale[k];
                    }
                }
            }
            calculateTransformations();
        }

        bool vc::optimization::OptimizationProblem::specific_optimize() {
            if (!hasProcrustesInitialization) {
                initializeWithProcrustes();
                if (!hasProcrustesInitialization) {
                    return false;
                }
            }

            if (!solvePointCorrespondenceError()) {
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