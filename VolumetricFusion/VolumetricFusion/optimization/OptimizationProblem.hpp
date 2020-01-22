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
#include <sstream>

#include "CharacteristicPoints.hpp"
#include "PointCorrespondenceError.hpp"
#include "ReprojectionError.hpp"
#include "../Utils.hpp"
#include "../CaptureDevice.hpp"
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

namespace vc::optimization {
    
    Eigen::Matrix4d glmToEigenMatrix(glm::mat4 matrix, bool verbose = false, std::string name = "") {
        Eigen::Matrix4d final_matrix = Eigen::Map<Eigen::Matrix<glm::f32, 4, 4>>(glm::value_ptr(matrix)).cast<double>();

        if (verbose) {
            std::cout << name << std::endl;
            std::cout << final_matrix;
        }

        return final_matrix;
    }

    glm::mat4 eigenToGlmMatrix(Eigen::Matrix4d matrix, bool verbose = false, std::string name = "") {
        glm::mat4 final_matrix = glm::make_mat4(matrix.data());

        if (verbose) {
            std::cout << name << std::endl;
            //std::cout << final_matrix;
        }

        return final_matrix;
    }

    class OptimizationProblem {
    protected:
        long sleepDuration = -1l;
        bool verbose = false;

        std::vector<CharacteristicPointsRenderer> characteristicPointsRenderers;

        std::vector<double> bestErrors;

        virtual void clear() {
            characteristicPoints = {
                CharacteristicPoints(),
                CharacteristicPoints(),
                CharacteristicPoints(),
                CharacteristicPoints()
            };
        }

    public:
        std::vector<float*> colors = {
            new float[4]{1.0f, 0.0f, 0.0f, 1.0f},
            new float[4]{0.0f, 1.0f, 0.0f, 1.0f},
            new float[4]{0.0f, 0.0f, 1.0f, 1.0f},
            new float[4]{1.0f, 1.0f, 0.0f, 1.0f}
        };

        std::vector<ACharacteristicPoints> characteristicPoints = {
            CharacteristicPoints(),
            CharacteristicPoints(),
            CharacteristicPoints(),
            CharacteristicPoints()
        };

        std::vector<Eigen::Matrix4d> currentTranslations;
        std::vector<Eigen::Matrix4d> currentRotations;
        std::vector<Eigen::Matrix4d> currentScales;

        std::vector<Eigen::Matrix4d> bestTransformations;

        void reset() {
             currentTranslations = {
                Eigen::Matrix4d::Identity(),
                Eigen::Matrix4d::Identity(),
                Eigen::Matrix4d::Identity(),
                Eigen::Matrix4d::Identity()
            };
             currentRotations = {
                Eigen::Matrix4d::Identity(),
                Eigen::Matrix4d::Identity(),
                Eigen::Matrix4d::Identity(),
                Eigen::Matrix4d::Identity()
            };
             currentScales = {
                Eigen::Matrix4d::Identity(),
                Eigen::Matrix4d::Identity(),
                Eigen::Matrix4d::Identity(),
                Eigen::Matrix4d::Identity()
            };

             bestTransformations = {
                Eigen::Matrix4d::Identity(),
                Eigen::Matrix4d::Identity(),
                Eigen::Matrix4d::Identity(),
                Eigen::Matrix4d::Identity()
            };
             bestErrors = {
                DBL_MAX,
                DBL_MAX,
                DBL_MAX,
                DBL_MAX
            };
        }

        OptimizationProblem(bool verbose = false, long sleepDuration = -1l) : verbose(verbose), sleepDuration(sleepDuration) {
            reset();
            clear();
        }

        Eigen::Matrix4d generateScaleMatrix(double x, double y, double z) {
            Eigen::Matrix4d scale = Eigen::Matrix4d::Identity();
            scale.diagonal() = Eigen::Vector4d(x, y, z, 1.0);
            return scale;
        }

        Eigen::Matrix4d generateRotationMatrix(double radians, Eigen::Vector3d axis = Eigen::Vector3d::Identity()) {
            Eigen::Matrix3d R = Eigen::AngleAxisd(radians, axis.normalized()).matrix();
            Eigen::Matrix4d Trans; // Your Transformation Matrix
            Trans.setIdentity();   // Set to Identity to make bottom row of Matrix 0,0,0,1
            Trans.block<3, 3>(0, 0) = R;
            return Trans;
        }

        Eigen::Matrix4d generateTranslationMatrix(double tx, double ty, double tz) {
            Eigen::Vector3d T = Eigen::Vector3d(tx, ty, tz);
            Eigen::Matrix4d Trans; // Your Transformation Matrix
            Trans.setIdentity();   // Set to Identity to make bottom row of Matrix 0,0,0,1
            Trans.block<3, 1>(0, 3) = T;
            return Trans;
        }


        Eigen::Matrix4d generateTransformationMatrix(Eigen::Vector3d angleAxis) {
            return generateTransformationMatrix(0, 0, 0, angleAxis.norm(), angleAxis.normalized());
        }

        Eigen::Matrix4d generateTransformationMatrix(Eigen::Vector4d translation, double radians, Eigen::Vector3d axis = Eigen::Vector3d::Identity()) {
            return generateTransformationMatrix(translation[0], translation[1], translation[2], radians, axis);
        }

        Eigen::Matrix4d generateTransformationMatrix(double tx, double ty, double tz, double radians, Eigen::Vector3d axis) {
            Eigen::Matrix3d R = Eigen::AngleAxisd(radians, axis.normalized()).matrix();
            // Find your Rotation Matrix
            Eigen::Vector3d T = Eigen::Vector3d(tx, ty, tz);
            // Find your translation Vector
            Eigen::Matrix4d Trans; // Your Transformation Matrix
            Trans.setIdentity();   // Set to Identity to make bottom row of Matrix 0,0,0,1
            Trans.block<3, 3>(0, 0) = R;
            Trans.block<3, 1>(0, 3) = T;
            return Trans;
        }

        void setupOpenGL() {
            for (int i = 0; i < 4; i++)
            {
                characteristicPointsRenderers.emplace_back(CharacteristicPointsRenderer());
            }
        }

        bool checkForAllMarkers(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines) {
            for (auto& pipeline : pipelines)
            {
                if (!pipeline->chArUco->hasMarkersDetected) {
                    return false;
                }
            }
            return true;
        }

        virtual void calculateTransformations() {
            // STUB for testing
        }


        virtual bool init(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines) {
            clear();

            if (pipelines.size() == 1) {
                return true;
            }
            
            if (!checkForAllMarkers(pipelines)) {
                return false;
            }

            return true;
        }

        virtual Eigen::Matrix4d getCurrentTransformation(int camera_index) {
            return currentTranslations[camera_index] * currentRotations[camera_index] * currentScales[camera_index];
        }

        Eigen::Matrix4d getBestTransformation(int camera_index) {
            return bestTransformations[camera_index];
        }

        Eigen::Matrix4d getCurrentRelativeTransformation(int from, int to) {
            return getCurrentTransformation(to) * getCurrentTransformation(from).inverse();
        }

        Eigen::Matrix4d getBestRelativeTransformation(int from, int to) {
            return getBestTransformation(to) * getBestTransformation(from).inverse();
        }

        void getCharacteristicPoints(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines) {
            std::vector<vc::optimization::ACharacteristicPoints> current(pipelines.size());
            for (int i = 0; i < pipelines.size(); i++)
            {
                current[i] = CharacteristicPoints(pipelines[i]);
            }
            characteristicPoints = current;
        }

        virtual void randomize() {
            currentTranslations[1] = generateTransformationMatrix(std::rand() % 1000 / 500.0 - 1.0, std::rand() % 1000 / 500.0 - 1.0, std::rand() % 1000 / 500.0 - 1.0, std::rand() % 360, Eigen::Vector3d::Random());
            currentRotations[1] = generateTransformationMatrix(std::rand() % 1000 / 500.0 - 1.0, std::rand() % 1000 / 500.0 - 1.0, std::rand() % 1000 / 500.0 - 1.0, std::rand() % 360, Eigen::Vector3d::Random());
            currentScales[1] = generateScaleMatrix(std::rand() % 1000 / 500.0 - 1.0, std::rand() % 1000 / 500.0 - 1.0, std::rand() % 1000 / 500.0 - 1.0);
        }

        bool optimizeOnPoints() {
            //randomize();
            vc::utils::sleepFor("Pre optimization", sleepDuration);

            auto success = specific_optimize();

            vc::utils::sleepFor("After optimization", sleepDuration);

            evaluate();

            return success;
        }

        bool optimize(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines = std::vector<std::shared_ptr<vc::capture::CaptureDevice>>()) {
            if (!init(pipelines)) {
                return false;
            }

            getCharacteristicPoints(pipelines);
            return optimizeOnPoints();
        }

        void evaluate() {
            for (int i = 0; i < characteristicPoints.size(); i++)
            {
                double error = calculateRelativeError(i, 0);

                if (error < bestErrors[i]) {
                    bestErrors[i] = error;
                    bestTransformations[i] = getCurrentTransformation(i); 
                    //std::cout << vc::utils::toString("Transformation " + std::to_string(i), bestTransformations[i]);
                }
            }
        }

        virtual bool specific_optimize() = 0;

        void render(glm::mat4 model, glm::mat4 view, glm::mat4 projection, int i) {
            //std::cout << characteristicPointsRenderers.size() << std::endl;
            //std::cout << characteristicPoints.size() << std::endl;
            //std::cout << transformations.size() << std::endl;

            try {
                //for (int i = 0; i < characteristicPoints.size(); i++)
                {
                    //std::cout << vc::utils::asHeader("Rendering: " + std::to_string(i));
                    //std::cout << vc::utils::toString("Best Transformation " + std::to_string(i), getBestRelativeTransformation(i, 0));

                    characteristicPointsRenderers[i].render(&characteristicPoints[i], model, view, projection, getBestRelativeTransformation(i, 0), colors[i]);
                }
            }
            catch (std::exception&)
            {
                std::cout << vc::utils::asHeader("Normal Exception");
            }
            catch (std::out_of_range&)
            {
                std::cout << vc::utils::asHeader("Out of range Exception");
            }
        }

        double calculateRelativeError(int from, int to) {
            auto relativeTransformation = getCurrentRelativeTransformation(from, to);

            std::vector<unsigned long long> matchingHashes = vc::utils::findOverlap(
                characteristicPoints[from].getHashes(verbose), characteristicPoints[to].getHashes(verbose)
            );

            auto& basePoints = characteristicPoints[to].getFilteredPoints(matchingHashes, verbose);
            auto& relativePoints = characteristicPoints[from].getFilteredPoints(matchingHashes, verbose);

            double error = 0;

            for (auto& hash : matchingHashes)
            {
                auto basePoint = basePoints[hash];
                auto relativePoint = relativePoints[hash];
                auto transformed = relativeTransformation * relativePoint;
                //std::cout << "Base: " << std::endl << basePoint << std::endl;
                //std::cout << "Relative: " << std::endl << relativePoint << std::endl;
                //std::cout << "Transformed: " << std::endl << transformed << std::endl;

                error += (basePoint - transformed).norm();
            }

            if (verbose) {
                std::cout << vc::utils::asHeader("Final error: " + std::to_string(error)) << std::endl;
            }

            return error;
        }
    };

    class MockOptimizationProblem : virtual public OptimizationProblem {
    protected:
        Eigen::Matrix4d expectedRelativeTransformation;

    public:
        ~MockOptimizationProblem() {
            std::cout << vc::utils::toString("Final total expected:", expectedRelativeTransformation);
            std::cout << vc::utils::toString("Final total:", Eigen::Matrix4d(getCurrentRelativeTransformation(1, 0)));
        }
        
        void setupMock() {
            verbose = true;
            sleepDuration = -1l;

            Eigen::Matrix4d baseTranslation = generateTranslationMatrix(0, 0, 1);
            Eigen::Matrix4d baseRotation = generateRotationMatrix(0, Eigen::Vector3d(0, 1, 0));
            Eigen::Matrix4d baseScale = generateScaleMatrix(1, 1, 1);
            Eigen::Matrix4d baseTransformation = baseTranslation * baseRotation * baseScale;

            Eigen::Matrix4d relativeTranslation = generateTranslationMatrix(0, 0, 3);
            Eigen::Matrix4d relativeRotation = generateRotationMatrix(M_PI / 2.0, Eigen::Vector3d(0, 1, 0));
            Eigen::Matrix4d relativeScale = generateScaleMatrix(1, 1, 1);
            Eigen::Matrix4d relativeTransformation = relativeTranslation * relativeRotation * relativeScale;
            
            //Eigen::Matrix4d baseTransformation = generateTransformationMatrix(0, 0, 0, 0, Eigen::Vector3d(0, 0, 0));
            //Eigen::Matrix4d relativeTransformation = generateTransformationMatrix(0, 0, 0, 0, Eigen::Vector3d(0, 0, 0));

            std::vector<Eigen::Vector4d> points{
                Eigen::Vector4d(1.0f, 1.0f, 0.0f, 1.0f) ,
                Eigen::Vector4d(1.0f, 1.0f, 1.0f, 1.0f) ,
                Eigen::Vector4d(1.0f, -1.0f, 0.0f, 1.0f),
                Eigen::Vector4d(-1.0f, 1.0f, 0.0f, 1.0f),
                Eigen::Vector4d(-1.0f, -1.0f, 0.0f, 1.0f),
                //Eigen::Vector4d(0.5f, 0.5f, 0.0f, 1.0f),
                //Eigen::Vector4d(0.5f, -0.5f, 0.0f, 1.0f),
                //Eigen::Vector4d(-0.5f, 0.5f, 0.0f, 1.0f),
                //Eigen::Vector4d(-0.5f, -0.5f, 0.0f, 1.0f)
            };

            characteristicPoints[0] = (MockCharacteristicPoints());
            characteristicPoints[1] = (MockCharacteristicPoints());

            for (int i = 0; i < points.size() / 2; i++)
            {
                characteristicPoints[0].markerCorners[i % 4].emplace_back(baseTransformation * points[i]);
                characteristicPoints[1].markerCorners[i % 4].emplace_back(relativeTransformation * points[i]);
            }

            for (int i = points.size() / 2; i < points.size(); i++)
            {
                characteristicPoints[0].charucoCorners[i] = baseTransformation * points[i];
                characteristicPoints[1].charucoCorners[i] = relativeTransformation * points[i];
            }

            // Outliers
            characteristicPoints[0].markerCorners[7].emplace_back(Eigen::Vector4d(-1.0f, -1.0f, 0.0f, 1.0f));
            characteristicPoints[1].markerCorners[9].emplace_back(Eigen::Vector4d(-1.0f, -1.0f, 0.0f, 1.0f));

            //currentTransformations[0] = Eigen::Matrix4d::Identity();
            //currentTransformations[1] = baseTransformation * relativeTransformation.inverse();

            expectedRelativeTransformation = baseTransformation * relativeTransformation.inverse();
        }

        bool vc::optimization::OptimizationProblem::specific_optimize() {
            long milliseconds = 1000;
            vc::utils::sleepFor("", milliseconds, false);
            return true;
        }
    };
}
#endif