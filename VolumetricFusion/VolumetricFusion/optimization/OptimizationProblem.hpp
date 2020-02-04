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

        int numberOfPipelines = 0;

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
        bool needsReset = false;

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

        std::vector<std::vector<Eigen::Matrix4d>> currentTranslations;
        std::vector<std::vector<Eigen::Matrix4d>> currentRotations;
        std::vector<std::vector<Eigen::Matrix4d>> currentScales;
                   
        std::vector<Eigen::Matrix4d> bestTransformations;

        std::vector<Eigen::Matrix4d> makeZero() {
            return { Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity() };
        }

        std::vector<std::vector<Eigen::Matrix4d>> makeAllZero() {
            return { makeZero(), makeZero(), makeZero(), makeZero() };
        }

        virtual void reset() {
            currentTranslations = makeAllZero();
            currentRotations = makeAllZero();
            currentScales = makeAllZero();
            bestTransformations = makeZero();

            bestErrors = {
                DBL_MAX,
                DBL_MAX,
                DBL_MAX,
                DBL_MAX
            };
            needsReset = false;
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
            if (needsReset) {
                reset();
            }

            clear();

            if (pipelines.size() == 1) {
                return true;
            }
            
            //if (!checkForAllMarkers(pipelines)) {
            //    return false;
            //}

            return true;
        }

        virtual Eigen::Matrix4d getCurrentTransformation(int from, int to) {
            return currentTranslations[from][to] * currentRotations[from][to] * currentScales[from][to];
        }

        Eigen::Matrix4d getBestTransformation(int cameraIndex) {
            return bestTransformations[cameraIndex];
        }
        
        void getCharacteristicPoints(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines, bool* activeOnes = new bool[4]{ true, true, true, true }) {
            std::vector<vc::optimization::ACharacteristicPoints> current(pipelines.size());
            for (int i = 0; i < pipelines.size(); i++)
            {
                if (activeOnes[i]) {
                    current[i] = CharacteristicPoints(pipelines[i]);
                }
            }
            characteristicPoints = current;
        }

        //virtual void randomize() {
        //    currentTranslations[1] = generateTransformationMatrix(std::rand() % 1000 / 500.0 - 1.0, std::rand() % 1000 / 500.0 - 1.0, std::rand() % 1000 / 500.0 - 1.0, std::rand() % 360, Eigen::Vector3d::Random());
        //    currentRotations[1] = generateTransformationMatrix(std::rand() % 1000 / 500.0 - 1.0, std::rand() % 1000 / 500.0 - 1.0, std::rand() % 1000 / 500.0 - 1.0, std::rand() % 360, Eigen::Vector3d::Random());
        //    currentScales[1] = generateScaleMatrix(std::rand() % 1000 / 500.0 - 1.0, std::rand() % 1000 / 500.0 - 1.0, std::rand() % 1000 / 500.0 - 1.0);
        //}

        bool optimizeOnPoints() {
            //randomize();
            vc::utils::sleepFor("Pre optimization", sleepDuration);

            if (!specific_optimize()) {
                return false;
            }

            vc::utils::sleepFor("After optimization", sleepDuration);

            evaluate();

            for (int i = 1; i < characteristicPoints.size(); i++)
            {
                if (bestErrors[i] == DBL_MAX) {
                    return false;
                }
            }

            return true;
        }

        virtual bool optimize(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines = std::vector<std::shared_ptr<vc::capture::CaptureDevice>>(), bool* activeOnes = new bool[4]{ true, true, true, true }) {
            if (!init(pipelines)) {
                return false;
            }

            numberOfPipelines = pipelines.size();

            getCharacteristicPoints(pipelines);
            return optimizeOnPoints();
        }

        std::vector<std::vector<int>> enumerate(int startIndex, std::vector<int> seenIndices = {}) {
            if (startIndex == 0) {
                return { { 0} };
            }
            else {
                std::vector<std::vector<int>> paths;
                std::vector<int> newSeenIndices;
                for (auto& seenIndex : seenIndices)
                {
                    newSeenIndices.emplace_back(seenIndex);
                }
                newSeenIndices.emplace_back(startIndex);
                for (int i = 0; i < characteristicPoints.size(); i++)
                {
                    if (vc::utils::contains(newSeenIndices, i)) {
                        continue;
                    }
                    std::vector<std::vector<int>> followingPaths = enumerate(i, newSeenIndices);
                    for (auto& followingPath : followingPaths)
                    {
                        auto nextHop = followingPath[followingPath.size() - 1];
                        auto transformation = getCurrentTransformation(startIndex, nextHop);
                        //std::stringstream ss;
                        //ss << index << " - " << startIndex;
                        //std::cout << vc::utils::toString(ss.str(), transformation);
                        if ((transformation - Eigen::Matrix4d::Identity()).isZero(1e-3)) {
                            continue;
                        }

                        followingPath.push_back(startIndex);
                        paths.emplace_back(followingPath);
                    }
                 }
                return paths;
            }
        }

        Eigen::Matrix4d pathToMatrix(std::vector<int> path) {
            Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
            
            for (int i = path.size() - 1; i >= 1; i--)
            {
                result = getCurrentTransformation(path[i], path[i - 1]) * result;
            }
            return result;
        }

        void evaluate() {
            bestTransformations[0] = Eigen::Matrix4d::Identity();
            bestErrors[0] = 0;

            for (int i = 1; i < characteristicPoints.size(); i++)
            {
                auto paths = enumerate(i);

                std::cout << vc::utils::toString(paths);
                for (auto& path : paths)
                {
                    double error = 0;

                    for (int i = path.size() - 1; i >= 1; i--)
                    {
                        error += calculateRelativeError(path[i], path[i - 1]);
                    }

                    if (error <= bestErrors[i]) {
                        bestErrors[i] = error;

                        bestTransformations[i] = pathToMatrix(path);
                        std::cout << vc::utils::toString("Transformation " + std::to_string(i), bestTransformations[i]);
                    }
                }
            }
        }

        virtual bool specific_optimize() = 0;

        void render(glm::mat4 model, glm::mat4 view, glm::mat4 projection, int i) {
            try {
                characteristicPointsRenderers[i].render(&characteristicPoints[i], model, view, projection, getBestTransformation(i), colors[i]);
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
            auto relativeTransformation = getCurrentTransformation(from, to);

            if (std::abs(relativeTransformation.bottomRightCorner< 1, 1>()[0] - 1) > 1e-2) {
                return DBL_MAX;
            }

            std::vector<int> matchingHashes = vc::utils::findOverlap(
                characteristicPoints[from].getHashes(verbose), characteristicPoints[to].getHashes(verbose)
            );

            if (matchingHashes.size() <= 4) {
                    return DBL_MAX;
            }

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
                std::stringstream ss;
                ss << "Final error (" << from << ", " << to << "): " << error;

                std::cout << vc::utils::asHeader(ss.str()) << std::endl;
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
            std::cout << vc::utils::toString("Final total:", Eigen::Matrix4d(getCurrentTransformation(1, 0)));
        }

        void justTranslation(
            std::vector<Eigen::Matrix4d>& translations,
            std::vector<Eigen::Matrix4d>& roatations,
            std::vector<Eigen::Matrix4d>& scales,
            std::vector<Eigen::Matrix4d>& transformations) {
            translations[0] = generateTranslationMatrix(0, 0, 0);
            roatations[0] = generateRotationMatrix(0, Eigen::Vector3d(0, 1, 0));
            scales[0] = generateScaleMatrix(1, 1, 1);
            transformations[0] = translations[0] * roatations[0] * scales[0];

            translations[1] = generateTranslationMatrix(0, 0, 1);
            roatations[1] = generateRotationMatrix(0, /*M_PI / 2.0,*/ Eigen::Vector3d(0, 1, 0));
            scales[1] = generateScaleMatrix(1, 1, 1);
            transformations[1] = translations[1] * roatations[1] * scales[1];

            translations[2] = generateTranslationMatrix(0, 0, 2);
            roatations[2] = generateRotationMatrix(0, /*M_PI / 2.0,*/ Eigen::Vector3d(0, 1, 0));
            scales[2] = generateScaleMatrix(1, 1, 1);
            transformations[2] = translations[2] * roatations[2] * scales[2];
        }

        void justRotation(
            std::vector<Eigen::Matrix4d>& translations,
            std::vector<Eigen::Matrix4d>& roatations,
            std::vector<Eigen::Matrix4d>& scales,
            std::vector<Eigen::Matrix4d>& transformations) {
            translations[0] = generateTranslationMatrix(0, 0, 0);
            roatations[0] = generateRotationMatrix(0, Eigen::Vector3d(0, 1, 0));
            scales[0] = generateScaleMatrix(1, 1, 1);
            transformations[0] = translations[0] * roatations[0] * scales[0];

            translations[1] = generateTranslationMatrix(0, 0, 0);
            roatations[1] = generateRotationMatrix(M_PI / 2.0, Eigen::Vector3d(0, 1, 0));
            scales[1] = generateScaleMatrix(1, 1, 1);
            transformations[1] = translations[1] * roatations[1] * scales[1];

            translations[2] = generateTranslationMatrix(0, 0, 0);
            roatations[2] = generateRotationMatrix(-M_PI / 2.0, Eigen::Vector3d(0, 1, 0));
            scales[2] = generateScaleMatrix(1, 1, 1);
            transformations[2] = translations[2] * roatations[2] * scales[2];
        }

        void setupMock() {
            reset();
            clear();

            verbose = true;
            sleepDuration = -1l;

            std::vector<Eigen::Matrix4d> translations(3);
            std::vector<Eigen::Matrix4d> roatations(3);
            std::vector<Eigen::Matrix4d> scales(3);
            std::vector<Eigen::Matrix4d> transformations(3);

            justTranslation(translations, roatations, scales, transformations);
            //justRotation(translations, roatations, scales, transformations);

            std::vector<Eigen::Vector4d> points{
                Eigen::Vector4d(1.0f, 1.0f, 0.0f, 1.0f) ,
                Eigen::Vector4d(1.0f, 1.0f, 1.0f, 1.0f) ,
                Eigen::Vector4d(1.0f, -1.0f, 0.0f, 1.0f),
                Eigen::Vector4d(-1.0f, 1.0f, 0.0f, 1.0f),
                Eigen::Vector4d(-1.0f, -1.0f, 0.0f, 1.0f),
                Eigen::Vector4d(0.5f, 0.5f, 1.0f, 1.0f),
                Eigen::Vector4d(0.5f, -0.5f, 1.0f, 1.0f),
                Eigen::Vector4d(-0.5f, 0.5f, -1.0f, 1.0f),
                Eigen::Vector4d(-0.5f, -0.5f, -1.0f, 1.0f),

                Eigen::Vector4d(-1.0f, 1.0f, 0.0f, 1.0f) ,
                Eigen::Vector4d(-1.0f, 1.0f, 1.0f, 1.0f) ,
                Eigen::Vector4d(-1.0f, -1.0f, 0.0f, 1.0f),
                Eigen::Vector4d(1.0f, 1.0f, 0.0f, 1.0f),
                Eigen::Vector4d(1.0f, -1.0f, 0.0f, 1.0f),
                Eigen::Vector4d(-0.5f, 0.5f, 1.0f, 1.0f),
                Eigen::Vector4d(-0.5f, -0.5f, 1.0f, 1.0f),
                Eigen::Vector4d(0.5f, 0.5f, -1.0f, 1.0f),
                Eigen::Vector4d(0.5f, -0.5f, -1.0f, 1.0f),

                Eigen::Vector4d(-1.0f, -1.0f, 0.0f, 1.0f) ,
                Eigen::Vector4d(-1.0f, -1.0f, 1.0f, 1.0f) ,
                Eigen::Vector4d(-1.0f, 1.0f, 0.0f, 1.0f),
                Eigen::Vector4d(1.0f, -1.0f, 0.0f, 1.0f),
                Eigen::Vector4d(1.0f, 1.0f, 0.0f, 1.0f),
                Eigen::Vector4d(-0.5f, -0.5f, 1.0f, 1.0f),
                Eigen::Vector4d(-0.5f, 0.5f, 1.0f, 1.0f),
                Eigen::Vector4d(0.5f, -0.5f, -1.0f, 1.0f),
                Eigen::Vector4d(0.5f, 0.5f, -1.0f, 1.0f),

                Eigen::Vector4d(-1.0f, -1.0f, -0.0f, 1.0f) ,
                Eigen::Vector4d(-1.0f, -1.0f, -1.0f, 1.0f) ,
                Eigen::Vector4d(-1.0f, 1.0f, -0.0f, 1.0f),
                Eigen::Vector4d(1.0f, -1.0f, -0.0f, 1.0f),
                Eigen::Vector4d(1.0f, 1.0f, -0.0f, 1.0f),
                Eigen::Vector4d(-0.5f, -0.5f, -1.0f, 1.0f),
                Eigen::Vector4d(-0.5f, 0.5f, -1.0f, 1.0f),
                Eigen::Vector4d(0.5f, -0.5f, 1.0f, 1.0f),
                Eigen::Vector4d(0.5f, 0.5f, 1.0f, 1.0f)
            };

            characteristicPoints[0] = (MockCharacteristicPoints());
            characteristicPoints[1] = (MockCharacteristicPoints());
            //characteristicPoints[2] = (MockCharacteristicPoints());
            //characteristicPoints[3] = (MockCharacteristicPoints());

            for (int i = 0; i < points.size(); i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    characteristicPoints[j].markerCorners[i % 4].emplace_back(transformations[j] * points[i]);
                }
            }
            
            // Outliers
            characteristicPoints[0].markerCorners[7].emplace_back(Eigen::Vector4d(-1.0f, -1.0f, 0.0f, 1.0f));
            characteristicPoints[1].markerCorners[8].emplace_back(Eigen::Vector4d(-1.0f, -1.0f, 0.0f, 1.0f));
            characteristicPoints[2].markerCorners[9].emplace_back(Eigen::Vector4d(-1.0f, -1.0f, 0.0f, 1.0f));

            //currentTranslations[0][0] = Eigen::Matrix4d::Identity();
            //currentTranslations[0][1] = baseTranslation * relativeTranslation.inverse();

            //currentRotations[0][0] = Eigen::Matrix4d::Identity();
            //currentRotations[0][1] = baseRotation * relativeRotation.inverse();

            //currentScales[0][0] = Eigen::Matrix4d::Identity();
            //currentScales[0][1] = baseScale * relativeScale.inverse();

            //currentTranslations[1][1] = Eigen::Matrix4d::Identity();
            //currentTranslations[1][0] = relativeTranslation * baseTranslation.inverse();

            //currentRotations[1][1] = Eigen::Matrix4d::Identity();
            //currentRotations[1][0] = relativeRotation * baseRotation.inverse();

            //currentScales[1][1] = Eigen::Matrix4d::Identity();
            //currentScales[1][0] = relativeScale * baseScale.inverse();

            //expectedRelativeTransformation = baseTransformation * relativeTransformation.inverse();
        }

        bool vc::optimization::OptimizationProblem::specific_optimize() {
            long milliseconds = 1000;
            vc::utils::sleepFor("", milliseconds, false);
            return true;
        }

        bool optimize(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines = std::vector<std::shared_ptr<vc::capture::CaptureDevice>>()) {
            
            return optimizeOnPoints();
        }
    };
}
#endif