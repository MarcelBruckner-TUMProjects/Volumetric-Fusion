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
        std::vector<ACharacteristicPoints> characteristicPoints;
        std::vector<CharacteristicPointsRenderer> characteristicPointsRenderers;

        std::vector<glm::vec3> colors{
            glm::vec3(1.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec3(1.0f, 1.0f, 0.0f)
        };

        std::vector<Eigen::Matrix4d> relativeTransformations = {
            Eigen::Matrix4d::Identity(),
            Eigen::Matrix4d::Identity(),
            Eigen::Matrix4d::Identity(),
            Eigen::Matrix4d::Identity()
        };

        void clear() {
            relativeTransformations = {
                Eigen::Matrix4d::Identity(),
                Eigen::Matrix4d::Identity(),
                Eigen::Matrix4d::Identity(),
                Eigen::Matrix4d::Identity()
            };
        }

    public:

        OptimizationProblem() {
            clear();
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

        virtual bool init(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines) {
            clear();
            
            if (!checkForAllMarkers(pipelines)) {
                return false;
            }

            return true;
        }
        
        virtual Eigen::Matrix4d getTransformation(int camera_index) {
            return relativeTransformations[camera_index];
        }

        Eigen::Matrix4d getRelativeTransformation(int from, int to) {
            return getTransformation(to) * getTransformation(from).inverse();
        }

        std::vector<ACharacteristicPoints> getCharacteristicPoints(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines) {
            characteristicPoints.clear();
            for (int i = 0; i < pipelines.size(); i++)
            {
                characteristicPoints.emplace_back(CharacteristicPoints(pipelines[i]));
            }
            return characteristicPoints;
        }

        bool vc::optimization::OptimizationProblem::optimize(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines = std::vector<std::shared_ptr<vc::capture::CaptureDevice>>(), bool onlyOpenCV = false) {
            if (!init(pipelines)) {
                return false;
            }
            if (onlyOpenCV) {
                return true;
            }

            //vc::utils::sleepFor("Pre recalculation after optimization", 4000);

            auto success = specific_optimize(getCharacteristicPoints(pipelines));

            //vc::utils::sleepFor("After recalculation after optimization", 4000);
            return success;
        }

        virtual bool specific_optimize(std::vector<ACharacteristicPoints> characteristicPoints) = 0;

        void render(glm::mat4 model, glm::mat4 view, glm::mat4 projection) {
            for (int i = 0; i < characteristicPoints.size(); i++)
            {
                //characteristicPointsRenderers[i].render(&characteristicPoints[i], model, view, projection, getTransformation(i), colors[i]);
            }
        }
    };

    class MockOptimizationProblem : virtual public OptimizationProblem {
    protected:
        std::vector<Eigen::Matrix4d> expectedTransformations;
        std::vector<ACharacteristicPoints> mockCharacteristicPoints;

    public:
        void clear() {
            mockCharacteristicPoints.clear();
        }
        
        void calculateFinalError() {
            auto relativeTransformation = getTransformation(1);

            auto basePoints = mockCharacteristicPoints[0].getFlattenedPoints();
            auto relativePoints = mockCharacteristicPoints[1].getFlattenedPoints();

            double error = 0;
            for (int i = 0; i < basePoints.size(); i++)
            {
                auto basePoint = basePoints[i];
                auto relativePoint = relativePoints[i];
                auto transformed = relativeTransformation * relativePoint;
                std::cout << "Base: " << std::endl << basePoint << std::endl;
                std::cout << "Relative: " << std::endl << relativePoint << std::endl;
                std::cout << "Transformed: " << std::endl << transformed << std::endl;

                error += (basePoint - transformed).norm();
            }
            std::cout << std::endl << std::endl << "****************************************************************************************************************" << std::endl;
            std::cout << "Final error: " << error << std::endl;
        }

        void setupMock() {
            clear();

            Eigen::Matrix4d baseTransformation = generateTransformationMatrix(0, 0, 1, 0, Eigen::Vector3d(0, 1, 0));
            Eigen::Matrix4d relativeTransformation = generateTransformationMatrix(0, 0, 3, M_PI / 2.0, Eigen::Vector3d(0, 1, 0));

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
            
            mockCharacteristicPoints.emplace_back(MockCharacteristicPoints());
            mockCharacteristicPoints.emplace_back(MockCharacteristicPoints());

            for (int i = 0; i < points.size(); i++)
            {
                mockCharacteristicPoints[0].markerCorners[0].emplace_back(baseTransformation * points[i]);
                mockCharacteristicPoints[1].markerCorners[0].emplace_back(relativeTransformation * points[i]);
            }

            expectedTransformations.emplace_back(baseTransformation);
            expectedTransformations.emplace_back(relativeTransformation);
        }

        bool vc::optimization::OptimizationProblem::specific_optimize(std::vector<ACharacteristicPoints> characteristicPoints) {
            long milliseconds = 1000;
            vc::utils::sleepFor(milliseconds);
            return true;
        }
    };
}
#endif