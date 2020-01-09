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
    
    ceres::Matrix glmToCeresMatrix(glm::mat4 matrix, bool verbose = false, std::string name = "") {
        Eigen::Matrix4d final_matrix = Eigen::Map<Eigen::Matrix<glm::f32, 4, 4>>(glm::value_ptr(matrix)).cast<double>();

        if (verbose) {
            std::cout << name << std::endl;
            std::cout << final_matrix;
        }

        return final_matrix;
    }

    class OptimizationProblem {
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

        std::vector<glm::mat4> relativeTransformations = {
            glm::mat4(1.0f),
            glm::mat4(1.0f),
            glm::mat4(1.0f),
            glm::mat4(1.0f)
        };

        void clear() {
            needsRecalculation = true;
            
            for (int i = 0; i < 4; i++)
            {
                translations[i] = (std::vector<double> { 0.0, 0.0, 0.0 });
                rotations[i] = (std::vector<double> { 0.0, 0.0, 0.0 });
                intrinsics[i] = (std::vector<double> { 0.0, 0.0, 0.0, 0.0 });
                distCoeffs[i] = (std::vector<double> { 0.0, 0.0, 0.0, 0.0 });
            }
        }

    public:
        void calculateRelativeTransformations() {
            glm::mat4 baseToMarkerTranslation = getTranslationMatrix(0);
            glm::mat4 baseToMarkerRotation = getRotationMatrix(0);
            //glm::mat4 baseToMarkerRotation = glm::mat4(1.0f);
            //glm::mat4 baseTransformation = baseToMarkerTranslation * glm::inverse(baseToMarkerRotation);

            glm::mat4 baseTransformation = baseToMarkerTranslation * baseToMarkerRotation;

            for (int i = 0; i < translations.size(); i++) {
                //programState.highestFrameIds[i] = MAX(pipelines[i]->chArUco->frameId, programState.highestFrameIds[i]);

                if (i == 0) {
                    //relativeTransformations[i] = glm::inverse(baseToMarkerTranslation);
                    //relativeTransformations[i] = glm::inverse(baseTransformation);
                    relativeTransformations[i] = glm::mat4(1.0f);
                    continue;
                }

                glm::mat4 markerToRelativeTranslation = getTranslationMatrix(i);
                glm::mat4 markerToRelativeRotation = getRotationMatrix(i);
                //glm::mat4 markerToRelativeRotation = glm::mat4(1.0f);

                glm::mat4 relativeTransformation = (
                    //glm::mat4(1.0f)
                    baseTransformation * glm::inverse(markerToRelativeRotation) * glm::inverse(markerToRelativeTranslation)
                    );

                relativeTransformations[i] = relativeTransformation;
            }

            needsRecalculation = false;
        }

        OptimizationProblem() {
            for (int i = 0; i < 4; i++)
            {
                translations.push_back(std::vector<double> { 0.0, 0.0, 0.0 });
                rotations.push_back(std::vector<double> { 0.0, 0.0, 0.0 });
                intrinsics.push_back(std::vector<double> { 0.0, 0.0, 0.0, 0.0 });
                distCoeffs.push_back(std::vector<double> { 0.0, 0.0, 0.0, 0.0 });
            }

            clear();
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

        bool init(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines) {
            clear();

            if (!checkForAllMarkers(pipelines)) {
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

            if (false) 
            {
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
            }

            needsRecalculation = true;

            //calculateRelativeTransformations();
            return true;
        }

        glm::mat4 getRotationMatrix(int camera_index) {
            try {
                cv::Vec3d rotation = cv::Vec3d(
                    rotations.at(camera_index).at(0),
                    rotations.at(camera_index).at(1),
                    rotations.at(camera_index).at(2)
                );
                cv::Matx33d tmp;
                cv::Rodrigues(rotation, tmp);
                auto finalRotation = glm::mat4(
                    tmp.val[0], tmp.val[3], tmp.val[6], 0,
                    tmp.val[1], tmp.val[4], tmp.val[7], 0,
                    tmp.val[2], tmp.val[5], tmp.val[8], 0,
                    0, 0, 0, 1
                );

                return finalRotation;
            }
            catch (std::out_of_range&) {
                return glm::mat4(1.0f);
            }
            catch (std::exception&) {
                return glm::mat4(1.0f);
            }
        }

        glm::mat4 getTranslationMatrix(int camera_index) {
            try {
                return glm::translate(glm::mat4(1.0f), glm::vec3(
                    translations.at(camera_index).at(0),
                    translations.at(camera_index).at(1),
                    translations.at(camera_index).at(2)
                ));
            }
            catch (std::out_of_range&) {
                return glm::mat4(1.0f);
            }
            catch (std::exception&) {
                return glm::mat4(1.0f);
            }
        }

        glm::mat4 getRelativeTransformation(int camera_index) {
            if (needsRecalculation) {
                calculateRelativeTransformations();
            }
            return relativeTransformations[camera_index];
        }

        std::vector<ACharacteristicPoints> getCharacteristicPoints(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines) {
            std::vector<ACharacteristicPoints> characteristicPoints;
            for (int i = 0; i < pipelines.size(); i++)
            {
                characteristicPoints.emplace_back(CharacteristicPoints(pipelines[i]));
            }
            return characteristicPoints;
        }

        bool vc::optimization::OptimizationProblem::optimize(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines, bool onlyOpenCV = false) {
            if (!init(pipelines)) {
                return false;
            }
            if (onlyOpenCV) {
                return true;
            }

            return specific_optimize(getCharacteristicPoints(pipelines));
        }

        virtual bool specific_optimize(std::vector<ACharacteristicPoints> characteristicPoints) = 0;
    };

    class MockOptimizationProblem : public OptimizationProblem {
    public:
        bool vc::optimization::OptimizationProblem::specific_optimize(std::vector<ACharacteristicPoints> characteristicPoints) {
            long milliseconds = 1000;
            vc::utils::sleepFor(milliseconds);
            return true;
        }
    };
}
#endif