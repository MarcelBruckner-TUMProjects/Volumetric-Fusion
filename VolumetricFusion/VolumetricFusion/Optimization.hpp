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

#include "PointCorrespondenceError.hpp"
#include "ReprojectionError.hpp"
#include "Utils.hpp"

namespace vc::optimization {

    using namespace ceres;

    class BAProblem {
    private:
        rs2::depth_frame* depth_frame;
        int depth_width;
        int depth_height;

        rs2::frame* color_frame;
        int color_width;
        int color_height;

        glm::mat3 cam2World;

        float color2DepthWidth;
        float color2DepthHeight;

    public:
        const int num_translation_parameters = 3;
        const int num_rotation_parameters = 3;
        const int num_intrinsic_parameters = 4;
        const int num_distCoeff_parameters = 4;

        std::vector<std::vector<double>> translations;
        std::vector<std::vector<double>> rotations;
        std::vector<std::vector<double>> intrinsics;
        std::vector<std::vector<double>> distCoeffs;

        bool hasSolution = false;

        std::vector<glm::mat4> relativeTransformations = {
            glm::mat4(1.0f),
            glm::mat4(1.0f),
            glm::mat4(1.0f),
            glm::mat4(1.0f)
        };

        BAProblem() {
        }

        ~BAProblem() {
        }

        void clear() {
            translations.clear();
            rotations.clear();
            intrinsics.clear();
            distCoeffs.clear();
        }

        bool init(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines) {
            clear();

            if (pipelines.size() == 0 || !checkForAllMarkers(pipelines)) {
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

                //intrinsics[i * num_intrinsic_parameters + 0] = (double)pipelines[i]->depth_camera->intrinsics.fx;
                //intrinsics[i * num_intrinsic_parameters + 1] = (double)pipelines[i]->depth_camera->intrinsics.fy;
                //intrinsics[i * num_intrinsic_parameters + 2] = (double)pipelines[i]->depth_camera->intrinsics.ppx;
                //intrinsics[i * num_intrinsic_parameters + 3] = (double)pipelines[i]->depth_camera->intrinsics.ppy;
                //for (int j = 0; j < num_distCoeff_parameters; j++)
                //{
                //    distCoeffs[i * num_distCoeff_parameters + j] = (double)pipelines[i]->depth_camera->distCoeffs[j];
                //}

                translations.emplace_back(translation);
                rotations.emplace_back(rotation);
            }

            //translations[1][0] += 1;

            hasSolution = true;

            calculateRelativeTransformations(pipelines.size());
            return true;
        }

        bool optimize(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines, bool reprojectionError = false) {
            /*  hasSolution = true;
              return true;*/


            hasSolution = false;
            if (!init(pipelines)) {
                return false;
            }


            if (reprojectionError) {
                solveReprojectionError(pipelines);
            }
            else {
                if (!solvePointCorrespondenceError(pipelines)) {
                    return false;
                }
            }

            vc::utils::sleepFor("Pre recalculation after optimization", 4000);
            calculateRelativeTransformations(pipelines.size());
            vc::utils::sleepFor("After recalculation after optimization", 4000);

            return true;
        }

        glm::mat4 getRotationMatrix(int camera_index) {
            if (!hasSolution) {
                return glm::mat4(1.0f);
            }
            /*glm::vec3 rotation = glm::vec3(
                rotations[camera_index][0],
                rotations[camera_index][1],
                rotations[camera_index][2]
            );*/
            cv::Vec3d rotation = cv::Vec3d(
                rotations[camera_index][0],
                rotations[camera_index][1],
                rotations[camera_index][2]
            );
            //rotation = glm::vec3(0.0f, 1.0f, 0.0f) * glm::radians(90.0f);
            /*float length = glm::length(rotation);
            glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), length, glm::normalize(rotation));
            return rotationMatrix;*/
            //cv::Vec3d rotation = cv::Vec3d(&rotations[camera_index * num_rotation_parameters]);
            cv::Matx33d tmp;
            cv::Rodrigues(rotation, tmp);
            //auto finalRotation = glm::mat4(
            //    tmp.val[0], tmp.val[1], tmp.val[2], 0,
            //    tmp.val[3], tmp.val[4], tmp.val[5], 0,
            //    tmp.val[6], tmp.val[7], tmp.val[8], 0,
            //    0, 0, 0, 1
            //);
            auto finalRotation = glm::mat4(
                tmp.val[0], tmp.val[3], tmp.val[6], 0,
                tmp.val[1], tmp.val[4], tmp.val[7], 0,
                tmp.val[2], tmp.val[5], tmp.val[8], 0,
                0, 0, 0, 1
            );

            return finalRotation;
        }

        glm::mat4 getTranslationMatrix(int camera_index) {
            if (!hasSolution) {
                return glm::mat4(1.0f);
            }
            return glm::translate(glm::mat4(1.0f), glm::vec3(
                translations[camera_index][0],
                translations[camera_index][1],
                translations[camera_index][2]
            ));
        }

        glm::vec4 pixel2Point(cv::Point2f observation) {
            int x = observation.x * color2DepthWidth;
            int y = observation.y * color2DepthHeight;

            glm::vec3 point = glm::vec3(x, y, 1.0f);
            point = point * cam2World;
            point *= depth_frame->get_distance(x, y);

            return glm::vec4(point.x, point.y, point.z, 1.0f);
        }

        bool setPipelineStuff(std::shared_ptr<vc::capture::CaptureDevice> pipe) {
            try {
                depth_frame = (rs2::depth_frame*) & pipe->data->filteredDepthFrames;
                depth_width = depth_frame->as<rs2::video_frame>().get_width();
                depth_height = depth_frame->as<rs2::video_frame>().get_height();

                color_frame = &pipe->data->filteredColorFrames;
                color_width = color_frame->as<rs2::video_frame>().get_width();
                color_height = color_frame->as<rs2::video_frame>().get_height();

                cam2World = pipe->depth_camera->cam2world;

                color2DepthWidth = 1.0f * depth_width / color_width;
                color2DepthHeight = 1.0f * depth_height / color_height;

                return true;
            }
            catch (rs2::error & e) {
                return false;
            }
        }

        //std::vector<std::vector<double>>  calculateTranslationErrors(std::vector<std::vector<double>> initialTranslations) {
        //    std::vector<std::vector<double>> errors;

        //    for (int i = 0; i < initialTranslations.size(); i++)
        //    {
        //        std::vector<double> error;
        //        for (int j = 0; j < initialTranslations[i].size(); j++)
        //        {
        //            error.emplace_back(initialTranslations[i][j] - translations[i][j]);
        //        }
        //        errors.emplace_back(error);
        //    }
        //    return errors;
        //}

        //std::vector<std::vector<double>>  calculateRotationErrors(std::vector<std::vector<double>> initialRotations) {
        //    std::vector<std::vector<double>> errors;

        //    for (int i = 0; i < initialRotations.size(); i++)
        //    {
        //        std::vector<double> error;
        //        for (int j = 0; j < initialRotations[i].size(); j++)
        //        {
        //            error.emplace_back(initialRotations[i][j] - rotations[i][j]);
        //        }
        //        errors.emplace_back(error);
        //    }
        //    return errors;
        //}

        void solveProblem(ceres::Problem* problem) {

            auto initialTranslations = translations;
            auto initialRotations = rotations;
            
            ceres::Solver::Options options;
            options.num_threads = 8;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = true;
            options.max_num_iterations = 500;
            ceres::Solver::Summary summary;
            ceres::Solve(options, problem, &summary);
            std::cout << summary.FullReport() << "\n";

            std::cout << vc::utils::toString("Initial", initialTranslations, initialRotations);
            std::cout << vc::utils::toString("Final", translations, rotations);
            //std::cout << vc::utils::toString("Error", calculateTranslationErrors(initialTranslations), calculateRotationErrors(initialRotations));

            std::cout << std::endl;
        }

        bool solvePointCorrespondenceError(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines) {

            // Create residuals for each observation in the bundle adjustment problem. The
            // parameters for cameras and points are added automatically.
            ceres::Problem problem;

            glm::mat4 baseToMarkerTranslation = getTranslationMatrix(0);
            glm::mat4 baseToMarkerRotation = getRotationMatrix(0);
            //glm::mat4 baseToMarkerRotation = glm::mat4(1.0f);
            glm::mat4 baseTransformation = glm::inverse(baseToMarkerTranslation) * glm::inverse(baseToMarkerRotation);
            //glm::mat4 baseTransformation = baseToMarkerTranslation * glm::mat4(1.0f);
            glm::f32* baseTrans = glm::value_ptr(baseTransformation);

            std::map<int, std::vector<glm::vec4>> baseMarkerCorners;
            std::map<int, glm::vec4> baseCharucoCorners;

            int o = 0;
            for (int i = 0; i < pipelines.size(); i++) {
                auto pipe = pipelines[i];

                //glm::mat4 markerToRelativeRotation = getRotationMatrix(i);
                //glm::mat4 markerToRelativeTranslation = getTranslationMatrix(i);

                if (!setPipelineStuff(pipe)) {
                    return false;
                }

                std::vector<int> ids = pipelines[i]->chArUco->ids;
                std::vector<std::vector<cv::Point2f>> markerCorners = pipelines[i]->chArUco->markerCorners;

                std::vector<cv::Point2f> charucoCorners = pipelines[i]->chArUco->charucoCorners;
                std::vector<int> charucoIds = pipelines[i]->chArUco->charucoIds;

                for (int j = 0; j < ids.size(); j++)
                {
                    int markerId = ids[j];

                    for (int cornerId = 0; cornerId < markerCorners[j].size(); cornerId++) {
                        glm::vec4 relativeFramePoint = pixel2Point(markerCorners[j][cornerId]);

                        if (i == 0) {
                            //baseMarkerCorners[id].emplace_back(inverseBaseTransformation * point);
                            baseMarkerCorners[markerId].emplace_back(relativeFramePoint);
                        }
                        else {

                            // TODO:
                            // Not sure if the correspondence matching is correct, so if the ids from charuco markers
                            // are correctly taken and matched with the charuco markers in the other camera frame
                            if (baseMarkerCorners.count(markerId) <= 0) {
                                continue;
                            }
                            auto baseFramePoint = baseMarkerCorners[markerId][cornerId];
                            //std::cout << vc::utils::toString(baseFramePoint, relativeFramePoint);
                            ceres::CostFunction* cost_function = PointCorrespondenceError::Create(
                                markerId, cornerId, relativeFramePoint, baseFramePoint, baseTrans
                            );
                            problem.AddResidualBlock(cost_function, NULL,
                                translations[i].data(),
                                rotations[i].data()
                            );
                        }
                    }
                }

                for (int j = 0; j < charucoIds.size(); j++)
                {
                    int charucoId = charucoIds[j];

                    glm::vec4 relativeFramePoint = pixel2Point(charucoCorners[j]);

                    if (i == 0) {
                        //baseCharucoCorners[id] = inverseBaseTransformation * point;
                        baseCharucoCorners[charucoId] = relativeFramePoint;
                    }
                    else {
                        if (baseCharucoCorners.count(charucoId) <= 0) {
                            continue;
                        }
                        auto baseFramePoint = baseCharucoCorners[charucoId];
                        //std::cout << vc::utils::toString(baseFramePoint, relativeFramePoint);
                        ceres::CostFunction* cost_function = PointCorrespondenceError::Create(
                            charucoId, -1, relativeFramePoint, baseFramePoint, baseTrans
                        );
                        problem.AddResidualBlock(cost_function, NULL,
                            translations[i].data(),
                            rotations[i].data()
                        );
                    }
                }
            }

            solveProblem(&problem);
            return true;
        }

        void solveReprojectionError(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines) {

            // Create residuals for each observation in the bundle adjustment problem. The
            // parameters for cameras and points are added automatically.
            ceres::Problem problem;

            for (int i = 0; i < pipelines.size(); i++) {
                auto pipe = pipelines[i];

                setPipelineStuff(pipe);

                for (auto markerCorners : pipe->chArUco->markerCorners) {
                    for (auto observation : markerCorners) {
                        glm::vec4 point = pixel2Point(observation);
                        //point = relativeTransformations[i] * point;
                        //point = vc::rendering::COORDINATE_CORRECTION * relativeTransformations[i] * point;

                        //ceres::CostFunction* cost_function = ReprojectionError::Create(observation, point, intrinsics, distCoeffs);
                        //problem.AddResidualBlock(cost_function, NULL,
                        //    translations[i].data(),
                        //    rotations[i].data()
                        //    //intrinsics + i * num_intrinsic_parameters,
                        //    //distCoeffs + i * num_distCoeff_parameters
                        //);
                    }
                }

                for (auto observation : pipe->chArUco->charucoCorners) {
                    glm::vec4 point = pixel2Point(observation);
                    //point = relativeTransformations[i] * point;
                    //point = vc::rendering::COORDINATE_CORRECTION * relativeTransformations[i] * point;

                    //ceres::CostFunction* cost_function = ReprojectionError::Create(observation, point, intrinsics, distCoeffs);
                    //problem.AddResidualBlock(cost_function, NULL,
                    //    translations[i].data(),
                    //    rotations[i].data()
                    //    //intrinsics + i * num_intrinsic_parameters,
                    //    //distCoeffs + i * num_distCoeff_parameters
                    //);
                }
            }

            solveProblem(&problem);
        }

        bool checkForAllMarkers(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines) {
            for (auto pipe : pipelines) {
                if (!pipe->chArUco->hasMarkersDetected) {
                    return false;
                }
            }
            return true;
        }

        void calculateRelativeTransformations(int num_pipelines) {
            glm::mat4 baseToMarkerTranslation = getTranslationMatrix(0);
            glm::mat4 baseToMarkerRotation = getRotationMatrix(0);
            //glm::mat4 baseToMarkerRotation = glm::mat4(1.0f);
            glm::mat4 baseTransformation = baseToMarkerTranslation * glm::inverse(baseToMarkerRotation);

            //glm::mat4 baseTransformation = ((baseToMarkerRotation) * (baseToMarkerTranslation));

            for (int i = 0; i < num_pipelines; i++) {
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

                    //baseToMarkerTranslation * (markerToRelativeRotation) * (baseToMarkerRotation) * glm::inverse(markerToRelativeTranslation)
                    //baseToMarkerTranslation * glm::inverse(markerToRelativeRotation) * (baseToMarkerRotation) * glm::inverse(markerToRelativeTranslation)
                    //baseToMarkerTranslation * (markerToRelativeRotation) * glm::inverse(baseToMarkerRotation) * glm::inverse(markerToRelativeTranslation)
                    //baseToMarkerTranslation * glm::inverse(markerToRelativeRotation) * glm::inverse(baseToMarkerRotation) * glm::inverse(markerToRelativeTranslation) 

                    //baseToMarkerTranslation * (baseToMarkerRotation) * (markerToRelativeRotation) * glm::inverse(markerToRelativeTranslation)
                    //baseToMarkerTranslation * glm::inverse((baseToMarkerRotation) * glm::inverse(markerToRelativeRotation)) * glm::inverse(markerToRelativeTranslation)
                    ///*baseToMarkerTranslation **/ glm::inverse(baseToMarkerRotation)* (markerToRelativeRotation)*glm::inverse(markerToRelativeTranslation) //######################################################################
                    baseTransformation * (markerToRelativeRotation)*glm::inverse(markerToRelativeTranslation) //######################################################################
                    //baseToMarkerTranslation * glm::inverse(baseToMarkerRotation) * (markerToRelativeRotation) * glm::inverse(markerToRelativeTranslation) //######################################################################
                    //baseToMarkerTranslation * glm::inverse(baseToMarkerRotation) * glm::inverse(markerToRelativeRotation) * glm::inverse(markerToRelativeTranslation)

                    //glm::inverse(markerToRelativeTranslation * markerToRelativeRotation) * baseToMarkerTranslation * baseToMarkerRotation
                    //glm::inverse(markerToRelativeTranslation * markerToRelativeRotation) * baseToMarkerRotation * baseToMarkerTranslation
                    //glm::inverse(markerToRelativeRotation * markerToRelativeTranslation) * baseToMarkerTranslation * baseToMarkerRotation
                    //glm::inverse(markerToRelativeRotation * markerToRelativeTranslation) * baseToMarkerRotation * baseToMarkerTranslation

                    //baseToMarkerTranslation * baseToMarkerRotation * glm::inverse(markerToRelativeTranslation * markerToRelativeRotation)
                    //baseToMarkerRotation * baseToMarkerTranslation * glm::inverse(markerToRelativeTranslation * markerToRelativeRotation)
                    //baseToMarkerTranslation * baseToMarkerRotation * glm::inverse(markerToRelativeRotation * markerToRelativeTranslation)
                    //baseToMarkerRotation * baseToMarkerTranslation * glm::inverse(markerToRelativeRotation * markerToRelativeTranslation)

                    //markerToRelativeTranslation * markerToRelativeRotation* glm::inverse(baseToMarkerTranslation * baseToMarkerRotation) 
                    //markerToRelativeTranslation * markerToRelativeRotation* glm::inverse(baseToMarkerRotation * baseToMarkerTranslation) 
                    //markerToRelativeRotation * markerToRelativeTranslation* glm::inverse(baseToMarkerTranslation * baseToMarkerRotation) 
                    //markerToRelativeRotation * markerToRelativeTranslation* glm::inverse(baseToMarkerRotation * baseToMarkerTranslation) 

                    //glm::inverse(baseToMarkerTranslation * baseToMarkerRotation) * markerToRelativeTranslation * markerToRelativeRotation
                    //glm::inverse(baseToMarkerRotation * baseToMarkerTranslation) * markerToRelativeTranslation * markerToRelativeRotation
                    //glm::inverse(baseToMarkerTranslation * baseToMarkerRotation) * markerToRelativeRotation * markerToRelativeTranslation
                    //glm::inverse(baseToMarkerRotation * baseToMarkerTranslation) * markerToRelativeRotation * markerToRelativeTranslation
                    );

                relativeTransformations[i] = relativeTransformation;
            }
        }

        void test() {
            clear();
            hasSolution = true;

            glm::vec3 baseTranslationVector = glm::vec3(0.0f, 0.0f, 1.0f);
            translations.emplace_back(std::vector<double>{ baseTranslationVector.x, baseTranslationVector.y, baseTranslationVector.z });
            glm::mat4 baseTranslation = getTranslationMatrix(0);

            glm::vec3 relativeTranslationVector = glm::vec3(0.0f, 0.0f, 3.0f);
            translations.emplace_back(std::vector<double>{ relativeTranslationVector.x, relativeTranslationVector.y, relativeTranslationVector.z });
            glm::mat4 relativeTranslation = getTranslationMatrix(1);
            translations[1][0] -= 3;
            translations[1][1] += 1;

            glm::vec3 baseAxis = glm::vec3(0.0f, 1.0f, 0.0f);
            baseAxis = glm::normalize(baseAxis);
            float baseAngle = glm::radians(00.0f);
            baseAxis *= baseAngle;
            rotations.emplace_back(std::vector<double>{baseAxis.x, baseAxis.y, baseAxis.z});
            glm::mat4 baseRotation = getRotationMatrix(0);
            glm::mat4 baseTransformation = baseTranslation * baseRotation;


            glm::vec3 relativeAxis = glm::vec3(0.0f, 1.0f, 0.0f);
            relativeAxis = glm::normalize(relativeAxis);
            float relativeAngle = glm::radians(90.0f);
            relativeAxis *= relativeAngle;
            rotations.emplace_back(std::vector<double>{relativeAxis.x, relativeAxis.y, relativeAxis.z});
            glm::mat4 relativeRotation = getRotationMatrix(1);
            glm::mat4 relativeTransformation = relativeTranslation * relativeRotation;
            rotations[1][0] -= 1;
            rotations[1][1] += 0.5;

            std::vector<std::vector<double>> expectedTranslations;
            expectedTranslations.emplace_back(std::vector<double>{baseTranslationVector.x, baseTranslationVector.y, baseTranslationVector.z});
            expectedTranslations.emplace_back(std::vector<double>{relativeTranslationVector.x, relativeTranslationVector.y, relativeTranslationVector.z});

            std::vector<std::vector<double>> expectedRotations;
            expectedRotations.emplace_back(std::vector<double>{baseAxis.x, baseAxis.y, baseAxis.z});
            expectedRotations.emplace_back(std::vector<double>{relativeAxis.x, relativeAxis.y, relativeAxis.z});


            std::vector<glm::vec4> points;
            points.emplace_back(glm::vec4(1.0f, 1.0f, 0.0f, 1.0f));
            points.emplace_back(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
            points.emplace_back(glm::vec4(1.0f, -1.0f, 0.0f, 1.0f));
            points.emplace_back(glm::vec4(-1.0f, 1.0f, 0.0f, 1.0f));
            points.emplace_back(glm::vec4(-1.0f, -1.0f, 0.0f, 1.0f));
            points.emplace_back(glm::vec4(0.5f, 0.5f, 0.0f, 1.0f));
            points.emplace_back(glm::vec4(0.5f, -0.5f, 0.0f, 1.0f));
            points.emplace_back(glm::vec4(-0.5f, 0.5f, 0.0f, 1.0f));
            points.emplace_back(glm::vec4(-0.5f, -0.5f, 0.0f, 1.0f));


            std::vector<glm::vec4> baseFramePoints;
            std::vector<glm::vec4> relativeFramePoints;
            for (int i = 0; i < points.size(); i++)
            {
                baseFramePoints.emplace_back(baseTransformation * points[i]);
                relativeFramePoints.emplace_back(relativeTransformation * points[i]);

            }

            mockOptimize(baseFramePoints, relativeFramePoints, glm::inverse(baseTranslation) * glm::inverse(baseRotation));
            
            std::cout << vc::utils::toString("Expected", expectedTranslations, expectedRotations);

            std::cout << std::endl;
        }

        bool mockOptimize(std::vector<glm::vec4> baseFramePoints, std::vector<glm::vec4> relativeFramePoints, glm::mat4 baseTransformation) {
            // Create residuals for each observation in the bundle adjustment problem. The
            // parameters for cameras and points are added automatically.
            ceres::Problem problem;

            int o = 0;
            for (int i = 0; i < baseFramePoints.size(); i++)
            {
                ceres::CostFunction* cost_function = PointCorrespondenceError::Create(
                    o++, 0, relativeFramePoints[i], baseFramePoints[i], glm::value_ptr(baseTransformation)
                );
                problem.AddResidualBlock(cost_function, NULL,
                    translations[1].data(),
                    rotations[1].data()
                );
            }

            solveProblem(&problem);
            return true;
        }
    };
}

#endif