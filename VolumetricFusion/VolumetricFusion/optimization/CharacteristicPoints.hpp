#pragma once
#ifndef _CHARACTERISTIC_POINTS_HEADER
#define _CHARACTERISTIC_POINTS_HEADER

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

#include "../Rendering.hpp"
#include "PointCorrespondenceError.hpp"
#include "ReprojectionError.hpp"
#include "../Utils.hpp"
#include "../CaptureDevice.hpp"
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

namespace vc::optimization {
    
    class ACharacteristicPoints {
    public:
        std::map<int, std::vector<Eigen::Vector4d>> markerCorners;
        std::map<int, Eigen::Vector4d> charucoCorners;

        ACharacteristicPoints(){}

        ACharacteristicPoints(
            std::map<int, std::vector<Eigen::Vector4d>> markerCorners, std::map<int, Eigen::Vector4d> charucoCorners) :
            markerCorners(markerCorners), charucoCorners(charucoCorners) {}


        std::vector<float> getAllVerticesForRendering(glm::vec3 color = glm::vec3(-1.0f, -1.0f, -1.0f)) {
            std::vector<float> vertices;

            for (auto& marker : markerCorners)
            {
                for (auto& markerCorner : marker.second)
                {
                    vertices.emplace_back(markerCorner[0]);
                    vertices.emplace_back(markerCorner[1]);
                    vertices.emplace_back(markerCorner[2]);

                    if (color.x < 0) {
                        for (int i = 0; i < 3; i++)
                        {
                            vertices.emplace_back((std::rand() % 1000) / 1000.0f);
                        }
                    }
                    else {
                        vertices.emplace_back(color.x);
                        vertices.emplace_back(color.y);
                        vertices.emplace_back(color.z);
                    }
                }
            }

            for (auto& charucoMarker : charucoCorners)
            {
                vertices.emplace_back(charucoMarker.second[0]);
                vertices.emplace_back(charucoMarker.second[1]);
                vertices.emplace_back(charucoMarker.second[2]);

                if (color.x < 0) {
                    for (int i = 0; i < 3; i++)
                    {
                        vertices.emplace_back((std::rand() % 1000) / 1000.0f);
                    }
                }
                else {
                    vertices.emplace_back(color.x);
                    vertices.emplace_back(color.y);
                    vertices.emplace_back(color.z);
                }
            }

            return vertices;
        }

        template<typename F>
        void iterateAllPoints(F& lambda) {
            for (auto& marker : markerCorners)
            {
                for (auto& markerCorner : marker.second)
                {
                    //std::cout << Eigen::Vector4d(markerCorner) << std::endl;
                    lambda(markerCorner);
                }
            }

            for (auto& charucoMarker : charucoCorners)
            {
                lambda(charucoMarker.second);
            }
        }

        std::vector<Eigen::Vector4d> getFlattenedPoints() {
            std::vector<Eigen::Vector4d> flattened;

            iterateAllPoints([&flattened](Eigen::Vector4d point) {
                flattened.emplace_back(point);
            });

            return flattened;
        }

        Eigen::Vector4d getCenterOfGravity(bool verbose = true) {
            Eigen::Vector4d centerOfGravity(0,0,0,0);

            iterateAllPoints([&centerOfGravity](Eigen::Vector4d point) {
                centerOfGravity += point;
            });
            
            centerOfGravity /= getNumberOfPoints();

            if (verbose) {
                std::cout << "Center of gravity:" << std::endl << centerOfGravity << std::endl;
            }
            return centerOfGravity;
        }

        float getAverageDistance() {
            double distance = 0;
            Eigen::Vector4d centerOfGravity = getCenterOfGravity(false);

            iterateAllPoints([&distance, &centerOfGravity](Eigen::Vector4d point) {
                distance += Eigen::Vector4d(centerOfGravity - point).norm();
            });
            
            distance /= getNumberOfPoints();

            std::cout << "Average distance:" << std::endl << distance << std::endl;

            return distance;
        }

        int getNumberOfPoints() {
            int num_points = 0;
            for (auto& marker : markerCorners)
            {
                num_points += marker.second.size();
            }
            num_points += charucoCorners.size();
            return num_points;
        }

        Eigen::Matrix4d estimateRotation(const std::vector<Eigen::Vector4d>& targetPoints, const Eigen::Vector4d& targetMean) {
            // TODO: Estimate the rotation from source to target points, following the Procrustes algorithm.
            // To compute the singular value decomposition you can use JacobiSVD() from Eigen.
            // Important: The covariance matrices should contain mean-centered source/target points.
            Eigen::Vector4d sourceMean = getCenterOfGravity(false);
            std::vector<Eigen::Vector4d>& sourcePoints = getFlattenedPoints();

            std::vector<Eigen::Vector4d> meanCenteredSourcePoints = getFlattenedPoints();
            std::vector<Eigen::Vector4d> meanCenteredTargetPoints = targetPoints;

            std::cout << "Relative points:" << std::endl << vc::utils::toString(meanCenteredSourcePoints, "\n\n") << std::endl << std::endl;
            std::cout << "Base points:" << std::endl << vc::utils::toString(meanCenteredTargetPoints, "\n\n");

            std::transform(meanCenteredSourcePoints.begin(), meanCenteredSourcePoints.end(), meanCenteredSourcePoints.begin(), [sourceMean](Eigen::Vector4d point) -> Eigen::Vector4d { return point - sourceMean; });
            std::transform(meanCenteredTargetPoints.begin(), meanCenteredTargetPoints.end(), meanCenteredTargetPoints.begin(), [targetMean](Eigen::Vector4d point) -> Eigen::Vector4d { return point - targetMean; });


            std::cout << "Mean centered Relative points:" << std::endl << vc::utils::toString(meanCenteredSourcePoints, "\n\n") << std::endl << std::endl;
            std::cout << "Mean centered Base points:" << std::endl << vc::utils::toString(meanCenteredTargetPoints, "\n\n");

            Eigen::Matrix4d B = Eigen::Matrix4d();
            B << meanCenteredSourcePoints[4] - meanCenteredSourcePoints[0], meanCenteredSourcePoints[3] - meanCenteredSourcePoints[0], meanCenteredSourcePoints[2] - meanCenteredSourcePoints[0], meanCenteredSourcePoints[1] - meanCenteredSourcePoints[0];
            Eigen::Matrix4d A = Eigen::Matrix4d();
            A << meanCenteredTargetPoints[4] - meanCenteredTargetPoints[0], meanCenteredTargetPoints[3] - meanCenteredTargetPoints[0], meanCenteredTargetPoints[2] - meanCenteredTargetPoints[0], meanCenteredTargetPoints[1] - meanCenteredTargetPoints[0];

            Eigen::JacobiSVD<Eigen::Matrix4d> svd = Eigen::JacobiSVD<Eigen::Matrix4d>(B * A.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);

            auto result = Eigen::Matrix4d(svd.matrixU() * svd.matrixV().transpose());

            std::cout << "Rotation: " << std::endl << result << std::endl;

            return result;
        }
    };
     
    class MockCharacteristicPoints : public ACharacteristicPoints {
    public:
        MockCharacteristicPoints() : ACharacteristicPoints(){}

        MockCharacteristicPoints(
            std::map<int, std::vector<Eigen::Vector4d>> markerCorners, std::map<int, Eigen::Vector4d> charucoCorners) :
            ACharacteristicPoints(markerCorners, charucoCorners) {}
    };

    class CharacteristicPoints : public ACharacteristicPoints {
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
        CharacteristicPoints(std::shared_ptr<vc::capture::CaptureDevice> pipeline) : ACharacteristicPoints(){
            if (!setPipelineStuff(pipeline)) {
                return;
            }

            std::vector<int> ids = pipeline->chArUco->ids;
            std::vector<std::vector<cv::Point2f>> markerCorners = pipeline->chArUco->markerCorners;

            std::vector<cv::Point2f> charucoCorners = pipeline->chArUco->charucoCorners;
            std::vector<int> charucoIds = pipeline->chArUco->charucoIds;

            for (int j = 0; j < ids.size(); j++)
            {
                int markerId = ids[j];

                for (int cornerId = 0; cornerId < markerCorners[j].size(); cornerId++) {
                        this->markerCorners[markerId].emplace_back(pixel2Point(markerCorners[j][cornerId]));
                }
            }

            for (int j = 0; j < charucoIds.size(); j++)
            {
                int charucoId = charucoIds[j];

                    this->charucoCorners[charucoId] = pixel2Point(charucoCorners[j]);
            }
        }

        Eigen::Vector4d pixel2Point(cv::Point2f observation) {
            try {
                int x = observation.x * color2DepthWidth;
                int y = observation.y * color2DepthHeight;

                glm::vec3 point = glm::vec3(x, y, 1.0f);
                point = point * cam2World;
                point *= depth_frame->get_distance(x, y);

                Eigen::Vector4d v(point.x, point.y, point.z, 1.0f);
                return v;
            }
            catch (rs2::error&) {
                return Eigen::Vector4d(0.0, 0.0, 0.0, 0.0);
            }
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
    };

    class CharacteristicPointsRenderer {
    private:
        unsigned int VBO, VAO;
        vc::rendering::Shader* SHADER;
        
        void setupOpenGL() {
            SHADER = new vc::rendering::VertexFragmentShader("shader/characteristic_points.vert", "shader/characteristic_points.frag");
            glGenVertexArrays(1, &VAO);
            glGenBuffers(1, &VBO);

            //setVertices();
        }

        int setVertices(ACharacteristicPoints* characteristicPoints, glm::vec3 color = glm::vec3(-1.0f, -1.0f, -1.0f)) {
            auto vertices = characteristicPoints->getAllVerticesForRendering(color);

            glBindVertexArray(0);

            glBindVertexArray(VAO);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);

            glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)3);
            glEnableVertexAttribArray(1);

            return vertices.size() / 6;
        }

    public:
        CharacteristicPointsRenderer() {
            setupOpenGL();
        }

        void render(ACharacteristicPoints* characteristicPoints, glm::mat4 model, glm::mat4 view, glm::mat4 projection, glm::mat4 relativeTransformation = glm::mat4(1.0f), glm::vec3 color = glm::vec3(-1.0f, -1.0f, -1.0f)) {

            int num_vertices = setVertices(characteristicPoints, color);

            SHADER->use();

            SHADER->setMat4("relativeTransformation", relativeTransformation);
            SHADER->setMat4("correction", vc::rendering::COORDINATE_CORRECTION);
            SHADER->setMat4("model", model);
            SHADER->setMat4("view", view);
            SHADER->setMat4("projection", projection);

            glBindVertexArray(VAO);

            glDrawArrays(GL_POINTS, 0, num_vertices);

            glBindVertexArray(0);
        }
    };
}

#endif // !_CHARACTERISTIC_POINTS_HEADER
