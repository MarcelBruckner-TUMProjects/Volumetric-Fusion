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
#include <mutex>

namespace vc::optimization {
    
    class ACharacteristicPoints{
    public:
        std::map<int, std::vector<Eigen::Vector4d>> markerCorners;
        std::map<int, Eigen::Vector4d> charucoCorners;

        std::vector<glm::vec3> allForRendering;

        ACharacteristicPoints(){}

        ACharacteristicPoints(
            std::map<int, std::vector<Eigen::Vector4d>> markerCorners, std::map<int, Eigen::Vector4d> charucoCorners) :
            markerCorners(markerCorners), charucoCorners(charucoCorners) {}


        std::vector<glm::vec3> getAllVerticesForRendering() {
            return allForRendering;
        }

        unsigned long long hash(int markerId, int cornerId, bool verbose = false) {
            unsigned long long value = std::hash<unsigned long long>()(std::hash<int>()(markerId) + std::hash<int>()(cornerId));

            if (verbose) {
                std::stringstream ss;
                ss << "Hash of " << markerId << ", " << cornerId << ": " << value;
                std::cout << vc::utils::asHeader(ss.str());
            }

            return value;
        }

        template<typename F>
        void iterateAllPoints(F& lambda, bool verbose = false) {
            for (auto& marker : markerCorners)
            {
                for (int i = 0; i < marker.second.size(); i++)
                {
                    lambda(marker.second[i], hash(marker.first, i, verbose));
                }
            }

            for (auto& charucoMarker : charucoCorners)
            {
                lambda(charucoMarker.second, hash(charucoMarker.first, -1, verbose));
            }
        }

        std::map<unsigned long long, Eigen::Vector4d> getFlattenedPoints(bool verbose = false) {
            std::map<unsigned long long, Eigen::Vector4d> flattened;

            iterateAllPoints([&flattened](Eigen::Vector4d& point, unsigned long long hash) {
                flattened[hash] = point;
            }, verbose);

            return flattened;
        }

        std::map<unsigned long long, Eigen::Vector4d> getFilteredPoints(std::vector<unsigned long long> keys, bool verbose = false) {
            std::map<unsigned long long, Eigen::Vector4d> flattened;

            iterateAllPoints([&flattened, &keys](Eigen::Vector4d& point, unsigned long long hash) {
                if (vc::utils::contains(keys, hash)) {
                    flattened[hash] = point;
                }
            }, verbose);

            return flattened;
        }

        std::vector<unsigned long long> getHashes(bool verbose = false) {
            std::vector<unsigned long long> flattened;

            //std::cout << vc::utils::asHeader("New hashset") << std::endl;

            iterateAllPoints([&flattened](Eigen::Vector4d& point, unsigned long long hash) {
                flattened.push_back(hash);
            }, verbose);

            return flattened;
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

        Eigen::Matrix3d cam2World;

        float color2DepthWidth;
        float color2DepthHeight;

        std::mutex mutex;

    public:
        CharacteristicPoints() : ACharacteristicPoints() {}

        CharacteristicPoints(std::shared_ptr<vc::capture::CaptureDevice> pipeline) : ACharacteristicPoints() {
            std::unique_lock<std::mutex> lock(mutex); 
            
            if (!setPipelineStuff(pipeline)) {
                return;
            }

            std::vector<int> ids = pipeline->chArUco->ids;
            std::vector<std::vector<cv::Point2f>> markerCorners = pipeline->chArUco->markerCorners;

            std::vector<int> charucoIds = pipeline->chArUco->charucoIds;
            std::vector<cv::Point2f> charucoCorners = pipeline->chArUco->charucoCorners;

            for (int j = 0; j < ids.size(); j++)
            {
                if (j >= markerCorners.size()) {
                    break;
                }
                int markerId = ids[j];

                for (int cornerId = 0; cornerId < markerCorners[j].size(); cornerId++) {
                    auto point = pixel2Point(markerCorners[j][cornerId]);
                    this->markerCorners[markerId].emplace_back();
                    allForRendering.push_back(glm::vec3(point[0], point[1], point[2]));
                }
            }

            for (int j = 0; j < charucoIds.size(); j++)
            {
                if (j >= charucoCorners.size()) {
                    break;
                }
                int charucoId = charucoIds[j];
                auto point = pixel2Point(charucoCorners[j]);
                this->charucoCorners[charucoId] = point;
                allForRendering.push_back(glm::vec3(point[0], point[1], point[2]));
            }
        }

        Eigen::Vector4d pixel2Point(cv::Point2f observation) {
            try {
                int x = observation.x * color2DepthWidth;
                int y = observation.y * color2DepthHeight;

                Eigen::Vector3d point = Eigen::Vector3d(x * 2.0f, y * 2.0f, 1.0f);

                point = cam2World * point;

                point *= depth_frame->get_distance(x, y);

                Eigen::Vector4d v(point[0], point[1], point[2], 1.0f);
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
            SHADER = new vc::rendering::VertexFragmentShader("shader/characteristic_points.vert", "shader/characteristic_points.frag", "shader/characteristic_points.geom");
            glGenVertexArrays(1, &VAO);
            glGenBuffers(1, &VBO);

            //setVertices();
        }

        int setVertices(ACharacteristicPoints* characteristicPoints) {
            auto vertices = characteristicPoints->getAllVerticesForRendering();

            glBindVertexArray(0);

            glBindVertexArray(VAO);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);

            glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_DYNAMIC_DRAW);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 1 * sizeof(glm::vec3), (void*)0);
            glEnableVertexAttribArray(0);
            //glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)3);
            //glEnableVertexAttribArray(1);

            return vertices.size();
        }

    public:
        CharacteristicPointsRenderer() {
            setupOpenGL();
        }

        void render(ACharacteristicPoints* characteristicPoints, glm::mat4 model, glm::mat4 view, glm::mat4 projection, Eigen::Matrix4d relativeTransformation, float* color) {
            int num_vertices = setVertices(characteristicPoints);

            //std::cout << vc::utils::toString("In render Best Transformation " , relativeTransformation);

            SHADER->use();

            SHADER->setColor("aColor", color[0], color[1], color[2], color[3]);
            SHADER->setFloat("cube_radius", 0.01f);
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