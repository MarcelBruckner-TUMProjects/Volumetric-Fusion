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

#include "PointCorrespondenceError.hpp"
#include "ReprojectionError.hpp"
#include "../Utils.hpp"
#include "../CaptureDevice.hpp"
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

namespace vc::optimization {

    class ACharacteristicPoints {
    public:
        std::map<int, std::vector<ceres::Vector>> markerCorners;
        std::map<int, ceres::Vector> charucoCorners;

        ACharacteristicPoints(){}

        ACharacteristicPoints(
            std::map<int, std::vector<ceres::Vector>> markerCorners, std::map<int, ceres::Vector> charucoCorners) :
            markerCorners(markerCorners), charucoCorners(charucoCorners) {}
    };

    class MockCharacteristicPoints : public ACharacteristicPoints {
    public:
        MockCharacteristicPoints() : ACharacteristicPoints(){}

        MockCharacteristicPoints(
            std::map<int, std::vector<ceres::Vector>> markerCorners, std::map<int, ceres::Vector> charucoCorners) :
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

        ceres::Vector pixel2Point(cv::Point2f observation) {
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
}

#endif // !_CHARACTERISTIC_POINTS_HEADER
