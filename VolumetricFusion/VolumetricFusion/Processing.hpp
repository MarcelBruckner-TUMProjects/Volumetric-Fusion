#pragma once


#ifndef _PROCESSING_HEADER_
#define _PROCESSING_HEADER_


#include <librealsense2/hpp/rs_processing.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "Data.hpp"
#include "PinholeCamera.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define _USE_MATH_DEFINES
#include <math.h>
namespace vc::data {
	class Data;
}

namespace vc::processing {

	namespace {
		/**
		Creates a processing block for our OpenCV needs.

		@param lambda a function that takes a reference to a cv::Mat as parameter:
				basic lambda syntax: processing_blocks::createProcessingBlock([](cv::Mat &image){...})
		*/
		template<typename F>
		rs2::processing_block createProcessingBlock(F& lambda, int imageDescriptor, int factor, int frame_id = -1) {
			return rs2::processing_block(
				[=](rs2::frame f, rs2::frame_source& src)
			{
				// For each input frame f, do:

				const int w = f.as<rs2::video_frame>().get_width();
				const int h = f.as<rs2::video_frame>().get_height();

				// frame --> cv
				cv::Mat image(cv::Size(w, h), imageDescriptor, (void*)f.get_data(), cv::Mat::AUTO_STEP);
				// do some (silly) processing

				// Here the magic happens
				lambda(image, f.get_frame_number());

				// Allocate new frame. Copy all missing data from f.
				// This assumes the output is same resolution and format
				// if not true, need to specify parameters explicitly
				auto res = src.allocate_video_frame(f.get_profile(), f);

				// copy from cv --> frame
				memcpy((void*)res.get_data(), image.data, (size_t)(w) * (size_t)(h) * (size_t)(factor));

				// Send the resulting frame to the output queue
				src.frame_ready(res);
			});
		}
	}

	template<typename F>
	rs2::processing_block createColorProcessingBlock(F& lambda) {
		// Don't bother the magic numbers, they describe the image channels
		return createProcessingBlock(lambda, CV_8UC3, 3);
	}

	//template<typename F>
	//rs2::processing_block createDepthProcessingBlock(F& lambda, int frame_id=-1) {
	//	gl// Don't bother the magic numbers, they describe the image channels
	//	return createProcessingBlock(lambda, CV_8UC1, 2);
	//}

	rs2::processing_block createEmpty()
	{
		return rs2::processing_block(
			[&](rs2::frame f, rs2::frame_source& src)
		{
			src.frame_ready(f);
		});
	}


	class ChArUco {
	public:
		rs2::frame_queue charucoProcessingQueues;
		std::shared_ptr<rs2::processing_block > charucoProcessingBlocks;

		// Pose estimation buffers
		// buffer <frame_id, value>
		bool visualize = false;
		bool hasMarkersDetected = false;
		
		std::vector<int> ids;
		std::vector<std::vector<cv::Point2f>> markerCorners;

		std::vector<cv::Point2f> charucoCorners;
		std::vector<int> charucoIds;

		cv::Vec3d rotation, translation;
		
		void startCharucoProcessing(vc::camera::PinholeCamera camera) {
			const auto charucoPoseEstimation = [&camera, this](cv::Mat& image, unsigned long long frameId) {
				cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
				cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(5, 5, 0.04f, 0.02f, dictionary);

				cv::aruco::detectMarkers(image, dictionary, markerCorners, ids);

				/*int i = 0;
				for (auto corner_list : markerCorners) {
					std::cout << ids[i++] << std::endl;
					for (auto corner : corner_list) {
						std::cout << corner << std::endl;
					}
				}*/

				// if at least one marker detected
				if (ids.size() > 0) {
					if (visualize) {
						cv::aruco::drawDetectedMarkers(image, markerCorners, ids);
					}
					cv::aruco::interpolateCornersCharuco(markerCorners, ids, image, board, charucoCorners, charucoIds);
					// if at least one charuco corner detected
					if (charucoIds.size() > 0) {
						if (visualize) {
							cv::aruco::drawDetectedCornersCharuco(image, charucoCorners, charucoIds, cv::Scalar(255, 0, 0));
						}
						bool valid = cv::aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, board, camera.K, camera.distCoeffs, rotation, translation);
						// if charuco pose is valid
						if (valid) {
							if (visualize) {
								cv::aruco::drawAxis(image, camera.K, camera.distCoeffs, rotation, translation, 0.1f);
							}

							hasMarkersDetected = true;
						}
					}
				}
			};

			charucoProcessingBlocks = std::make_shared<rs2::processing_block>(vc::processing::createColorProcessingBlock(charucoPoseEstimation));
			charucoProcessingBlocks->start(charucoProcessingQueues); // Bind output of the processing block to be enqueued into the queue
		}
	};
}

#endif //!_PROCESSING_HEADER_