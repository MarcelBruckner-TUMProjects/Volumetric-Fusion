#pragma once


#ifndef _PROCESSING_HEADER_
#define _PROCESSING_HEADER_


#include <librealsense2/hpp/rs_processing.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Eigen/Dense";

#include "Data.hpp"


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
				memcpy((void*)res.get_data(), image.data, w * h * factor);

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


	class Processing {
	public:
		rs2::frame_queue charucoProcessingQueues;
		std::shared_ptr<rs2::processing_block > charucoProcessingBlocks;

		// Pose estimation buffers
		// buffer <frame_id, value>
		bool hasMarkersDetected = false;
		std::vector<int> charucoIdBuffers;
		Eigen::Matrix4d rotation;
		Eigen::Matrix4d translation;
		void startCharucoProcessing(vc::data::Camera& camera) {
			const auto charucoPoseEstimation = [&camera, this](cv::Mat& image, unsigned long long frameId) {
				cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
				cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(5, 5, 0.04, 0.02, dictionary);
				/*cv::Ptr<cv::aruco::DetectorParameters> params;
				params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_NONE;*/

				std::vector<int> ids;
				std::vector<std::vector<cv::Point2f>> corners;
				cv::aruco::detectMarkers(image, dictionary, corners, ids);
				// if at least one marker detected
				if (ids.size() > 0) {
					cv::aruco::drawDetectedMarkers(image, corners, ids);
					std::vector<cv::Point2f> charucoCorners;
					std::vector<int> charucoIds;
					cv::aruco::interpolateCornersCharuco(corners, ids, image, board, charucoCorners, charucoIds);
					// if at least one charuco corner detected
					if (charucoIds.size() > 0) {
						cv::aruco::drawDetectedCornersCharuco(image, charucoCorners, charucoIds, cv::Scalar(255, 0, 0));
						cv::Vec3d r, t;
						bool valid = cv::aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, board, camera.cameraMatrices, camera.distCoeffs, r, t);
						// if charuco pose is valid
						if (valid) {
							cv::aruco::drawAxis(image, camera.cameraMatrices, camera.distCoeffs, r, t, 0.1);

							charucoIdBuffers = charucoIds;
							Eigen::Matrix4d tmpTranslation;
							tmpTranslation.setIdentity();
							tmpTranslation.block<3, 1>(0, 3) << t[0], t[1], t[2];
							translation = tmpTranslation;

							cv::Matx33d tmp;
							cv::Rodrigues(r, tmp);
							Eigen::Matrix4d tmpRotation;
							tmpRotation.setIdentity();
							tmpRotation.block<3, 3>(0, 0) <<
								tmp.val[0], tmp.val[1], tmp.val[2],
								tmp.val[3], tmp.val[4], tmp.val[5],
								tmp.val[6], tmp.val[7], tmp.val[8];
							rotation = tmpRotation;

							hasMarkersDetected = true;

							//std::stringstream ss;
							//ss << "************************************************************************************" << std::endl;
							//ss << "Device " << i << ", Frame " << frame_id << ":" << std::endl << "Translation: " << std::endl << tmpTranslation << std::endl << "Rotation: " << std::endl << tmpRotation << std::endl;
							//std::cout << ss.str();
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