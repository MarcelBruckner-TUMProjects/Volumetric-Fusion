#pragma once
//#include <librealsense2/hpp/rs_processing.hpp>
//
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/aruco/charuco.hpp>

namespace vc::processing_blocks {

	//namespace {
	//	/**
	//	Creates a processing block for our OpenCV needs.

	//	@param lambda a function that takes a reference to a cv::Mat as parameter:
	//			basic lambda syntax: processing_blocks::createProcessingBlock([](cv::Mat &image){...})
	//	*/
	//	template<typename F>
	//	rs2::processing_block createProcessingBlock(F& lambda, int imageDescriptor, int factor, 
	//		cv::Matx33f& cameraMatrix,
	//		std::vector<float> distortionCoefficients,
	//		std::map<unsigned long long, std::vector<int>>& charucoIdBuffer,
	//		std::map<unsigned long long, cv::Vec3d>& rotationBuffer,
	//		std::map<unsigned long long, cv::Vec3d>& translationBuffer) {
	//		return rs2::processing_block(
	//			[=](rs2::frame f, rs2::frame_source& src)
	//		{
	//			// For each input frame f, do:

	//			const int w = f.as<rs2::video_frame>().get_width();
	//			const int h = f.as<rs2::video_frame>().get_height();

	//			// frame --> cv
	//			cv::Mat image(cv::Size(w, h), imageDescriptor, (void*)f.get_data(), cv::Mat::AUTO_STEP);
	//			// do some (silly) processing

	//			// Here the magic happens
	//			lambda(image, f.get_frame_number(),
	//				cameraMatrix,
	//				distortionCoefficients,
	//				charucoIdBuffer,
	//				rotationBuffer,
	//				translationBuffer);

	//			// Allocate new frame. Copy all missing data from f.
	//			// This assumes the output is same resolution and format
	//			// if not true, need to specify parameters explicitly
	//			auto res = src.allocate_video_frame(f.get_profile(), f);

	//			// copy from cv --> frame
	//			memcpy((void*)res.get_data(), image.data, w * h * factor);

	//			// Send the resulting frame to the output queue
	//			src.frame_ready(res);
	//		});
	//	}
	//}

	//template<typename F>
	//rs2::processing_block createColorProcessingBlock(F& lambda,
	//	cv::Matx33f& cameraMatrix,
	//	std::vector<float> distortionCoefficients,
	//	std::map<unsigned long long, std::vector<int>>& charucoIdBuffer,
	//	std::map<unsigned long long, cv::Vec3d>& rotationBuffer,
	//	std::map<unsigned long long, cv::Vec3d>& translationBuffer) {
	//	// Don't bother the magic numbers, they describe the image channels
	//	return createProcessingBlock(lambda, CV_8UC3, 3,
	//		cameraMatrix,
	//		distortionCoefficients,
	//		charucoIdBuffer,
	//		rotationBuffer,
	//		translationBuffer);
	//}

	////template<typename F>
	////rs2::processing_block createDepthProcessingBlock(F& lambda, int frame_id=-1) {
	////	// Don't bother the magic numbers, they describe the image channels
	////	return createProcessingBlock(lambda, CV_8UC1, 2);
	////}

	//rs2::processing_block createEmpty()
	//{
	//	return rs2::processing_block(
	//		[&](rs2::frame f, rs2::frame_source& src)
	//	{
	//		src.frame_ready(f);
	//	});
	//}


	///// <summary>
	///// The color processing block
	///// </summary>
	//rs2::processing_block charucoPoseEstimation =
	//	rs2::processing_block([](
	//		cv::Mat& image, unsigned long long frame_id,
	//		cv::Matx33f& cameraMatrix,
	//		std::vector<float> distortionCoefficients,
	//		std::map<unsigned long long, std::vector<int>>& charucoIdBuffer,
	//		std::map<unsigned long long, cv::Vec3d>& rotationBuffer,
	//		std::map<unsigned long long, cv::Vec3d>& translationBuffer
	//		)
	//{
	//	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
	//	cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(3, 3, 0.04, 0.02, dictionary);
	//	/*cv::Ptr<cv::aruco::DetectorParameters> params;
	//	params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_NONE;*/

	//	std::vector<int> ids;
	//	std::vector<std::vector<cv::Point2f>> corners;
	//	cv::aruco::detectMarkers(image, dictionary, corners, ids);
	//	// if at least one marker detected
	//	if (ids.size() > 0) {
	//		cv::aruco::drawDetectedMarkers(image, corners, ids);
	//		std::vector<cv::Point2f> charucoCorners;
	//		std::vector<int> charucoIds;
	//		cv::aruco::interpolateCornersCharuco(corners, ids, image, board, charucoCorners, charucoIds);
	//		// if at least one charuco corner detected
	//		if (charucoIds.size() > 0) {
	//			cv::aruco::drawDetectedCornersCharuco(image, charucoCorners, charucoIds, cv::Scalar(255, 0, 0));
	//			cv::Vec3d rotation, translation;
	//			bool valid = cv::aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix, distortionCoefficients, rotation, translation);
	//			// if charuco pose is valid
	//			if (valid) {
	//				cv::aruco::drawAxis(image, cameraMatrix, distortionCoefficients, rotation, translation, 0.1);

	//				charucoIdBuffer[frame_id] = charucoIds;
	//				translationBuffer[frame_id] = translation;
	//				rotationBuffer[frame_id] = rotation;
	//			}
	//		}

	//		//std::vector<cv::Vec4i> diamondIds;
	//		//std::vector<std::vector<cv::Point2f>> diamondCorners;
	//		//// detect diamon diamonds
	//		//cv::aruco::detectCharucoDiamond(image, corners, ids, squareLength / markerLength, diamondCorners, diamondIds);
	//		//// estimate poses
	//		//std::vector<cv::Vec3d> rvecs, tvecs;
	//		//cv::aruco::estimatePoseSingleMarkers(diamondCorners, squareLength, cameraMatrix, distCoeff, rvecs, tvecs);
	//		//// draw axis
	//		//for (unsigned int i = 0; i < rvecs.size(); i++)
	//		//	cv::aruco::drawAxis(image, cameraMatrix, distCoeff, rvecs[i], tvecs[i], 1);
	//	}
	//});
}