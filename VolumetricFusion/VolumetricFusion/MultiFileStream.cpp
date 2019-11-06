#pragma region Includes
// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
// Include short list of convenience functions for rendering
#if APPLE
#include "../example.hpp"
#include "FileAccess.hpp"
//#include "CaptureDevice.h"
#else
#include "example.hpp"
#include "VolumetricFusion/FileAccess.hpp"
//#include "VolumetricFusion/CaptureDevice.h"
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>

#include <map>
#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <filesystem>

#if APPLE
#include <glut.h>
#else
#include <GL/gl.h>
#endif

#include <imgui.h>
#include "imgui_impl_glfw.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "Enums.hpp"
using namespace vc::enums;

#include "ImGuiHelpers.hpp"
using namespace imgui_helpers;
#include "ProcessingBlocks.hpp"
#include "MultiFileStream.h"
#include "Eigen/Dense"
#include <VolumetricFusion\Settings.hpp>

#pragma endregion

template<typename T, typename V>
std::vector<T> extractKeys(std::map<T, V> const& input_map);

template<typename T>
std::vector<T> findOverlap(std::vector<T> a, std::vector<T> b);

int main(int argc, char* argv[]) try {

#pragma region Non window setting
	vc::settings::FolderSettings folderSettings;
	vc::settings::State state = vc::settings::State(CaptureState::PLAYING);
#pragma endregion
	
#pragma region Window initialization
	// Create a simple OpenGL window for rendering:
	window app(1280, 960, "VolumetricFusion - MultiStreamViewer");

	ImGui_ImplGlfw_Init(app, false);

	// Construct an object to manage view state
	glfw_state viewOrientation;
	// Let the user control and manipulate the scenery orientation
	register_glfw_callbacks(app, viewOrientation);

	viewOrientation.yaw = -2.6;
	viewOrientation.pitch = 0.8;
	viewOrientation.last_x = 738.7;
	viewOrientation.last_y = 1015.8;
	viewOrientation.offset_x = 2.0;
	viewOrientation.offset_y = -2.0;
#pragma endregion

#pragma region Pipeline initialization
	rs2::context ctx; // Create librealsense context for managing devices
	std::map<int, std::shared_ptr<rs2::pipeline>> pipelines;

	std::vector<std::string> streamNames(4);
	int i = 0;
	switch (state.captureState) {
	case CaptureState::STREAMING:
		for (auto&& device : ctx.query_devices())
		{
			if (i >= 4) {
				break;
			}
			auto pipe = std::make_shared<rs2::pipeline>(ctx);

			rs2::config cfg;
			std::string deviceName = device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
			cfg.enable_device(deviceName);
			cfg.enable_all_streams();
			cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 760, 6);
			pipe->start(cfg);
			pipelines[i] = pipe;
			streamNames[i] = deviceName;
			i++;
		}
		break;
	case CaptureState::RECORDING:
	{
		auto devices = ctx.query_devices();
		if (devices.size() > 0) {
			file_access::resetDirectory(folderSettings.recordingsFolder, true);
		}
		for (auto&& device : ctx.query_devices())
		{
			auto pipe = std::make_shared<rs2::pipeline>(ctx);

			rs2::config cfg;
			std::string deviceName = device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
			cfg.enable_device(deviceName);
			cfg.enable_all_streams();
			cfg.enable_record_to_file(folderSettings.recordingsFolder + deviceName + ".bag");
			pipe->start(cfg);
			pipelines[i] = pipe;
			streamNames[i] = deviceName;
			i++;
		}
		break;
	}
	case CaptureState::PLAYING:
	{
		std::vector<std::string> filenames = file_access::listFilesInFolder(folderSettings.recordingsFolder);

		for (int i = 0; i < filenames.size() && i < 4; i++)
		{
			auto pipe = std::make_shared<rs2::pipeline>(ctx);

			rs2::config cfg;
			std::string deviceName = filenames[i];
			cfg.enable_device_from_file(deviceName);
			cfg.enable_all_streams();
			pipe->start(cfg);
			pipelines[i] = pipe;
			streamNames[i] = deviceName;
		}
	}
	break;

	default:
		break;
	}

	if (pipelines.size() <= 0) {
		throw(rs2::error("No device or file found!"));
	}
	while (streamNames.size() < 4) {
		streamNames.push_back("");
	}
#pragma endregion

#pragma region Variables
	   
	std::map<int, texture> textures;
	std::map<int, rs2::colorizer> colorizers;

	// Declare filters
	rs2::decimation_filter decimationFilter(1);         // Decimation - reduces depth frame density
	rs2::threshold_filter thresholdFilter(0.0f, 1.1f); // Threshold  - removes values outside recommended range
	// 0 - fill_from_left - Use the value from the left neighbor pixel to fill the hole
	// 1 - farest_from_around - Use the value from the neighboring pixel which is furthest away from the sensor
	// 2 - nearest_from_around - - Use the value from the neighboring pixel closest to the sensor
	//rs2::hole_filling_filter hole_filter(1); // terrible results ...

	std::vector<rs2::filter*> filters;
	filters.emplace_back(&decimationFilter);
	filters.emplace_back(&thresholdFilter);
	//filters.emplace_back(&hole_filter);

	// Create a thread for getting frames from the device and process them
	// to prevent UI thread from blocking due to long computations.
	std::atomic_bool stopped(false);
	std::atomic_bool paused(false);

	std::map<int, rs2::frame> filteredColorFrames;
	std::map<int, rs2::frame> filteredDepthFrames;

	std::map<int, std::thread> captureThreads;

	std::map<int, rs2::pointcloud> pointclouds;
	std::map<int, rs2::frame> colorizedDepthFrames;
	std::map<int, rs2::points> points;

	// Create custom color processing block and their output queues:
	std::map<int, rs2::frame_queue> charucoProcessingQueues;
	std::map<int, std::shared_ptr<rs2::processing_block>> charucoProcessingBlocks;

	// Create custom depth processing block and their output queues:
	/*std::map<int, rs2::frame_queue> depth_processing_queues;
	std::map<int, std::shared_ptr<rs2::processing_block>> depth_processing_blocks;*/

	// Pose estimation camera stuff
	std::map<int, rs2_intrinsics> intrinsics;
	std::map<int, cv::Matx33f> cameraMatrices;
	std::map<int, std::vector<float>> distCoeffs;

	// Pose estimation buffers
	// buffer <pipelineId, <frame_id, value>>
	std::map<int, std::map<unsigned long long, std::vector<int>>> charucoIdBuffers;
	std::map<int, std::map<unsigned long long, Eigen::Matrix4d>> rotationBuffers;
	std::map<int, std::map<unsigned long long, Eigen::Matrix4d>> translationBuffers;

	// Calculated relative transformations between cameras per frame
	std::map<std::tuple<int, int>, std::map<int, Eigen::Matrix4d>> relativeTransformations;

	// Camera calibration thread
	std::thread calibrationThread;
	std::atomic_bool calibrateCameras = true;
#pragma endregion

#pragma region Camera intrinsics

	for (int i = 0; i < pipelines.size(); i++) {
		intrinsics[i] = pipelines[i]->get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
		cameraMatrices[i] = cv::Matx33f(
			intrinsics[i].fx, 0, intrinsics[i].ppx, 
			0, intrinsics[i].fy, intrinsics[i].ppy, 
			0,0,1
		);

		for (float c : intrinsics[i].coeffs) {
			distCoeffs[i].push_back(c);
		}
	}
#pragma endregion

#pragma region Camera Calibration Thread

		calibrationThread = std::thread([&stopped, &calibrateCameras, &rotationBuffers, &translationBuffers, &charucoIdBuffers, &relativeTransformations]() {
			while(!stopped){
				if (!calibrateCameras) {
					continue;
				}

				for (int i = 0; i < charucoIdBuffers.size(); i++) {
					std::map<unsigned long long, std::vector<int>> baseCharucoIdBuffer = charucoIdBuffers[i];
					std::vector<unsigned long long> outerFrameIds = extractKeys(baseCharucoIdBuffer);

					for (int j = 0; j < charucoIdBuffers.size(); j++) {
						if (i == j) {
							continue;
						}

						std::map<unsigned long long, std::vector<int>> relativeCharucoIdBuffer = charucoIdBuffers[j];
						std::vector<unsigned long long> innerFrameIds = extractKeys(relativeCharucoIdBuffer);

						std::vector<unsigned long long> overlapingFrames = findOverlap(outerFrameIds, innerFrameIds);

						for (auto frame : overlapingFrames) 
						{
							Eigen::Matrix4d baseToMarkerTranslation = translationBuffers[i][frame];
							Eigen::Matrix4d baseToMarkerRotation = rotationBuffers[i][frame];

							Eigen::Matrix4d markerToRelativeTranslation = translationBuffers[j][frame].inverse();
							Eigen::Matrix4d markerToRelativeRotation = rotationBuffers[j][frame].inverse();

							Eigen::Matrix4d relativeTransformation = markerToRelativeTranslation * markerToRelativeRotation * baseToMarkerRotation * baseToMarkerTranslation;

							relativeTransformations[std::make_tuple(i, j)][frame] = relativeTransformation;

							std::stringstream ss;
							ss << "************************************************************************************" << std::endl;
							ss << "Devices " << i << ", " << j << " - Frame " << frame << std::endl << std::endl;
							ss << "Translations: " << std::endl << baseToMarkerTranslation << std::endl << markerToRelativeTranslation << std::endl << std::endl;
							ss << "Rotations: " << std::endl << baseToMarkerRotation << std::endl << markerToRelativeRotation << std::endl << std::endl;
							ss << "Combined: " << std::endl << relativeTransformation << std::endl;
							std::cout << ss.str();
						}
					}
				}
			}
		});
#pragma endregion

#pragma region Processing Threads

	for (int i = 0; i < pipelines.size(); ++i) {
		captureThreads[i] = std::thread([&, i]() {
			auto pipe = pipelines[i];

			rs2::align alignToColor(RS2_STREAM_COLOR);

			while (!stopped.load()) //While application is running
			{
				while (paused.load()) {
					continue;
				}

				try {
					rs2::frameset data = pipe->wait_for_frames(); // Wait for next set of frames from the camera

					data = alignToColor.process(data);

					rs2::frame depthFrame = data.get_depth_frame(); //Take the depth frame from the frameset
					if (!depthFrame) { // Should not happen but if the pipeline is configured differently
						return;       //  it might not provide depth and we don't want to crash
					}

					rs2::frame filteredDepthFrame = depthFrame; // Does not copy the frame, only adds a reference

					rs2::frame colorFrame = data.get_color_frame();

					if (calibrateCameras) {
						// Send color frame for processing
						charucoProcessingBlocks[i]->invoke(colorFrame);
						// Wait for results
						colorFrame = charucoProcessingQueues[i].wait_for_frame();
					}

					filteredColorFrames[i] = colorFrame;

					// Apply filters.
					for (auto&& filter : filters) {
						filteredDepthFrame = filter->process(filteredDepthFrame);
					}

					// Push filtered & original data to their respective queues
					filteredDepthFrames[i] = filteredDepthFrame;

					points[i] = pointclouds[i].calculate(depthFrame);  // Generate pointcloud from the depth data
					colorizedDepthFrames[i] = colorizers[i].process(depthFrame);		// Colorize the depth frame with a color map
					pointclouds[i].map_to(colorizedDepthFrames[i]);      // Map the colored depth to the point cloud
				}
				catch (const std::exception & e) {
					std::stringstream stream;
					stream << "******************** THREAD ERROR *******************" << std::endl << e.what() << "****************************************************" << std::endl;
				}
			}
			pipe->stop();
		});
	}
#pragma endregion

#pragma region Processing blocks

	for (int i = 0; i < pipelines.size(); i++) {
		cv::Matx33f cameraMatrix = cameraMatrices[i];
		auto distCoeff = distCoeffs[i];

		const auto charucoPoseEstimation = [&rotationBuffers, &translationBuffers, &charucoIdBuffers, cameraMatrix, distCoeff, i](cv::Mat& image, unsigned long long frameId) {
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
						cv::Vec3d rotation, translation;
						bool valid = cv::aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix, distCoeff, rotation, translation);
						// if charuco pose is valid
						if (valid) {
							cv::aruco::drawAxis(image, cameraMatrix, distCoeff, rotation, translation, 0.1);

							charucoIdBuffers[i][frameId] = charucoIds;
							Eigen::Matrix4d tmpTranslation;
							tmpTranslation.setIdentity();
							tmpTranslation.block<3,1>(0,3) << translation[0], translation[1], translation[2];
							translationBuffers[i][frameId] = tmpTranslation;
							
							cv::Matx33d tmp;
							cv::Rodrigues(rotation, tmp);
							Eigen::Matrix4d tmpRotation;
							tmpRotation.setIdentity();
							tmpRotation.block<3, 3>(0, 0) <<
								tmp.val[0], tmp.val[1], tmp.val[2],
								tmp.val[3], tmp.val[4], tmp.val[5],
								tmp.val[6], tmp.val[7], tmp.val[8];
							rotationBuffers[i][frameId] = tmpRotation;

							//std::stringstream ss;
							//ss << "************************************************************************************" << std::endl;
							//ss << "Device " << i << ", Frame " << frame_id << ":" << std::endl << "Translation: " << std::endl << tmpTranslation << std::endl << "Rotation: " << std::endl << tmpRotation << std::endl;
							//std::cout << ss.str();
						}
					}
				}
		};


		charucoProcessingBlocks[i] = std::make_shared<rs2::processing_block>(processing_blocks::createColorProcessingBlock(charucoPoseEstimation));
		charucoProcessingBlocks[i]->start(charucoProcessingQueues[i]); // Bind output of the processing block to be enqueued into the queue
	}
#pragma endregion
	
#pragma region Main loop

	while (app)
	{
		int topOffset = 70;
		const float width = static_cast<float>(app.width());
		const float height = static_cast<float>(app.height()) - (topOffset * 0.5f);
		const int widthHalf = width / 2;
		const int heightHalf = height / 2;

		// Retina display (Mac OS) have double the pixel density
		int w2, h2;
		glfwGetFramebufferSize(app, &w2, &h2);
		const bool isRetinaDisplay = w2 == width * 2 && h2 == height * 2;

		imgui_helpers::initialize(app, w2, h2, streamNames, widthHalf, heightHalf, width, height);
		addSwitchViewButton(state.renderState, calibrateCameras);
		addPauseResumeToggle(paused);
		addSaveFramesButton(folderSettings.capturesFolder, pipelines, colorizedDepthFrames, points);
		if (state.renderState == RenderState::MULTI_POINTCLOUD) {
			addAlignPointCloudsButton(paused, points);
		}

		if (state.renderState != RenderState::ONLY_DEPTH) {
			addCalibrateToggle(calibrateCameras);
		}
		if (state.renderState != RenderState::ONLY_COLOR) {
			//addToggleDepthProcessingButton(depthProcessing);
		}
		imgui_helpers::addGenerateCharucoDiamond(folderSettings.charucoFolder);
		imgui_helpers::addGenerateCharucoBoard(folderSettings.charucoFolder);
		imgui_helpers::finalize();
		
		switch (state.renderState) {
		case RenderState::COUNT:
		case RenderState::MULTI_POINTCLOUD:
		{
			// Draw the pointclouds
			for (int i = 0; i < pipelines.size() && i < 4; ++i)
			{
				if (colorizedDepthFrames[i] && points[i]) {
					viewOrientation.tex.upload(colorizedDepthFrames[i]);   //  and upload the texture to the view (without this the view will be B&W)
					if (isRetinaDisplay) {
						glViewport(width * (i % 2), height - (height * (i / 2)), width, height);
					}
					else {
						glViewport(widthHalf * (i % 2), heightHalf - (heightHalf * (i / 2)), widthHalf, heightHalf);
					}

					if (filteredColorFrames[i]) {
						draw_pointcloud_and_colors(widthHalf, heightHalf, viewOrientation, points[i], filteredColorFrames[i], 0.2f);
					}
					else {
						draw_pointcloud(widthHalf, heightHalf, viewOrientation, points[i]);
					}

					if (distCoeffs.size()) {
					    draw_rectangle(widthHalf, heightHalf, 0, 0, 0, viewOrientation);
					}
				}
			}
		}
		break;

		case RenderState::ONLY_COLOR:
		{
			for (int i = 0; i < pipelines.size() && i < 4; ++i)
			{
				if (filteredColorFrames[i] != false) {
					rect r{
							static_cast<float>(widthHalf * (i % 2)),
							static_cast<float>(heightHalf - (heightHalf * (i / 2))),
							static_cast<float>(widthHalf),
							static_cast<float>(heightHalf)
					};
					if (isRetinaDisplay) {
						r = rect{ width * (i % 2), height - (height * (i / 2)), width, height };
					}
					textures[i].render(filteredColorFrames[i], r);
				}
			}

			break;
		}

		case RenderState::ONLY_DEPTH:
		{
			for (int i = 0; i < pipelines.size() && i < 4; ++i)
			{
				if (filteredDepthFrames[i] != false) {
					rect r{
					    static_cast<float>(widthHalf * (i % 2)),
                        static_cast<float>(heightHalf - (heightHalf * (i / 2))),
                        static_cast<float>(widthHalf),
                        static_cast<float>(heightHalf)
					};
					if (isRetinaDisplay) {
						r = rect{ width * (i % 2), height - (height * (i / 2)), width, height };
					}
					textures[i].render(colorizers[i].process(filteredDepthFrames[i]), r);
				}
			}

			break;
		}
		}
	}
#pragma endregion

#pragma region Final cleanup

	stopped.store(true);
	for (auto& thread : captureThreads) {
		thread.second.join();
	}
#pragma endregion

	return EXIT_SUCCESS;
}
#pragma region Error handling
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
#pragma endregion

#pragma region Helper functions

template<typename T, typename V>
std::vector<T> extractKeys(std::map<T,V> const& input_map) {
	std::vector<T> retval;
	for (auto const& element : input_map) {
		retval.push_back(element.first);
	}
	return retval;
}


template<typename T>
std::vector<T> findOverlap(std::vector<T> a, std::vector<T> b) {
	std::vector<T> c;
	
	for (T x : a) {
		if (std::find(b.begin(), b.end(), x) != b.end()) {
			c.push_back(x);
		}
	}

	return c;
}

#pragma endregion