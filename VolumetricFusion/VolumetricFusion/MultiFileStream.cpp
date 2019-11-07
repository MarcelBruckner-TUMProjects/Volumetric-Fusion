#pragma region Includes
// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
// Include short list of convenience functions for rendering

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
#include "Settings.hpp"
#include "Data.hpp"
#include "CaptureDevice.hpp"
#include "VolumetricFusion/FileAccess.hpp"

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
	//std::vector<vc::data::Data> datas;
	std::vector< vc::capture::CaptureDevice> pipelines;
	std::vector<std::string> streamNames;

	if (state.captureState == CaptureState::RECORDING || state.captureState == CaptureState::STREAMING) {
		auto devices = ctx.query_devices();
		if (state.captureState == CaptureState::RECORDING) {
			if (devices.size() > 0) {
				vc::file_access::resetDirectory(folderSettings.recordingsFolder, true);
			}
		}
		int i = 0;
		for (auto&& device : ctx.query_devices())
		{
			if (state.captureState == CaptureState::RECORDING) {
				pipelines.push_back(vc::capture::RecordingCaptureDevice(ctx, device, folderSettings.recordingsFolder));
			}
			else if (state.captureState == CaptureState::STREAMING) {
				pipelines.push_back(vc::capture::StreamingCaptureDevice(ctx, device));
			}
			i++;
		}
	}
	else if (state.captureState == CaptureState::PLAYING){
		std::vector<std::string> filenames = vc::file_access::listFilesInFolder(folderSettings.recordingsFolder);

		for (int i = 0; i < filenames.size() && i < 4; i++)
		{
			pipelines.push_back(vc::capture::PlayingCaptureDevice(ctx, filenames[i]));
		}
	}


	if (pipelines.size() <= 0) {
		throw(rs2::error("No device or file found!"));
	}
	while (streamNames.size() < 4) {
		streamNames.push_back("");
	}
#pragma endregion

#pragma region Variables
	   
	// Create a thread for getting frames from the device and process them
	// to prevent UI thread from blocking due to long computations.
	std::atomic_bool stopped(false);
	std::atomic_bool paused(false);
	
	std::map<int, std::thread> captureThreads;
	
	// Create custom depth processing block and their output queues:
	/*std::map<int, rs2::frame_queue> depth_processing_queues;
	std::map<int, std::shared_ptr<rs2::processing_block>> depth_processing_blocks;*/
		
	// Calculated relative transformations between cameras per frame
	std::map<std::tuple<int, int>, std::map<int, Eigen::Matrix4d>> relativeTransformations;

	// Camera calibration thread
	std::thread calibrationThread;
	std::atomic_bool calibrateCameras = true;
#pragma endregion

#pragma region Camera intrinsics

	for (int i = 0; i < pipelines.size(); i++) {
		pipelines[i].startPipeline();
		pipelines[i].data->setIntrinsics(pipelines[i].pipeline->get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics());
	}
#pragma endregion

#pragma region Camera Calibration Thread

		calibrationThread = std::thread([&pipelines, &stopped, &calibrateCameras, &relativeTransformations]() {
			while(!stopped){
				if (!calibrateCameras) {
					continue;
				}

				for (int i = 0; i < pipelines.size(); i++) {
					std::map<unsigned long long, std::vector<int>> baseCharucoIdBuffer = pipelines[i].data->processing.charucoIdBuffers;
					std::vector<unsigned long long> outerFrameIds = extractKeys(baseCharucoIdBuffer);

					for (int j = 0; j < pipelines.size(); j++) {
						if (i == j) {
							continue;
						}

						std::map<unsigned long long, std::vector<int>> relativeCharucoIdBuffer = pipelines[i].data->processing.charucoIdBuffers;
						std::vector<unsigned long long> innerFrameIds = extractKeys(relativeCharucoIdBuffer);

						std::vector<unsigned long long> overlapingFrames = findOverlap(outerFrameIds, innerFrameIds);

						for (auto frame : overlapingFrames) 
						{
							Eigen::Matrix4d baseToMarkerTranslation = pipelines[i].data->processing.translationBuffers[frame];
							Eigen::Matrix4d baseToMarkerRotation = pipelines[i].data->processing.rotationBuffers[frame];

							Eigen::Matrix4d markerToRelativeTranslation = pipelines[j].data->processing.translationBuffers[frame].inverse();
							Eigen::Matrix4d markerToRelativeRotation = pipelines[j].data->processing.rotationBuffers[frame].inverse();

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
		captureThreads[i] = std::thread(&vc::capture::CaptureDevice::captureThreadFunction, pipelines[i]);
		pipelines[i].paused->store(false);
	}
#pragma endregion

#pragma region Processing blocks

	for (int i = 0; i < pipelines.size(); i++) {
		cv::Matx33f cameraMatrix = pipelines[i].data->camera.cameraMatrices;
		auto distCoeff = pipelines[i].data->camera.distCoeffs;

		const auto charucoPoseEstimation = [&pipelines, cameraMatrix, distCoeff, i](cv::Mat& image, unsigned long long frameId) {
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

							pipelines[i].data->processing.charucoIdBuffers[frameId] = charucoIds;
							Eigen::Matrix4d tmpTranslation;
							tmpTranslation.setIdentity();
							tmpTranslation.block<3,1>(0,3) << translation[0], translation[1], translation[2];
							pipelines[i].data->processing.translationBuffers[frameId] = tmpTranslation;
							
							cv::Matx33d tmp;
							cv::Rodrigues(rotation, tmp);
							Eigen::Matrix4d tmpRotation;
							tmpRotation.setIdentity();
							tmpRotation.block<3, 3>(0, 0) <<
								tmp.val[0], tmp.val[1], tmp.val[2],
								tmp.val[3], tmp.val[4], tmp.val[5],
								tmp.val[6], tmp.val[7], tmp.val[8];
							pipelines[i].data->processing.rotationBuffers[frameId] = tmpRotation;

							//std::stringstream ss;
							//ss << "************************************************************************************" << std::endl;
							//ss << "Device " << i << ", Frame " << frame_id << ":" << std::endl << "Translation: " << std::endl << tmpTranslation << std::endl << "Rotation: " << std::endl << tmpRotation << std::endl;
							//std::cout << ss.str();
						}
					}
				}
		};


		pipelines[i].data->processing.charucoProcessingBlocks = std::make_shared<rs2::processing_block>(processing_blocks::createColorProcessingBlock(charucoPoseEstimation));
		pipelines[i].data->processing.charucoProcessingBlocks->start(pipelines[i].data->processing.charucoProcessingQueues); // Bind output of the processing block to be enqueued into the queue
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
		//TODO
//		addSaveFramesButton(folderSettings.capturesFolder, pipelines, colorizedDepthFrames, points);
		if (state.renderState == RenderState::MULTI_POINTCLOUD) {
			//addAlignPointCloudsButton(paused, points);
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
				if (pipelines[i].data->colorizedDepthFrames && pipelines[i].data->points) {
					viewOrientation.tex.upload(pipelines[i].data->colorizedDepthFrames);   //  and upload the texture to the view (without this the view will be B&W)
					if (isRetinaDisplay) {
						glViewport(width * (i % 2), height - (height * (i / 2)), width, height);
					}
					else {
						glViewport(widthHalf * (i % 2), heightHalf - (heightHalf * (i / 2)), widthHalf, heightHalf);
					}

					if (pipelines[i].data->filteredColorFrames) {
						draw_pointcloud_and_colors(widthHalf, heightHalf, viewOrientation, pipelines[i].data->points, pipelines[i].data->filteredColorFrames, 0.2f);
					}
					else {
						draw_pointcloud(widthHalf, heightHalf, viewOrientation, pipelines[i].data->points);
					}

					if (pipelines.size()) {
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
				if (pipelines[i].data->filteredColorFrames != false) {
					rect r{
						static_cast<float>(widthHalf * (i % 2)),
							static_cast<float>(heightHalf - (heightHalf * (i / 2))),
							static_cast<float>(widthHalf),
							static_cast<float>(heightHalf)
					};
					if (isRetinaDisplay) {
						r = rect{ width * (i % 2), height - (height * (i / 2)), width, height };
					}
					pipelines[i].data->tex.render(pipelines[i].data->filteredColorFrames, r);
				}
			}

			break;
		}

		case RenderState::ONLY_DEPTH:
		{
			for (int i = 0; i < pipelines.size() && i < 4; ++i)
			{
				if (pipelines[i].data->filteredDepthFrames != false) {
					rect r{
					    static_cast<float>(widthHalf * (i % 2)),
                        static_cast<float>(heightHalf - (heightHalf * (i / 2))),
                        static_cast<float>(widthHalf),
                        static_cast<float>(heightHalf)
					};
					if (isRetinaDisplay) {
						r = rect{ width * (i % 2), height - (height * (i / 2)), width, height };
					}
					pipelines[i].data->tex.render(pipelines[i].data->colorizer.process(pipelines[i].data->filteredDepthFrames), r);
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