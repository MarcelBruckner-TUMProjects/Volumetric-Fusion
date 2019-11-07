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
#include "Processing.hpp"
#include "MultiFileStream.h"
#include "Eigen/Dense"
#include "Settings.hpp"
#include "Data.hpp"
#include "CaptureDevice.hpp"
#include "VolumetricFusion/FileAccess.hpp"
#include <signal.h>

#pragma endregion

template<typename T, typename V>
std::vector<T> extractKeys(std::map<T, V> const& input_map);

template<typename T>
std::vector<T> findOverlap(std::vector<T> a, std::vector<T> b);

void my_function_to_handle_aborts(int signalNumber) {
	std::cout << "Something aborted" << std::endl;
}

int main(int argc, char* argv[]) try {

	signal(SIGABRT, &my_function_to_handle_aborts);

	vc::settings::FolderSettings folderSettings;
	vc::settings::State state = vc::settings::State(CaptureState::STREAMING);
	
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


	/*Do this early in your program's initialization */

	rs2::context ctx; // Create librealsense context for managing devices
	//std::vector<vc::data::Data> datas;
	std::vector<std::shared_ptr<  vc::capture::CaptureDevice>> pipelines;
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
				pipelines.emplace_back(std::make_shared < vc::capture::RecordingCaptureDevice>(ctx, device, folderSettings.recordingsFolder));
			}
			else if (state.captureState == CaptureState::STREAMING) {
				pipelines.emplace_back(std::make_shared < vc::capture::StreamingCaptureDevice>(ctx, device));
			}
			i++;
		}
	}
	else if (state.captureState == CaptureState::PLAYING){
		std::vector<std::string> filenames = vc::file_access::listFilesInFolder(folderSettings.recordingsFolder);

		for (int i = 0; i < filenames.size() && i < 4; i++)
		{
			pipelines.emplace_back(std::make_shared < vc::capture::PlayingCaptureDevice>(ctx, filenames[i]));
		}
	}


	if (pipelines.size() <= 0) {
		throw(rs2::error("No device or file found!"));
	}
	while (streamNames.size() < 4) {
		streamNames.emplace_back("");
	}
	   
	// Create a thread for getting frames from the device and process them
	// to prevent UI thread from blocking due to long computations.
	std::atomic_bool stopped(false);
	std::atomic_bool paused(false);
		
	// Create custom depth processing block and their output queues:
	/*std::map<int, rs2::frame_queue> depth_processing_queues;
	std::map<int, std::shared_ptr<rs2::processing_block>> depth_processing_blocks;*/
		
	// Calculated relative transformations between cameras per frame
	std::map<std::tuple<int, int>, std::map<int, Eigen::Matrix4d>> relativeTransformations;

	// Camera calibration thread
	std::thread calibrationThread;
	std::atomic_bool calibrateCameras = true;
	
	for (int i = 0; i < pipelines.size(); i++) {
		pipelines[i]->startPipeline();
		pipelines[i]->resumeThread();
		pipelines[i]->calibrate(calibrateCameras);
	}

#pragma region Camera Calibration Thread

		calibrationThread = std::thread([&pipelines, &stopped, &calibrateCameras, &relativeTransformations]() {
			while(!stopped){
				if (!calibrateCameras) {
					continue;
				}

				for (int i = 0; i < pipelines.size(); i++) {
					std::map<unsigned long long, std::vector<int>> baseCharucoIdBuffer = pipelines[i]->processing->charucoIdBuffers;
					std::vector<unsigned long long> outerFrameIds = extractKeys(baseCharucoIdBuffer);

					for (int j = 0; j < pipelines.size(); j++) {
						if (i == j) {
							continue;
						}

						std::map<unsigned long long, std::vector<int>> relativeCharucoIdBuffer = pipelines[i]->processing->charucoIdBuffers;
						std::vector<unsigned long long> innerFrameIds = extractKeys(relativeCharucoIdBuffer);

						std::vector<unsigned long long> overlapingFrames = findOverlap(outerFrameIds, innerFrameIds);

						for (auto frame : overlapingFrames) 
						{
							Eigen::Matrix4d baseToMarkerTranslation = pipelines[i]->processing->translationBuffers[frame];
							Eigen::Matrix4d baseToMarkerRotation = pipelines[i]->processing->rotationBuffers[frame];

							Eigen::Matrix4d markerToRelativeTranslation = pipelines[j]->processing->translationBuffers[frame].inverse();
							Eigen::Matrix4d markerToRelativeRotation = pipelines[j]->processing->rotationBuffers[frame].inverse();

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

		vc::imgui_helpers::initialize(app, w2, h2, streamNames, widthHalf, heightHalf, width, height);
		vc::imgui_helpers::addSwitchViewButton(state.renderState, calibrateCameras);
		if (vc::imgui_helpers::addPauseResumeToggle(paused)) {
			for (int i = 0; i < pipelines.size(); i++)
			{
				pipelines[i]->paused->store(paused);
			}
		}

//		addSaveFramesButton(folderSettings.capturesFolder, pipelines, colorizedDepthFrames, points);
		if (state.renderState == RenderState::MULTI_POINTCLOUD) {
			//addAlignPointCloudsButton(paused, points);
		}

		if (state.renderState != RenderState::ONLY_DEPTH) {
			if (vc::imgui_helpers::addCalibrateToggle(calibrateCameras)) {
				for (int i = 0; i < pipelines.size(); i++)
				{
					pipelines[i]->calibrateCameras ->store(calibrateCameras);
				}
			}
		}
		if (state.renderState != RenderState::ONLY_COLOR) {
			//addToggleDepthProcessingButton(depthProcessing);
		}
		vc::imgui_helpers::addGenerateCharucoDiamond(folderSettings.charucoFolder);
		vc::imgui_helpers::addGenerateCharucoBoard(folderSettings.charucoFolder);
		vc::imgui_helpers::finalize();
		
		switch (state.renderState) {
		case RenderState::COUNT:
		case RenderState::MULTI_POINTCLOUD:
		{
			// Draw the pointclouds
			for (int i = 0; i < pipelines.size() && i < 4; ++i)
			{
				if (pipelines[i]->data->colorizedDepthFrames && pipelines[i]->data->points) {
					viewOrientation.tex.upload(pipelines[i]->data->colorizedDepthFrames);   //  and upload the texture to the view (without this the view will be B&W)
					if (isRetinaDisplay) {
						glViewport(width * (i % 2), height - (height * (i / 2)), width, height);
					}
					else {
						glViewport(widthHalf * (i % 2), heightHalf - (heightHalf * (i / 2)), widthHalf, heightHalf);
					}

					if (pipelines[i]->data->filteredColorFrames) {
						draw_pointcloud_and_colors(widthHalf, heightHalf, viewOrientation, pipelines[i]->data->points, pipelines[i]->data->filteredColorFrames, 0.2f);
					}
					else {
						draw_pointcloud(widthHalf, heightHalf, viewOrientation, pipelines[i]->data->points);
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
				if (pipelines[i]->data->filteredColorFrames != false) {
					rect r{
						static_cast<float>(widthHalf * (i % 2)),
							static_cast<float>(heightHalf - (heightHalf * (i / 2))),
							static_cast<float>(widthHalf),
							static_cast<float>(heightHalf)
					};
					if (isRetinaDisplay) {
						r = rect{ width * (i % 2), height - (height * (i / 2)), width, height };
					}
					pipelines[i]->data->tex.render(pipelines[i]->data->filteredColorFrames, r);
				}
			}

			break;
		}

		case RenderState::ONLY_DEPTH:
		{
			for (int i = 0; i < pipelines.size() && i < 4; ++i)
			{
				if (pipelines[i]->data->filteredDepthFrames != false) {
					rect r{
					    static_cast<float>(widthHalf * (i % 2)),
                        static_cast<float>(heightHalf - (heightHalf * (i / 2))),
                        static_cast<float>(widthHalf),
                        static_cast<float>(heightHalf)
					};
					if (isRetinaDisplay) {
						r = rect{ width * (i % 2), height - (height * (i / 2)), width, height };
					}
					pipelines[i]->data->tex.render(pipelines[i]->data->colorizer.process(pipelines[i]->data->filteredDepthFrames), r);
				}
			}

			break;
		}
		}
	}
#pragma endregion

#pragma region Final cleanup

	stopped.store(true);
	for (int i = 0; i < pipelines.size(); i++) {
		pipelines[i]->stopThread();
	}
#pragma endregion

	std::exit(EXIT_SUCCESS);
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
		retval.emplace_back(element.first);
	}
	return retval;
}


template<typename T>
std::vector<T> findOverlap(std::vector<T> a, std::vector<T> b) {
	std::vector<T> c;
	
	for (T x : a) {
		if (std::find(b.begin(), b.end(), x) != b.end()) {
			c.emplace_back(x);
		}
	}

	return c;
}

#pragma endregion