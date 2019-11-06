#pragma once
#pragma region Includes
// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
// Include short list of convenience functions for rendering
#if APPLE
#include "FileAccess.hpp"
//#include "CaptureDevice.h"
#else
//#include "example.hpp"
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
#define NOMINMAX
#include <windows.h>
#include <GL/gl.h>
#endif


#include <imgui.h>
#include "imgui_impl_glfw.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "Enums.hpp"
using namespace enums;

#include "Settings.hpp"
#include "ImGuiHelpers.hpp"
#include "VolumetricFusion/CaptureDevice.hpp"
using namespace imgui_helpers;
#pragma endregion

template<typename T, typename V>
std::vector<T> extract_keys(std::map<T, V> const& input_map);

int main(int argc, char* argv[]) try {

	vc::settings::MarkerSettings markerSettings = vc::settings::MarkerSettings();
	vc::settings::OutputFolders outputFolders = vc::settings::OutputFolders();
	vc::settings::States state = vc::settings::States();
	
#pragma region Window initialization
	// Create a simple OpenGL window for rendering:
	window window_main(1280, 960, "VolumetricFusion - MultiStreamViewer");

	ImGui_ImplGlfw_Init(window_main, false);


	// Construct an object to manage view state
	glfw_state view_orientation{};
	// Let the user control and manipulate the scenery orientation
	register_glfw_callbacks(window_main, view_orientation);

	view_orientation.yaw = -2.6;
	view_orientation.pitch = 0.8;
	view_orientation.last_x = 738.7;
	view_orientation.last_y = 1015.8;
	view_orientation.offset_x = 2.0;
	view_orientation.offset_y = -2.0;
#pragma endregion


#pragma region Variables

	// Declare filters
	rs2::decimation_filter decimation_filter(1);         // Decimation - reduces depth frame density
	rs2::threshold_filter threshold_filter(0.0f, 1.1f); // Threshold  - removes values outside recommended range
	// 0 - fill_from_left - Use the value from the left neighbor pixel to fill the hole
	// 1 - farest_from_around - Use the value from the neighboring pixel which is furthest away from the sensor
	// 2 - nearest_from_around - - Use the value from the neighboring pixel closest to the sensor
	//rs2::hole_filling_filter hole_filter(1); // terrible results ...

	std::vector<rs2::filter*> filters;
	filters.emplace_back(&decimation_filter);
	filters.emplace_back(&threshold_filter);
	//filters.emplace_back(&hole_filter);

	// Create a thread for getting frames from the device and process them
	// to prevent UI thread from blocking due to long computations.

	std::shared_ptr<std::atomic_bool> isStopped = std::make_shared<std::atomic_bool>(false);
	std::shared_ptr<std::atomic_bool> isPaused = std::make_shared<std::atomic_bool>(false);
	std::shared_ptr<std::atomic_bool> isColorProcessing = std::make_shared<std::atomic_bool>(true);


	// Camera calibration thread
	std::thread calibrationThread;
	std::atomic_bool calibrateCameras = true;
#pragma endregion

#pragma region Capture device initialization
	rs2::context ctx; // Create librealsense context for managing devices
	//std::map<int, std::shared_ptr<rs2::pipeline>> pipelines;

	//std::vector<std::string> streamNames(4);
	std::vector<vc::CaptureDevice> captureDevices;

	int i = 0;
	if (state.captureState != CaptureState::PLAYING) {
		auto devices = ctx.query_devices();
		if (state.captureState == CaptureState::RECORDING) {
			if (devices.size() > 0) {
				file_access::resetDirectory(outputFolders.recordingsFolder, true);
			}
		}

		for (auto&& device : devices)
		{
			if (i >= 4) {
				break;
			}

			if (state.captureState == CaptureState::STREAMING) {
				captureDevices.push_back(vc::StreamingCaptureDevice(ctx, device, isStopped, isPaused, isColorProcessing));
			}
			if (state.captureState == CaptureState::RECORDING) {
				captureDevices.push_back(vc::RecordingCaptureDevice(ctx, device, outputFolders.recordingsFolder, isStopped, isPaused, isColorProcessing));
			}
			i++;
		}
	}
	else {
		std::vector<std::string> figure_filenames = file_access::listFilesInFolder(outputFolders.recordingsFolder);

		for (int i = 0; i < figure_filenames.size() && i < 4; i++)
		{
			captureDevices.push_back(vc::PlayingCaptureDevice(ctx, figure_filenames[i], isStopped, isPaused, isColorProcessing));
		}
	}


	if (captureDevices.size() <= 0) {
		throw(rs2::error("No device or file found!"));
	}

	for each (auto captureDevice in captureDevices)
	{
		captureDevice->start();
		captureDevice->setIntrinsics();
	}
#pragma endregion

	
#pragma region Camera Calibration Thread

		calibrationThread = std::thread([&, i]() {
			while(!isStopped && captureDevices.size() > 1){
				if (!calibrateCameras) {
					continue;
				}

				for (int i = 0; i < captureDevices.size(); ++i) {
				}
			}
		});
#pragma endregion

		
#pragma region Main loop

	std::vector<std::string> streamNames;

	for each (auto captureDevice in captureDevices)
	{
		streamNames.push_back(captureDevice->deviceName);
	}

	while (streamNames.size() < 4) {
		streamNames.push_back("");
	}

	while (window_main)
	{
		int topOffset = 70;
		const float width = static_cast<float>(window_main.width());
		const float height = static_cast<float>(window_main.height()) - (topOffset * 0.5f);
		const int width_half = width / 2;
		const int height_half = height / 2;

		// Retina display (Mac OS) have double the pixel density
		int w2, h2;
		glfwGetFramebufferSize(window_main, &w2, &h2);
		const bool is_retina_display = w2 == width * 2 && h2 == height * 2;

		imgui_helpers::initialize(window_main, w2, h2, streamNames, width_half, height_half, width, height);
		addSwitchViewButton(state.renderState, isColorProcessing);
		addPauseResumeButton(isPaused);
		addSaveFramesButton(outputFolders.capturesFolder, captureDevices);
		if (state.renderState == RenderState::MULTI_POINTCLOUD) {
			addAlignPointCloudsButton(isPaused, captureDevices);
		}

		if (state.renderState != RenderState::ONLY_DEPTH) {
			addToggleColorProcessingButton(isColorProcessing);
		}
		if (state.renderState != RenderState::ONLY_COLOR) {
			//addToggleDepthProcessingButton(depthProcessing);
		}
		imgui_helpers::addGenerateCharucoDiamond(outputFolders.charucoFolder);
		imgui_helpers::addGenerateCharucoBoard(outputFolders.charucoFolder);
		imgui_helpers::finalize();
		
		switch (state.renderState) {
		case RenderState::COUNT:
		case RenderState::MULTI_POINTCLOUD:
		{
			int i = 0;
			for each (auto captureDevice in captureDevices)
			{
				captureDevice->renderPointcloud(i, view_orientation, is_retina_display, width, height, width_half, height_half);
				i++;
			}
		}
		break;

		case RenderState::ONLY_COLOR:
		{
			int i = 0;
			for each (auto captureDevice in captureDevices)
			{
					rect r{
							static_cast<float>(width_half * (i % 2)),
							static_cast<float>(height_half - (height_half * (i / 2))),
							static_cast<float>(width_half),
							static_cast<float>(height_half)
					};
					if (is_retina_display) {
						r = rect{ width * (i % 2), height - (height * (i / 2)), width, height };
					}
					captureDevice->renderOnlyColor(r);
				
				i++;
			}

			break;
		}

		case RenderState::ONLY_DEPTH:
		{
			int i = 0;
			for each (auto captureDevice in captureDevices)
			{
				if (captureDevice->filteredDepthFrame) {
					rect r{
					    static_cast<float>(width_half * (i % 2)),
                        static_cast<float>(height_half - (height_half * (i / 2))),
                        static_cast<float>(width_half),
                        static_cast<float>(height_half)
					};
					if (is_retina_display) {
						r = rect{ width * (i % 2), height - (height * (i / 2)), width, height };
					}
					captureDevice->renderOnlyDepth(r);
				}
				i++;
			}

			break;
		}
		}
	}
#pragma endregion

#pragma region Final cleanup

	*isStopped = true;
	for each (auto captureDevice in captureDevices)
	{
		captureDevice->stop();
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
std::vector<T> extract_keys(std::map<T,V> const& input_map) {
	std::vector<T> retval;
	for (auto const& element : input_map) {
		retval.push_back(element.first);
	}
	return retval;
}

#pragma endregion