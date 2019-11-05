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
using namespace enums;

#include "ImGuiHelpers.hpp"
using namespace imgui_helpers;
#include "ProcessingBlocks.hpp"
#include "MultiFileStream.h"

int main(int argc, char* argv[]) try {

	CaptureState captureState = CaptureState::PLAYING;
	RenderState renderState = RenderState::ONLY_COLOR;

	std::string captures_folder = "captures/";

	//std::string recordings_folder = "recordings/";
	std::string recordings_folder = "single_stream_recording/";

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

	rs2::context ctx; // Create librealsense context for managing devices
	std::map<int, std::shared_ptr<rs2::pipeline>> pipelines;

	std::vector<std::string> stream_names(4);
	int i = 0;
	switch (captureState) {
	case CaptureState::STREAMING:
		for (auto&& device : ctx.query_devices())
		{
			if (i >= 4) {
				break;
			}
			auto pipe = std::make_shared<rs2::pipeline>(ctx);

			rs2::config cfg;
			std::string device_name = device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
			cfg.enable_device(device_name);
			cfg.enable_all_streams();
			pipe->start(cfg);
			pipelines[i] = pipe;
			stream_names[i] = device_name;
			i++;
		}
		break;
	case CaptureState::RECORDING:
	{
		auto devices = ctx.query_devices();
		if (devices.size() > 0) {
			file_access::resetDirectory(recordings_folder, true);
		}
		for (auto&& device : ctx.query_devices())
		{
			auto pipe = std::make_shared<rs2::pipeline>(ctx);

			rs2::config cfg;
			std::string device_name = device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
			cfg.enable_device(device_name);
			cfg.enable_all_streams();
			cfg.enable_record_to_file(recordings_folder + device_name + ".bag");
			pipe->start(cfg);
			pipelines[i] = pipe;
			stream_names[i] = device_name;
			i++;
		}
		break;
	}
	case CaptureState::PLAYING:
	{
		std::vector<std::string> figure_filenames = file_access::listFilesInFolder(recordings_folder);

		for (int i = 0; i < figure_filenames.size() && i < 4; i++)
		{
			auto pipe = std::make_shared<rs2::pipeline>(ctx);

			rs2::config cfg;
			std::string device_name = figure_filenames[i];
			cfg.enable_device_from_file(device_name);
			cfg.enable_all_streams();
			pipe->start(cfg);
			pipelines[i] = pipe;
			stream_names[i] = device_name;
		}
	}
	break;

	default:
		break;
	}

	if (pipelines.size() <= 0) {
		throw(rs2::error("No device or file found!"));
	}
	while (stream_names.size() < 4) {
		stream_names.push_back("");
	}


	// We'll keep track of the last frame of each stream available to make the presentation persistent
	std::map<int, rs2::frame> render_frames;

	std::map<int, texture> textures;
	std::map<int, rs2::colorizer> colorizers;

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
	std::atomic_bool stopped(false);
	std::atomic_bool paused(false);
	std::atomic_bool depthProcessing(false);
	std::atomic_bool colorProcessing(false);
	//std::vector<rs2::frame_queue> filtered_datas(pipelines.size());
	std::map<int, rs2::frame> filtered_color_frames;
	std::map<int, rs2::frame> filtered_depth_frames;
	std::map<int, std::thread> processing_threads;

	std::map<int, rs2::pointcloud> filtered_pointclouds;
	std::map<int, rs2::frame> colorized_depth_frames;
	std::map<int, rs2::points> filtered_points;

	// Create custom color processing block and their output queues:
	std::map<int, rs2::frame_queue> color_processing_queues;
	std::map<int, std::shared_ptr<rs2::processing_block>> color_processing_blocks;

	// Create custom depth processing block and their output queues:
	std::map<int, rs2::frame_queue> depth_processing_queues;
	std::map<int, std::shared_ptr<rs2::processing_block>> depth_processing_blocks;

	for (int i = 0; i < pipelines.size(); i++) {
	    const auto cpb_callback = [](cv::Mat& image) {
			std::vector<int> markerIds;
			std::vector<std::vector<cv::Point2f>> markerCorners;
			cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
			cv::aruco::detectMarkers(image, dictionary, markerCorners, markerIds);
			cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);
        };
		color_processing_blocks[i] = std::make_shared<rs2::processing_block>(processing_blocks::createColorProcessingBlock(cpb_callback));
		color_processing_blocks[i]->start(color_processing_queues[i]); // Bind output of the processing block to be enqueued into the queue

		const auto dpb_callback = [](cv::Mat& image) {
          cv::medianBlur(image, image, 11);
        };
		depth_processing_blocks[i] = std::make_shared<rs2::processing_block>(processing_blocks::createDepthProcessingBlock(dpb_callback));
		depth_processing_blocks[i]->start(depth_processing_queues[i]); // Bind output of the processing block to be enqueued into the queue
	}

	for (int i = 0; i < pipelines.size(); ++i) {
		processing_threads[i] = std::thread([&, i]() {
			auto pipe = pipelines[i];

			rs2::align align_to_color(RS2_STREAM_COLOR);

			while (!stopped) //While application is running
			{
				while (paused.load()) {
					continue;
				}

				try {
					rs2::frameset data = pipe->wait_for_frames(); // Wait for next set of frames from the camera

					data = align_to_color.process(data);

					rs2::frame depth_frame = data.get_depth_frame(); //Take the depth frame from the frameset
					if (!depth_frame) { // Should not happen but if the pipeline is configured differently
						return;       //  it might not provide depth and we don't want to crash
					}

					if (depthProcessing) {
						// Send depth frame for processing
						depth_processing_blocks[i]->invoke(depth_frame);
						// Wait for results
						depth_frame = depth_processing_queues[i].wait_for_frame();
					}
					rs2::frame filtered_depth_frame = depth_frame; // Does not copy the frame, only adds a reference

					rs2::frame color_frame = data.get_color_frame();

					if (colorProcessing) {
						// Send color frame for processing
						color_processing_blocks[i]->invoke(color_frame);
						// Wait for results
						color_frame = color_processing_queues[i].wait_for_frame();
					}

					filtered_color_frames[i] = color_frame;

					// Apply filters.
					for (auto&& filter : filters) {
						filtered_depth_frame = filter->process(filtered_depth_frame);
					}

					// Push filtered & original data to their respective queues
					filtered_depth_frames[i] = filtered_depth_frame;

					filtered_points[i] = filtered_pointclouds[i].calculate(depth_frame);  // Generate pointcloud from the depth data
					colorized_depth_frames[i] = colorizers[i].process(depth_frame);		// Colorize the depth frame with a color map
					filtered_pointclouds[i].map_to(colorized_depth_frames[i]);      // Map the colored depth to the point cloud
				}
				catch (const std::exception & e) {
					std::stringstream stream;
					stream << "******************** THREAD ERROR *******************" << std::endl << e.what() << "****************************************************" << std::endl;
				}
			}
			pipe->stop();
		});
	}

	bool align_frames = false;

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
		const bool is_retina_display = w2 == width * 2 && h2 == window_main.height() * 2;

		imgui_helpers::initialize(window_main, w2, h2, stream_names, width_half, height_half, width, height);
		addSwitchViewButton(renderState, depthProcessing, colorProcessing);
		addPauseResumeButton(paused);
		addSaveFramesButton(captures_folder, pipelines, colorized_depth_frames, filtered_points);
		if (renderState == RenderState::MULTI_POINTCLOUD) {
			addAlignPointCloudsButton(paused, filtered_points);
		}

		if (renderState != RenderState::ONLY_DEPTH) {
			addToggleColorProcessingButton(colorProcessing);
			std::cout << colorProcessing << std::endl;
		}
		if (renderState != RenderState::ONLY_COLOR) {
			addToggleDepthProcessingButton(depthProcessing);
		}
		imgui_helpers::finalize();

		render_frames = std::map<int, rs2::frame>();

		switch (renderState) {
		case RenderState::COUNT:
		case RenderState::MULTI_POINTCLOUD:
		{
			// Draw the pointclouds
			for (int i = 0; i < pipelines.size() && i < 4; ++i)
			{
				if (colorized_depth_frames[i] && filtered_points[i]) {
					view_orientation.tex.upload(colorized_depth_frames[i]);   //  and upload the texture to the view (without this the view will be B&W)
					if (is_retina_display) {
						glViewport(width * (i % 2), height - (height * (i / 2)), width, height);
					}
					else {
						glViewport(width_half * (i % 2), height_half - (height_half * (i / 2)), width_half, height_half);
					}

					if (filtered_color_frames[i]) {
						draw_pointcloud_and_colors(width_half, height_half, view_orientation, filtered_points[i], filtered_color_frames[i], 0.2f);
					}
					else {
						draw_pointcloud(width_half, height_half, view_orientation, filtered_points[i]);
					}
				}
			}
		}
		break;

		case RenderState::ONLY_COLOR:
		{
			for (int i = 0; i < pipelines.size() && i < 4; ++i)
			{
				if (filtered_color_frames[i] != false) {
					rect r{
                            static_cast<float>(width_half * (i % 2)),
                            static_cast<float>(height_half - (height_half * (i / 2))),
                            static_cast<float>(width_half),
                            static_cast<float>(height_half)
                    };
					if (is_retina_display) {
						r = rect{ width * (i % 2), height - (height * (i / 2)), width, height };
					}
					textures[i].render(filtered_color_frames[i], r);
				}
			}

			break;
		}

		case RenderState::ONLY_DEPTH:
		{
			for (int i = 0; i < pipelines.size() && i < 4; ++i)
			{
				if (filtered_depth_frames[i] != false) {
					rect r{
					    static_cast<float>(width_half * (i % 2)),
                        static_cast<float>(height_half - (height_half * (i / 2))),
                        static_cast<float>(width_half),
                        static_cast<float>(height_half)
					};
					if (is_retina_display) {
						r = rect{ width * (i % 2), height - (height * (i / 2)), width, height };
					}
					textures[i].render(colorizers[i].process(filtered_depth_frames[i]), r);
				}
			}

			break;
		}
		}
	}

	stopped = true;
	for (auto& thread : processing_threads) {
		thread.second.join();
	}

	return EXIT_SUCCESS;
}
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
