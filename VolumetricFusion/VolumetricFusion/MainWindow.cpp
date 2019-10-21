// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.
#pragma once

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include "example.hpp"          // Include short list of convenience functions for rendering

#include <algorithm>            // std::min, std::max

#include <map>
#include <vector>

#include "CaptureDevice.h"
#include "Recorder.h"
#include "future"

// Helper functions
void register_glfw_callbacks(window& app, glfw_state& app_state);

int main(int argc, char* argv[]) try
{
	bool record = false;
	std::string baseDir = "C:\\Users\\Marcel Bruckner\\Documents\\Volumetric-Fusion\\points\\";

	// Create a simple OpenGL window for rendering:
	window app(1280, 960, "Volumetric Fusion");
	// Construct an object to manage view state
	glfw_state app_state;
	// register callbacks to allow manipulation of the pointcloud
	register_glfw_callbacks(app, app_state);

	rs2::context					ctx;			// Create librealsense context for managing devices

	std::vector<CaptureDevice*>		devices;

	// Start a streaming pipe per each connected device
	for (auto&& dev : ctx.query_devices())
	{
		CaptureDevice* device = new CaptureDevice(ctx, dev);
		devices.push_back(device);
		device->start();
	}

	if (record) {
		std::experimental::filesystem::remove_all(baseDir);
		std::experimental::filesystem::create_directory(baseDir);

		for (auto device : devices) {
			std::experimental::filesystem::create_directory(baseDir + "\\" + device->getSerialNr()
			);
		}
	}

	// Main app loop
	while (app)
	{
		for (auto device : devices) {
			device->acquireFrame();
			
			// TODO background segmentation
			// TODO merge pointclouds

			rs2::video_frame color = device->getColorFrame();
			rs2::points points = device->getPoints();
			unsigned long long frameNumber = device->getFrameNumber();
			std::string serialNr = device->getSerialNr();

			if (record && frameNumber >= 0) {
				auto fut = std::async(std::launch::async, [points, baseDir, serialNr, frameNumber, color] {((rs2::points)points).export_to_ply(baseDir + "\\" + serialNr + "\\" + std::to_string(frameNumber) + ".ply", color); });
			}

			app_state.tex.upload(color);
			draw_pointcloud(app.width(), app.height(), app_state, points);
		}
	}

	return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
	std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
	return EXIT_FAILURE;
}
catch (const std::exception & e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}