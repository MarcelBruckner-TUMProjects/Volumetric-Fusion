// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

// Include short list of convenience functions for rendering
#if APPLE
#include "../example.hpp"          
#else
#include "example.hpp" 
#endif

#include <algorithm>            // std::min, std::max

#include <map>
#include <vector>

#include "CaptureDevice.h"

// Helper functions
void register_glfw_callbacks(window& app, glfw_state& app_state);

int main(int argc, char* argv[]) try
{
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
	
	// Main app loop
	while (app)
	{
		for (auto device : devices) {
			device->acquireFrame();

			// TODO merge pointclouds

			app_state.tex.upload(device->getColorFrame());
			draw_pointcloud(app.width(), app.height(), app_state, device->getPoints());
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
