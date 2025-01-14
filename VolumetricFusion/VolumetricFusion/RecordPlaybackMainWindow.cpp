// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#if APPLE
#include "../example.hpp"          // Include short list of convenience functions for rendering
#else
#include "example.hpp"          // Include short list of convenience functions for rendering
#endif
#include <chrono>

#include <imgui.h>
#include "imgui_impl_glfw.h"

// Includes for time display
#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <io.h>
#include <direct.h>

#if APPLE
#include "FileAccess.hpp"
#else
#include "VolumetricFusion/FileAccess.hpp"
#endif

#include <string>
#include <iostream>


// Helper function for dispaying time conveniently
std::string pretty_time(std::chrono::nanoseconds duration);
void create_new_pipeline_device(rs2::context& ctx, rs2::device&& dev, std::map<std::string, std::shared_ptr<rs2::pipeline>>& pipes, std::map<std::string, rs2::colorizer>& colorizers, std::map<std::string, texture>& depth_images, std::map<std::string, texture>& color_images, std::map<std::string, rs2::frameset>& framesets, std::map<std::string, rs2::frame>& depths, std::map<std::string, rs2::frame>& colors, std::map<std::string, rs2::pointcloud>& pc, std::map<std::string, rs2::points>& points);
// Helper function for rendering a seek bar
void draw_seek_bar(rs2::playback& playback, int* seek_pos, float2& location, float width);

// Helper functions
void register_glfw_callbacks(window& app, glfw_state& app_state);

enum class RenderState {
	POINTCLOUD,
	DEPTH_AND_COLOR,
	/*DEPTH ,
	COLOR */
	COUNT
};

enum class CaptureState {
	STREAMING,
	RECORDING,
	RECORDING_PAUSED,
	PLAYING,
	PLAYING_PAUSED,
	COUNT
};

int main(int argc, char* argv[]) try
{
	// Create a simple OpenGL window for rendering:
	window app(1280, 720, "RealSense Record and Playback Example");
	ImGui_ImplGlfw_Init(app, false);
	// Construct an object to manage view state
	glfw_state app_state;
	// register callbacks to allow manipulation of the pointcloud
	register_glfw_callbacks(app, app_state);

	// Recordings folder
	std::string folder = "recordings/";
	// Recordings filename
	std::vector<std::string> filenames;
	std::map<std::string, std::string> pipeToFilenameMap;

	file_access::iterateFilesInFolder(folder, [&](const auto& entry) { 
		auto path = entry.path();
		std::string extension = path.extension().string();
		if (extension == ".bag") {
			filenames.push_back(path.filename().string());
		}
	}, true);

	bool recorded = filenames.size() > 0;
	
	// The render state
	RenderState renderState = RenderState::POINTCLOUD;

	// The capture state
	CaptureState captureState = CaptureState::STREAMING;

	// Create librealsense context for managing devices
	const rs2::context ctx;

	// Declare a texture for the depth image on the GPU
	std::map < std::string, texture> depth_images;
	// Declare a texture for the depth image on the GPU
	std::map < std::string, texture> color_images;

	// We'll keep track of the last frame of each stream available to make the presentation persistent
	std::map<int, rs2::frame> render_frames;

	// Declare frameset and frames which will hold the data from the camera
	std::map < std::string, rs2::frameset> framesets;
	std::map < std::string, rs2::frame> depths;
	std::map < std::string, rs2::frame> colors;

	// Declare pointcloud object, for calculating pointclouds and texture mappings
	std::map < std::string, rs2::pointcloud> pointclouds;
	// We want the points object to be persistent so we can display the last cloud when a frame drops
	std::map < std::string, rs2::points> points;

	// Declare depth colorizer for pretty visualization of depth data
	std::map < std::string, rs2::colorizer> colorizers;
	
	std::vector<std::string> physicalDeviceNames;
	// Shared pointers to devices
	std::map<std::string, std::shared_ptr<rs2::pipeline>> pipes_map;
	for (auto&& dev : ctx.query_devices()) {
		auto pipe = std::make_shared<rs2::pipeline>(ctx);
		rs2::config cfg;
		std::string device_name = dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
		cfg.enable_device(device_name);
		pipe->start(cfg);

		pipes_map[device_name] = pipe;
		colorizers[device_name] = rs2::colorizer();
		depth_images[device_name] = texture();
		color_images[device_name] = texture();
		framesets[device_name] = rs2::frameset();
		depths[device_name] = rs2::frame();
		colors[device_name] = rs2::frame();
		pointclouds[device_name] = rs2::pointcloud();
		points[device_name] = rs2::points();
		physicalDeviceNames.push_back(device_name);
	}

	// The number of pyhsical devices
	int numPhysicalDevices = pipes_map.size();

	// Weather a physical RealSense device is connected
	bool isPhysicalDeviceConnected = numPhysicalDevices > 0;

	// Throw excpetion if no device is connected and no recording is found
	if(!isPhysicalDeviceConnected && !recorded){
	_THROW(rs2::error("No device and no recording found!"));
	}

	// The number of found files that contain a recording and will serve as non physical device
	int numOnlyRecordedDevices = 0;
	if(recorded){
		for (int i = 0; pipes_map.size() < filenames.size(); i++) {
			auto pipe = std::make_shared<rs2::pipeline>(ctx);
			rs2::config cfg;
			std::string device_name = "__" + i + std::string("__");
			cfg.enable_device_from_file(device_name);

			pipes_map[device_name] = pipe;
			colorizers[device_name] = rs2::colorizer();
			depth_images[device_name] = texture();
			color_images[device_name] = texture();
			framesets[device_name] = rs2::frameset();
			depths[device_name] = rs2::frame();
			colors[device_name] = rs2::frame();
			pointclouds[device_name] = rs2::pointcloud();
			points[device_name] = rs2::points();
			numOnlyRecordedDevices++;
		}
	}

	std::vector<std::string> allDeviceNames;
	for (const auto& entry : pipes_map) {
		allDeviceNames.push_back(entry.first);
	}

	// Map devices to filenames for recording and replaying
	int i = 0;
	for each (const auto& entry in pipes_map)
	{
		if (i < filenames.size()) {
			pipeToFilenameMap[entry.first] = folder + filenames[i++] ;
		}
		else {
			pipeToFilenameMap[entry.first] = folder + entry.first + ".bag";
		}
	}

	//// Start streaming with the default configuration
	//for (auto& pipe_entry : pipes_map) {
	//	pipe_entry.second->start();
	//}

	// Create a variable to control the seek bar
	int seek_pos;


	while (app) {

		// While application is running
		// Flags for displaying ImGui window
		static const int flags = ImGuiWindowFlags_NoCollapse
			| ImGuiWindowFlags_NoScrollbar
			| ImGuiWindowFlags_NoSavedSettings
			| ImGuiWindowFlags_NoTitleBar
			| ImGuiWindowFlags_NoResize
			| ImGuiWindowFlags_NoMove;

		ImGui_ImplGlfw_NewFrame(1);
		ImGui::SetNextWindowSize({ app.width(), app.height() });
		ImGui::Begin("app", nullptr, flags);

		/* ################################## IMGUI RENDERING ######################################## */
		
		// Set options for the ImGui buttons
		ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, { 1, 1, 1, 1 });
		ImGui::PushStyleColor(ImGuiCol_Button, { 36 / 255.f, 44 / 255.f, 51 / 255.f, 1 });
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, { 40 / 255.f, 170 / 255.f, 90 / 255.f, 1 });
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, { 36 / 255.f, 44 / 255.f, 51 / 255.f, 1 });
		ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12);

		
		if (captureState == CaptureState::STREAMING || captureState == CaptureState::RECORDING || captureState == CaptureState::RECORDING_PAUSED) // Disable recording while device is playing
		{
			if (captureState != CaptureState::RECORDING)
			{
				ImGui::SetCursorPos({ app.width() / 2 - 250, app.height() - 80 });
				ImGui::Text("Click 'record' to start recording");
			}
			ImGui::SetCursorPos({ app.width() / 2 - 250, app.height() - 60 });
			if (ImGui::Button("record", { 50, 50 }))
			{
				file_access::resetFolder(folder);
				recorded = false;

				for (auto& pipe_entry : pipes_map) {
					std::string device_name = pipe_entry.first;
					std::string filename = pipeToFilenameMap[device_name];

					// Initialize a shared pointer to a device with the current device on the pipeline
					rs2::device device = pipe_entry.second->get_active_profile().get_device();

					// If it is the start of a new recording (device is not a recorder yet)
					if (!device.as<rs2::recorder>())
					{
						pipe_entry.second->stop(); // Stop the pipeline with the default configuration
						pipe_entry.second = std::make_shared<rs2::pipeline>();
						rs2::config cfg; // Declare a new configuration
						cfg.enable_record_to_file(filename);
						cfg.enable_device(pipe_entry.first);

						pipe_entry.second->start(cfg); //File will be opened at this point
						device = pipe_entry.second->get_active_profile().get_device();
					}
					else
					{ // If the recording is resumed after a pause, there's no need to reset the shared pointer
						device.as<rs2::recorder>().resume(); // rs2::recorder allows access to 'resume' function
					}
				}
				captureState = CaptureState::RECORDING;
			}

			/*
			When pausing, device still holds the file.
			*/
			if (captureState == CaptureState::RECORDING || captureState == CaptureState::RECORDING_PAUSED)
			{
				if (captureState == CaptureState::RECORDING)
				{
					ImGui::SetCursorPos({ app.width() / 2 - 250, app.height() - 80 });
					std::string gui_text = "Recording to folder '" + folder + "'";
					ImGui::TextColored({ 255 / 255.f, 64 / 255.f, 54 / 255.f, 1 }, gui_text.c_str());
				}

				// Pause the playback if button is clicked
				ImGui::SetCursorPos({ app.width() / 2 - 150, app.height() - 60 });
				if (ImGui::Button("pause\nrecord", { 50, 50 }))
				{
					for (auto& pipe_entry : pipes_map) {
						// Initialize a shared pointer to a device with the current device on the pipeline
						rs2::device device = pipe_entry.second->get_active_profile().get_device();

						device.as<rs2::recorder>().pause();
					}
					
					captureState = CaptureState::RECORDING_PAUSED;
				}

				ImGui::SetCursorPos({ app.width() / 2 - 50, app.height() - 60 });
				if (ImGui::Button(" stop\nrecord", { 50, 50 }))
				{
					for (auto& pipe_entry : pipes_map) {
						// Initialize a shared pointer to a device with the current device on the pipeline
						rs2::device device = pipe_entry.second->get_active_profile().get_device();

						pipe_entry.second->stop(); // Stop the pipeline that holds the file and the recorder
						pipe_entry.second = std::make_shared<rs2::pipeline>(); //Reset the shared pointer with a new pipeline

						rs2::config cfg;
						cfg.enable_device(pipe_entry.first);
						pipe_entry.second->start(cfg);

						device = pipe_entry.second->get_active_profile().get_device();
					}
					recorded = true; // Now we can run the file
					
					captureState = CaptureState::STREAMING;
				}
			}
		}

		// PLAYING RELATED

		// After a recording is done, we can play it
		if (recorded) {
			ImGui::SetCursorPos({ app.width() / 2 + 100,  app.height() - 80 });
			ImGui::Text("Click 'play' to start playing");
			ImGui::SetCursorPos({ app.width() / 2 + 100, app.height() - 60 });
			if (ImGui::Button("play", { 50, 50 }))
			{
				for (auto& pipe_entry : pipes_map) {
					std::string device_name = pipe_entry.first;
					std::string filename = pipeToFilenameMap[device_name];

					if (device_name.find(std::string("__")) == std::string::npos) {
						// Initialize a shared pointer to a device with the current device on the pipeline
						rs2::device device = pipe_entry.second->get_active_profile().get_device();
						if (!device.as<rs2::playback>())
						{
							pipe_entry.second->stop(); // Stop streaming with default configuration
							pipe_entry.second = std::make_shared<rs2::pipeline>();
							rs2::config cfg;
							cfg.enable_device_from_file(filename);
							pipe_entry.second->start(cfg); //File will be opened in read mode at this point
						}
						else
						{
							device.as<rs2::playback>().resume();
						}
					}
					else {
						//pipe_entry.second->stop(); // Stop streaming with default configuration
						pipe_entry.second = std::make_shared<rs2::pipeline>();
						rs2::config cfg;
						cfg.enable_device_from_file(filename);
						pipe_entry.second->start(cfg); //File will be opened in read mode at this point
					}
				}
				captureState = CaptureState::PLAYING;
			}
		}
		
		// If device is playing a recording, we allow pause and stop
		if (captureState == CaptureState::PLAYING)
		{
			// Render a seek bar for the player
			//float2 location = { app.width() / 4, app.height() - 50 };
			//draw_seek_bar(playback, &seek_pos, location, app.width() / 4);

			ImGui::SetCursorPos({ app.width() / 2 + 200, app.height() - 60 });
			if (ImGui::Button(" pause\nplaying", { 50, 50 }))
			{
				for (auto& pipe_entry : pipes_map) {
					// Initialize a shared pointer to a device with the current device on the pipeline
					rs2::device device = pipe_entry.second->get_active_profile().get_device();
					rs2::playback playback = device.as<rs2::playback>();

					playback.pause();
				}
				captureState = CaptureState::PLAYING_PAUSED;
			}

			if (isPhysicalDeviceConnected) {
				ImGui::SetCursorPos({ app.width() / 2 + 300, app.height() - 60 });
				if (ImGui::Button("  stop\nplaying", { 50, 50 }))
				{
					for (auto& pipe_entry : pipes_map) {
						// Initialize a shared pointer to a device with the current device on the pipeline
						rs2::device device = pipe_entry.second->get_active_profile().get_device();
						pipe_entry.second->stop();
						pipe_entry.second = std::make_shared<rs2::pipeline>();
						rs2::config cfg;
						cfg.enable_device(pipe_entry.first);
						pipe_entry.second->start(cfg);
						device = pipe_entry.second->get_active_profile().get_device();
					}
					captureState = CaptureState::STREAMING;
				}
			}
		}


		// SWITCH VIEW
		ImGui::SetCursorPos({ app.width() - 60, app.height() - 60 });
		if (ImGui::Button("switch\n view", { 50, 50 }))
		{
			int s = (int)renderState;
			s++;
			s %= (int)RenderState::COUNT;
			renderState = (RenderState)s;
		}


		ImGui::PopStyleColor(4);
		ImGui::PopStyleVar();

		ImGui::End();
		ImGui::Render();
		/* ################################## IMGUI RENDERING ######################################## */

		std::vector<std::string>* renderDeviceNames = &allDeviceNames;
		if (captureState == CaptureState::STREAMING || captureState == CaptureState::RECORDING || captureState == CaptureState::RECORDING_PAUSED) {
			renderDeviceNames = &physicalDeviceNames;
		}

		int numRenderDevices = renderDeviceNames->size();

		float width = app.width() * 0.2f;
		float height = (app.height() * 0.75f) / (float)ceil(numRenderDevices / 2);
		
		render_frames = std::map<int, rs2::frame>();

		// The device number of the currently used device
		int currentDeviceNumber = 0;
		for (std::string device_name : *renderDeviceNames) {
			auto pipe = pipes_map[device_name];
			rs2::device device = pipe->get_active_profile().get_device();

			//// If the device is streaming live and not from a file
			if (!device.as<rs2::playback>())
			{
				framesets[device_name] = pipe->wait_for_frames(); // wait for next set of frames from the camera
				depths[device_name] = framesets[device_name].get_depth_frame(); // Find and colorize the depth data
				colors[device_name] = framesets[device_name].get_color_frame(); // Get the color data
			}

			//// If device is playing a recording, we allow pause and stop
			if (device.as<rs2::playback>())
			{
				if (pipe->poll_for_frames(&framesets[device_name])) // Check if new frames are ready
				{
					depths[device_name] = framesets[device_name].get_depth_frame(); // Find and colorize the depth data for rendering
					colors[device_name] = framesets[device_name].get_color_frame(); // Get the color data for rendering
				}
			}

			/* ################################## RENDERING ######################################## */
			try {
				switch (renderState)
				{
				case RenderState::POINTCLOUD:
					// Tell pointcloud object to map to this color frame
					pointclouds[device_name].map_to(colors[device_name]);

					// Generate the pointcloud and texture mappings
					points[device_name] = pointclouds[device_name].calculate(depths[device_name]);

					// Upload the color frame to OpenGL
					app_state.tex.upload(colors[device_name]);

					// Draw the pointcloud
					draw_pointcloud(app.width(), app.height(), app_state, points[device_name]);
					break;
				case RenderState::DEPTH_AND_COLOR:
					render_frames[depths[device_name].get_profile().unique_id()] = colorizers[device_name].process(depths[device_name]);
					render_frames[colors[device_name].get_profile().unique_id()] = colors[device_name];

					break;
				default:
					break;
				}
			}
			catch (const rs2::error & e)
			{
				std::cout << "RealSense error while RENDERING! " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
			}


			/* ################################## RENDERING ######################################## */

			// Leave here !!
			currentDeviceNumber++;

		}

		if (renderState == RenderState::DEPTH_AND_COLOR) {
			app.show(render_frames);
		}
	}
	return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
	std::cout << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
	return EXIT_FAILURE;
}
catch (const std::exception & e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}


std::string pretty_time(std::chrono::nanoseconds duration)
{
	using namespace std::chrono;
	auto hhh = duration_cast<hours>(duration);
	duration -= hhh;
	auto mm = duration_cast<minutes>(duration);
	duration -= mm;
	auto ss = duration_cast<seconds>(duration);
	duration -= ss;
	auto ms = duration_cast<milliseconds>(duration);

	std::ostringstream stream;
	stream << std::setfill('0') << std::setw(hhh.count() >= 10 ? 2 : 1) << hhh.count() << ':' <<
		std::setfill('0') << std::setw(2) << mm.count() << ':' <<
		std::setfill('0') << std::setw(2) << ss.count();
	return stream.str();
}


void draw_seek_bar(rs2::playback& playback, int* seek_pos, float2& location, float width)
{
	int64_t playback_total_duration = playback.get_duration().count();
	auto progress = playback.get_position();
	double part = (1.0 * progress) / playback_total_duration;
	*seek_pos = static_cast<int>(std::max(0.0, std::min(part, 1.0)) * 100);
	auto playback_status = playback.current_status();
	ImGui::PushItemWidth(width);
	ImGui::SetCursorPos({ location.x, location.y });
	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12);
	if (ImGui::SliderInt("##seek bar", seek_pos, 0, 100, "", true))
	{
		//Seek was dragged
		if (playback_status != RS2_PLAYBACK_STATUS_STOPPED) //Ignore seek when playback is stopped
		{
			auto duration_db = std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(playback.get_duration());
			auto single_percent = duration_db.count() / 100;
			auto seek_time = std::chrono::duration<double, std::nano>((*seek_pos) * single_percent);
			playback.seek(std::chrono::duration_cast<std::chrono::nanoseconds>(seek_time));
		}
	}
	std::string time_elapsed = pretty_time(std::chrono::nanoseconds(progress));
	ImGui::SetCursorPos({ location.x + width + 10, location.y });
	ImGui::Text("%s", time_elapsed.c_str());
	ImGui::PopStyleVar();
	ImGui::PopItemWidth();
}
