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

#include <map>
#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <filesystem>

#include <gl/GL.h>

#include <imgui.h>
#include "imgui_impl_glfw.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

enum class RenderState {
	MULTI_POINTCLOUD,
	SINGLE_COLOR,
	COUNT
};

enum class CaptureState {
	STREAMING,
	RECORDING,
	PLAYING,
	COUNT
};

class Settings {
public:
	CaptureState captureState = CaptureState::STREAMING;
	RenderState renderState = RenderState::SINGLE_COLOR;
	std::string captures_folder = "captures/";
	std::string recordings_folder = "recordings/";
}settings;

// Helper functions for rendering the UI
void render_ui(float w, float h);

int main(int argc, char * argv[]) try {
	//Settings settings;
		
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
	switch (settings.captureState) {
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
		for (auto&& device : ctx.query_devices())
		{
			auto pipe = std::make_shared<rs2::pipeline>(ctx);

			rs2::config cfg;
			std::string device_name = device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
			cfg.enable_device(device_name);
			cfg.enable_all_streams();
			cfg.enable_record_to_file(settings.recordings_folder + device_name + ".bag");
			pipe->start(cfg);
			pipelines[i] = pipe;
			stream_names[i] = device_name;
			i++;
		}
		break;
	case CaptureState::PLAYING:
	{
		std::vector<std::string> figure_filenames = file_access::listFilesInFolder(settings.recordings_folder);

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
		_THROW(rs2::error("No device or file found!"));
	}
	if (settings.captureState == CaptureState::RECORDING) {
		file_access::resetFolder(settings.recordings_folder);
	}
	while (stream_names.size() < 4) {
		stream_names.push_back("");
	}
    

	// We'll keep track of the last frame of each stream available to make the presentation persistent
	std::map<int, rs2::frame> render_frames;

	texture single_color_frame;

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
    std::vector<rs2::frame_queue> filtered_datas(4);
    std::vector<rs2::frame> filtered_aligned_colors(4);
    std::vector<std::thread> processing_threads(4);
    rs2::frame_queue queue;
	for (int i = 0; i < pipelines.size(); ++i) {
		processing_threads[i] = std::thread([i, &stopped, &paused, &pipelines, &filters, &filtered_datas, &filtered_aligned_colors]() {
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
					rs2::frame filtered = depth_frame; // Does not copy the frame, only adds a reference

					rs2::frame color_frame = data.get_color_frame();
					filtered_aligned_colors[i] = color_frame;

					// Apply filters.
					for (auto&& filter : filters) {
						filtered = filter->process(filtered);
					}

					// Push filtered & original data to their respective queues
					filtered_datas[i].enqueue(filtered);
				}
				catch (const std::exception & e) {
					std::stringstream stream;
					stream << "******************** THREAD ERROR *******************" << std::endl << e.what() << "****************************************************" <<std::endl;
				}
			}
			pipe->stop();
		});
	}


    std::vector<rs2::pointcloud> filtered_pc(4);
    std::vector<rs2::frame> active_frames(4);
    std::vector<rs2::points> filtered_points(4);
    std::vector<rs2::colorizer> color_maps(4);
    texture color_image;

    bool align_frames = false;

	while (window_main)
	{
		const float w = static_cast<float>(window_main.width());
		const float h = static_cast<float>(window_main.height());
		const int w_half = w / 2;
		const int h_half = h / 2;

		// Retina display (Mac OS) have double the pixel density
		int w2, h2;
		glfwGetFramebufferSize(window_main, &w2, &h2);
		const bool is_retina_display = w2 == w * 2 && h2 == h * 2;

		draw_text(10, 20, stream_names[0].c_str());
		draw_text(w_half, 20, stream_names[1].c_str());
		draw_text(10, h_half + 10, stream_names[2].c_str());
		draw_text(w_half, h_half + 10, stream_names[3].c_str());

		// Flags for displaying ImGui window
		static const int flags = ImGuiWindowFlags_NoCollapse
			| ImGuiWindowFlags_NoScrollbar
			| ImGuiWindowFlags_NoSavedSettings
			| ImGuiWindowFlags_NoTitleBar
			| ImGuiWindowFlags_NoResize
			| ImGuiWindowFlags_NoMove;
		// UI Rendering
		ImGui_ImplGlfw_NewFrame(1);
		ImGui::SetNextWindowSize({ w, h });
		ImGui::Begin("window_main", nullptr, flags);

		ImGui::SameLine(ImGui::GetWindowWidth() - 160);
		std::string pauseResumeText = "Pause";
		if (paused) {
			pauseResumeText = "Resume";
		}
		if (ImGui::Button(pauseResumeText.c_str())) {
			paused = !paused;
		}

		ImGui::SameLine(ImGui::GetWindowWidth() - 100);
		if (ImGui::Button("Save frames")) {
			file_access::isDirectory(settings.captures_folder, true);
			// Write images to disk
			for (int i = 0; i < 4; ++i) {
				auto vf = active_frames[i].as<rs2::video_frame>();

				auto filename = std::to_string(vf.get_timestamp());
				//filename = filename.erase(filename.find(".bag"), filename.length());

				std::stringstream png_file;
				png_file << settings.captures_folder << "frame_" << filename << ".png";
				stbi_write_png(png_file.str().c_str(), vf.get_width(), vf.get_height(),
					vf.get_bytes_per_pixel(), vf.get_data(), vf.get_stride_in_bytes());

				std::string ply_file = settings.captures_folder + "frame_" + filename + ".ply";
				filtered_points[i].export_to_ply(ply_file, vf);

				std::cout << "Saved frame " << i << " to \"" << png_file.str() << "\""
					<< std::endl;
				std::cout << "Saved frame " << i << " to \"" << ply_file << "\""
					<< std::endl;
			}
		}

		ImGui::SameLine(ImGui::GetWindowWidth() - 260);
		if (ImGui::Button("Align frames")) {
			paused = true;
			auto points = filtered_points[0];

			/*auto vertices = points.get_vertices();              // get vertices
			auto tex_coords = points.get_texture_coordinates(); // and texture coordinates
			for (int i = 0; i < points.size(); i++)
			{
				if (vertices[i].z)
				{
					// upload the point and texture coordinates only for points we have depth data for
					glVertex3fv(vertices[i]);
					glTexCoord2fv(tex_coords[i]);
				}
			}*/

			paused = false;
			std::cout << "Aligned the current lframes" << std::endl;
		}

		ImGui::SameLine(ImGui::GetWindowWidth() - 360);
		if (ImGui::Button("Switch view")) {
			int s = (int) settings.renderState;
			s = (s + 1) % (int)RenderState::COUNT;
			settings.renderState = (RenderState) s;

			std::cout << "Switching render state to " << std::to_string((int)settings.renderState) << std::endl;
		}

		ImGui::End();
		ImGui::Render();

		render_frames = std::map<int, rs2::frame>();

		switch (settings.renderState) {
		case RenderState::MULTI_POINTCLOUD:
		{
			// Draw the pointclouds
			for (int i = 0; i < pipelines.size() && i < 4; ++i)
			{
				rs2::frame f;
				if (!paused && filtered_datas[i].poll_for_frame(&f)) { // Try to take the depth and points from the queue
					filtered_points[i] = filtered_pc[i].calculate(f);  // Generate pointcloud from the depth data
					active_frames[i] = color_maps[i].process(f);       // Colorize the depth frame with a color map
					filtered_pc[i].map_to(active_frames[i]);           // Map the colored depth to the point cloud
				}
				else if (!paused) {
					i -= 1; // avoids stuttering
					continue;
				}

				if (active_frames[i] && filtered_points[i]) {
					view_orientation.tex.upload(active_frames[i]);   //  and upload the texture to the view (without this the view will be B&W)
					if (is_retina_display) {
						glViewport(w * (i % 2), h - (h * (i / 2)), w, h);
					}
					else {
						glViewport(w_half * (i % 2), h_half - (h_half * (i / 2)), w_half, h_half);
					}

					if (filtered_aligned_colors[i]) {
						draw_pointcloud_and_colors(w_half, h_half, view_orientation, filtered_points[i], filtered_aligned_colors[i], 0.2f);
					}
					else {
						draw_pointcloud(w_half, h_half, view_orientation, filtered_points[i]);
					}
				}
			}
		}
		break;

		case RenderState::SINGLE_COLOR:
		{
			auto color = filtered_aligned_colors[0];
			auto data = color.get_data();
			if (color != NULL) {
				single_color_frame.render(color, { 0,0, window_main.width() , window_main.height() * 0.95f });
			}

			glBegin(GL_LINES);

			glColor3f(1, 0, 0);

			glVertex2d(50, 50);
			glVertex2d(150, 50);
			glVertex2d(150, 50);
			glVertex2d(150, 150);
			glVertex2d(150, 150);
			glVertex2d(50, 150);
			glVertex2d(50, 150);
			glVertex2d(50, 50);

			glEnd();
			break;
		}
		}
	}

    stopped = true;
    for (auto &thread : processing_threads) {
        thread.join();
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