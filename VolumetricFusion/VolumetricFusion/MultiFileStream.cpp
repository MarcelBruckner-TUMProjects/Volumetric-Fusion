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

#include <map>
#include <string>
#include <thread>
#include <atomic>
#include <filesystem>

#include <imgui.h>
#include "imgui_impl_glfw.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <Windows.h>
#include <synchapi.h>


// Helper functions for rendering the UI
void render_ui(float w, float h);

int main(int argc, char * argv[]) try {
    std::string figure_filenames[] = {
            "figure_front_20191026_211104.bag",
            "figure_90_clockwise_20191026_211121.bag",
            "figure_180_clockwise_20191026_211134.bag",
            "figure_270_clockwise_20191026_211146.bag"
    };
    //std::string figure_filenames[] = {"test.bag","test.bag","test.bag","test.bag"};

    for (auto filename : figure_filenames) {
        if (!file_access::exists_test(filename)) {
            throw std::runtime_error("Missing file: " + filename);
        }
    }

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
    std::vector<std::pair<int, rs2::pipeline>> pipelines(4);

    // Start a streaming pipe per each connected device
    //for (auto&& dev : ctx.query_devices())
    for (int i = 0; i < 4; ++i)
    {
        rs2::pipeline pipe(ctx);
        rs2::config cfg;
        //cfg.enable_device(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
        cfg.enable_device_from_file(figure_filenames[i]);
        cfg.enable_all_streams();
        pipe.start(cfg);
        auto profile = pipe.get_active_profile();
        //pipelines.emplace_back(std::make_tuple(i, pipe));
        pipelines[i] = std::make_pair(i, pipe);
    }

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
    rs2::align align_to_color(RS2_STREAM_COLOR);
    rs2::frame_queue queue;
    for (int i = 0; i < 4; ++i) {
        processing_threads[i] = std::thread([i, &stopped, &paused, &pipelines, &filters, &filtered_datas, &filtered_aligned_colors, &align_to_color]() {
          auto pipe = pipelines[i].second;
          auto isi = pipelines[i].first == i;
          while (!stopped) //While application is running
          {
              while (paused.load()) {
                  continue;
              }

              rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera

              data = align_to_color.process(data);

              rs2::frame depth_frame = data.get_depth_frame(); //Take the depth frame from the frameset
              if (!depth_frame) { // Should not happen but if the pipeline is configured differently
                  return;       //  it might not provide depth and we don't want to crash
              }
              rs2::frame filtered = depth_frame; // Does not copy the frame, only adds a reference

              rs2::frame color_frame = data.get_color_frame();
              filtered_aligned_colors[i] = color_frame;

              // Apply filters.
              for (auto &&filter : filters) {
                  filtered = filter->process(filtered);
              }

              // Push filtered & original data to their respective queues
              filtered_datas[i].enqueue(filtered);
          }
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
        const bool is_retina_display = w2 == w*2 && h2 == h*2;

        draw_text(10, 20, figure_filenames[0].c_str());
        draw_text(w_half, 20, figure_filenames[1].c_str());
        draw_text(10, h_half + 10, figure_filenames[2].c_str());
        draw_text(w_half, h_half + 10, figure_filenames[3].c_str());

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
        ImGui::SameLine(ImGui::GetWindowWidth() - 100);
        auto button_save_frame_pressed = ImGui::Button("Save frames");
        bool button_pause_pressed = false;
        ImGui::SameLine(ImGui::GetWindowWidth() - 160);
        if (!paused) {
            button_pause_pressed = ImGui::Button("Pause");
        } else {
            button_pause_pressed = ImGui::Button("Resume");
        }
        ImGui::SameLine(ImGui::GetWindowWidth() - 260);
        auto button_align_frames_pressed = ImGui::Button("Align frames");
        ImGui::End();
        ImGui::Render();

        // Draw the pointclouds
        for (int i = 0; i < 4; ++i) {
            rs2::frame f;
            if (!paused && filtered_datas[i].poll_for_frame(&f)) { // Try to take the depth and points from the queue
                filtered_points[i] = filtered_pc[i].calculate(f);  // Generate pointcloud from the depth data
                active_frames[i] = color_maps[i].process(f);       // Colorize the depth frame with a color map
                filtered_pc[i].map_to(active_frames[i]);           // Map the colored depth to the point cloud
            } else if (!paused) {
                i -= 1; // avoids stuttering
                continue;
            }

            if (active_frames[i] && filtered_points[i]) {
                view_orientation.tex.upload(active_frames[i]);   //  and upload the texture to the view (without this the view will be B&W)
                if (is_retina_display) {
                    glViewport(w * (i % 2), h - (h * (i / 2)), w, h);
                } else {
                    glViewport(w_half * (i % 2), h_half - (h_half * (i / 2)), w_half, h_half);
                }

                if (filtered_aligned_colors[i]) {
                    draw_pointcloud_and_colors(w_half, h_half, view_orientation, filtered_points[i], filtered_aligned_colors[i], 0.2f);
                } else {
                    draw_pointcloud(w_half, h_half, view_orientation, filtered_points[i]);
                }
            }
        }

        if (button_save_frame_pressed) {
            // Write images to disk
            for (int i = 0; i < 4; ++i) {
                auto vf = active_frames[i].as<rs2::video_frame>();

                auto filename = figure_filenames[i];
                filename = filename.erase(filename.find(".bag"), filename.length());

                std::stringstream png_file;
                png_file << "frame_" << filename << ".png";
                stbi_write_png(png_file.str().c_str(), vf.get_width(), vf.get_height(),
                               vf.get_bytes_per_pixel(), vf.get_data(), vf.get_stride_in_bytes());

                std::string ply_file = "frame_" + filename + ".ply";
                filtered_points[i].export_to_ply(ply_file, vf);

                auto current_path = std::filesystem::current_path().u8string();
                current_path = current_path.erase(current_path.length() - 1, 1);

                std::cout << "Saved frame " << i << " to \"" << current_path << "/" << png_file.str() << "\""
                          << std::endl;
                std::cout << "Saved frame " << i << " to \"" << current_path << "/" << ply_file << "\""
                          << std::endl;
            }
        }

        if (button_pause_pressed) {
            paused = !paused;
        }

        if (button_align_frames_pressed) {
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


        // 15 frames per second are what I recorded the video at (avoids stuttering and reduces cpu load)
#if APPLE
		nanosleep((const struct timespec[]){{0, 1000000000L / 15L}}, NULL);
#else
		Sleep(1000000000L / 15L);
#endif
        //nanosleep((const struct timespec[]){{2, 0}}, NULL);
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