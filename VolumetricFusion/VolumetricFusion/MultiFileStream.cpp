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

#include <imgui.h>
#include "imgui_impl_glfw.h"


// Helper functions for rendering the UI
void render_ui(float w, float h);

int main(int argc, char * argv[]) try {
    std::string figure_filenames[] = {
            "figure_front_20191026_211104.bag",
            "figure_90_clockwise_20191026_211121.bag",
            "figure_180_clockwise_20191026_211134.bag",
            "figure_270_clockwise_20191026_211146.bag"
    };
    for (auto filename : figure_filenames) {
        if (!file_access::exists_test(filename)) {
            throw std::runtime_error("Missing file: " + filename);
        }
    }

    // Create a simple OpenGL window for rendering:
    window app(1280, 960, "VolumetricFusion - MultiStreamViewer");

    ImGui_ImplGlfw_Init(app, false);

    // Construct an object to manage view state
    glfw_state view_orientation{};
    // Let the user control and manipulate the scenery orientation
    register_glfw_callbacks(app, view_orientation);

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
        pipe.start(cfg);
        auto profile = pipe.get_active_profile();
        //pipelines.emplace_back(std::make_tuple(i, pipe));
        pipelines[i] = std::make_tuple(i, pipe);
    }

    // Declare filters
    rs2::decimation_filter dec_filter(1);         // Decimation - reduces depth frame density
    rs2::threshold_filter thr_filter(0.0f, 1.1f); // Threshold  - removes values outside recommended range

    std::vector<rs2::filter*> filters;
    filters.emplace_back(&thr_filter);
    filters.emplace_back(&dec_filter);

    // Create a thread for getting frames from the device and process them
    // to prevent UI thread from blocking due to long computations.
    std::atomic_bool stopped(false);
    std::vector<rs2::frame_queue> filtered_datas(4);
    std::vector<std::thread> processing_threads(4);
    rs2::frame_queue queue;
    for (int i = 0; i < 4; ++i) {
        processing_threads[i] = std::thread([i, &stopped, &pipelines, &filters, &filtered_datas]() {
          auto pipe = pipelines[i].second;
          auto isi = pipelines[i].first == i;
          while (!stopped) //While application is running
          {
              rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
              rs2::frame depth_frame = data.get_depth_frame(); //Take the depth frame from the frameset
              if (!depth_frame) { // Should not happen but if the pipeline is configured differently
                  return;       //  it might not provide depth and we don't want to crash
              }
              rs2::frame filtered = depth_frame; // Does not copy the frame, only adds a reference

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
    std::vector<rs2::frame> filtered_colored(4);
    std::vector<rs2::points> filtered_points(4);
    std::vector<rs2::colorizer> color_maps(4);

    while (app)
    {
        const float w = static_cast<float>(app.width());
        const float h = static_cast<float>(app.height());
        const int w_half = w / 2;
        const int h_half = h / 2;

        //render_ui(w, h);

        draw_text(10, 50, figure_filenames[0].c_str());
        draw_text(w_half, 50, figure_filenames[1].c_str());
        draw_text(10, h_half + 10, figure_filenames[2].c_str());
        draw_text(w_half, h_half + 10, figure_filenames[3].c_str());

        for (int i = 0; i < 4; ++i)
        {
            rs2::frame f;
            if (filtered_datas[i].poll_for_frame(&f))  // Try to take the depth and points from the queue
            {
                filtered_points[i] = filtered_pc[i].calculate(f); // Generate pointcloud from the depth data
                filtered_colored[i] = color_maps[i].process(f);     // Colorize the depth frame with a color map
                filtered_pc[i].map_to(filtered_colored[i]);         // Map the colored depth to the point cloud
                view_orientation.tex.upload(filtered_colored[i]);   //  and upload the texture to the view (without this the view will be B&W)

                if (filtered_colored[i] && filtered_points[i]) {
                    glViewport(w_half * (i % 2), h_half - (h_half * (i / 2)), w_half, h_half);
                    draw_pointcloud(int(w) / 2, int(h) / 2, view_orientation, filtered_points[i]);
                }
            } else {
                i -= 1; // avoids stuttering
            }
        }
        // 15 frames per second are what I recorded the video at (avoids stuttering and reduces cpu load)
        nanosleep((const struct timespec[]){{0, 1000000000L / 15L}}, NULL);
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


void render_ui(float w, float h)
{
    // Flags for displaying ImGui window
    static const int flags = ImGuiWindowFlags_NoCollapse
                             | ImGuiWindowFlags_NoScrollbar
                             | ImGuiWindowFlags_NoSavedSettings
                             | ImGuiWindowFlags_NoTitleBar
                             | ImGuiWindowFlags_NoResize
                             | ImGuiWindowFlags_NoMove;

    ImGui_ImplGlfw_NewFrame(1);
    ImGui::SetNextWindowSize({ w, h });
    ImGui::Begin("app", nullptr, flags);

    ImGui::End();
    ImGui::Render();
}
