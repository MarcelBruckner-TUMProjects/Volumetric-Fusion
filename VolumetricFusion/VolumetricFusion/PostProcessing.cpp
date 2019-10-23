// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
// Include short list of convenience functions for rendering
#if APPLE
#include "../example.hpp"
#include "FileAccess.hpp"
#include "CaptureDevice.h"
#else
#include "example.hpp"
#include "VolumetricFusion/FileAccess.hpp"
#include "VolumetricFusion/CaptureDevice.h"
#endif

#include <map>
#include <string>
#include <thread>
#include <atomic>

#include <imgui.h>
#include "imgui_impl_glfw.h"


// Helper functions for rendering the UI
void render_ui(float w, float h);
// Helper function for getting data from the queues and updating the view
void update_data(rs2::frame_queue& data, rs2::frame& depth, rs2::points& points, rs2::pointcloud& pc, glfw_state& view, rs2::colorizer& color_map);

int main(int argc, char * argv[]) try
{
    // Recordings filename
    std::string filename = "recording_peace_kevin.bag";
    auto recorded = file_access::exists_test(filename);
    if (!recorded) {
        throw std::runtime_error("Missing file: " + filename);
    }

    // Create a simple OpenGL window for rendering:
    window app(1280, 720, "VolumetricFusion Post-Processing");

    ImGui_ImplGlfw_Init(app, false);


    // Construct an object to manage view state
    glfw_state view_orientation{};
    // Let the user control and manipulate the scenery orientation
    register_glfw_callbacks(app, view_orientation);

    // Declare pointcloud objects, for calculating pointclouds and texture mappings
    rs2::pointcloud original_pc;
    rs2::pointcloud filtered_pc;

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    rs2::config cfg;
    // Use a configuration object to request only depth from the pipeline
    //cfg.enable_stream(RS2_STREAM_DEPTH, 640, 0, RS2_FORMAT_Z16, 30);
    cfg.enable_device_from_file(filename);

    // Start streaming with the above configuration
    pipe.start(cfg);

    rs2::context ctx;
    rs2::device device = pipe.get_active_profile().get_device();
    CaptureDevice*  capture_device = new CaptureDevice(ctx, device);

    // Declare filters
    rs2::decimation_filter dec_filter(5);  // Decimation - reduces depth frame density
    rs2::threshold_filter thr_filter(0.3f, 2.3f);   // Threshold  - removes values outside recommended range
    //rs2::spatial_filter spat_filter;    // Spatial    - edge-preserving spatial smoothing
    //rs2::temporal_filter temp_filter;   // Temporal   - reduces temporal noise

    // Declare disparity transform from depth to disparity and vice versa
    //const std::string disparity_filter_name = "Disparity";
    //rs2::disparity_transform depth_to_disparity(true);
    //rs2::disparity_transform disparity_to_depth(false);

    // Initialize a vector that holds filters and their options
    //std::vector<filter_options> filters;
    std::vector<rs2::filter*> filters;
    filters.emplace_back(&thr_filter);
    filters.emplace_back(&dec_filter);

    // Declaring two concurrent queues that will be used to enqueue and dequeue frames from different threads
    rs2::frame_queue original_data;
    rs2::frame_queue filtered_data;

    // Declare depth colorizer for pretty visualization of depth data
    rs2::colorizer color_map;

    // Atomic boolean to allow thread safe way to stop the thread
    std::atomic_bool stopped(false);

    // Create a thread for getting frames from the device and process them
    // to prevent UI thread from blocking due to long computations.
    std::thread processing_thread([&]() {
        while (!stopped) //While application is running
        {
            rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
            rs2::frame depth_frame = data.get_depth_frame(); //Take the depth frame from the frameset
            if (!depth_frame) // Should not happen but if the pipeline is configured differently
                return;       //  it might not provide depth and we don't want to crash

            rs2::frame filtered = depth_frame; // Does not copy the frame, only adds a reference

            // Apply filters.
            bool revert_disparity = false;
            for (auto&& filter : filters)
            {
                filtered = filter->process(filtered);
            }

            // Push filtered & original data to their respective queues
            filtered_data.enqueue(filtered);
            original_data.enqueue(depth_frame);
        }
    });

    // Declare objects that will hold the calculated pointclouds and colored frames
    // We save the last set of data to minimize flickering of the view
    rs2::frame colored_depth;
    rs2::frame colored_filtered;
    rs2::points original_points;
    rs2::points filtered_points;

    // Save the time of last frame's arrival
    auto last_time = std::chrono::high_resolution_clock::now();
    // Maximum angle for the rotation of the pointcloud
    const double max_angle = 15.0;
    // We'll use rotation_velocity to rotate the pointcloud for a better view of the filters effects
    float rotation_velocity = 0.3f;

    while (app)
    {
        float w = static_cast<float>(app.width());
        float h = static_cast<float>(app.height());

        // Render the GUI
        render_ui(w, h);

        // Try to get new data from the queues and update the view with new texture
        update_data(original_data, colored_depth, original_points, original_pc, view_orientation, color_map);
        update_data(filtered_data, colored_filtered, filtered_points, filtered_pc, view_orientation, color_map);

        draw_text(10, 50, "Original");
        draw_text(static_cast<int>(w / 2), 50, "Filtered");

        // Draw the pointclouds of the original and the filtered frames (if the are available already)
        if (colored_depth && original_points)
        {
            glViewport(0, int(h) / 2, int(w) / 2, int(h) / 2);
            draw_pointcloud(int(w) / 2, int(h) / 2, view_orientation, original_points);
        }
        if (colored_filtered && filtered_points)
        {
            glViewport(int(w) / 2, int(h) / 2, int(w) / 2, int(h) / 2);
            draw_pointcloud(int(w) / 2, int(h) / 2, view_orientation, filtered_points);
        }
    }

    // Signal the processing thread to stop, and join
    // (Not the safest way to join a thread, please wrap your threads in some RAII manner)
    stopped = true;
    processing_thread.join();

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

void update_data(rs2::frame_queue& data, rs2::frame& colorized_depth, rs2::points& points, rs2::pointcloud& pc, glfw_state& view, rs2::colorizer& color_map)
{
    rs2::frame f;
    if (data.poll_for_frame(&f))  // Try to take the depth and points from the queue
    {
        points = pc.calculate(f); // Generate pointcloud from the depth data
        colorized_depth = color_map.process(f);     // Colorize the depth frame with a color map
        pc.map_to(colorized_depth);         // Map the colored depth to the point cloud
        view.tex.upload(colorized_depth);   //  and upload the texture to the view (without this the view will be B&W)
    }
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
