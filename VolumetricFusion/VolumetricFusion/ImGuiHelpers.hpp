#pragma once
#include <atomic>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

#if APPLE
#include "../example.hpp"
#else
#include "example.hpp"
#endif
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

namespace imgui_helpers {

	void initialize(window& window_main, int& w2, int& h2, std::vector<std::string>& stream_names, const int& width_half, const int& height_half, const float& width, const float& height)
	{
		draw_text(10, 20, stream_names[0].c_str());
		draw_text(width_half, 20, stream_names[1].c_str());
		draw_text(10, height_half + 10, stream_names[2].c_str());
		draw_text(width_half, height_half + 10, stream_names[3].c_str());

		// Flags for displaying ImGui window
		static const int flags = ImGuiWindowFlags_NoCollapse
			| ImGuiWindowFlags_NoScrollbar
			| ImGuiWindowFlags_NoSavedSettings
			| ImGuiWindowFlags_NoTitleBar
			| ImGuiWindowFlags_NoResize
			| ImGuiWindowFlags_NoMove;
		// UI Rendering
		ImGui_ImplGlfw_NewFrame(1);
		ImGui::SetNextWindowSize({ width, height });
		ImGui::Begin("window_main", nullptr, flags);
	}

	void finalize()
	{
		ImGui::End();
		ImGui::Render();
	}

	template<typename F>
	void addTopBarButton(const char* text, F& onButtonPressedAction, float pos_x = 0.0F, float spacing_w = -1.0F) {
		ImGui::SameLine(pos_x, spacing_w);
		if (ImGui::Button(text)) {
			onButtonPressedAction();
		}
	}

	void addSwitchViewButton(RenderState &renderState, std::atomic_bool& calibrate)
	{
	    const auto callback = [&]() {
          int s = (int)renderState;
          s = (s + 1) % (int)RenderState::COUNT;
          renderState = (RenderState)s;

          calibrate = false;

          std::cout << "Switching render state to " << std::to_string((int)renderState) << std::endl;
        };
		addTopBarButton("Switch view", callback);
	}

	void addToggleButton(const char* offText, const char* onText, std::atomic_bool &variable) {
		const char* text = offText;
		if (variable) {
			text = onText;
		}
		const auto callback = [&]() {
          variable = !variable;
        };
		addTopBarButton(text, callback);
	}

	void addPauseResumeToggle(std::atomic_bool& paused)
	{
		imgui_helpers::addToggleButton("Pause", "Resume", paused);
	}
	
	void addCalibrateToggle(std::atomic_bool& calibrate)
	{
		imgui_helpers::addToggleButton("Calibrate", "Stop Calibration", calibrate);
	}

	void addAlignPointCloudsButton(std::atomic_bool& paused, std::map<int, rs2::points>& filtered_points)
	{
        const auto callback = [&]() {
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
        };
		addTopBarButton("Align Pointclouds", callback);
	}

	void addSaveFramesButton(std::string& captures_folder, std::map<int, std::shared_ptr<rs2::pipeline>>& pipelines, std::map<int, rs2::frame>& colorized_depth_frames, std::map<int, rs2::points>& filtered_points) {
		const auto callback = [&]() {
          file_access::isDirectory(captures_folder, true);
          // Write images to disk
          for (int i = 0; i < pipelines.size(); ++i) {
              auto vf = colorized_depth_frames[i].as<rs2::video_frame>();

              auto filename = std::to_string(vf.get_timestamp());
              //filename = filename.erase(filename.find(".bag"), filename.length());

              std::stringstream png_file;
              png_file << captures_folder << "frame_" << filename << ".png";
              stbi_write_png(png_file.str().c_str(), vf.get_width(), vf.get_height(),
                             vf.get_bytes_per_pixel(), vf.get_data(), vf.get_stride_in_bytes());

              std::string ply_file = captures_folder + "frame_" + filename + ".ply";
              filtered_points[i].export_to_ply(ply_file, vf);

              std::cout << "Saved frame " << i << " to \"" << png_file.str() << "\""
                        << std::endl;
              std::cout << "Saved frame " << i << " to \"" << ply_file << "\""
                        << std::endl;
          }
        };
	    addTopBarButton("Save Frames", callback);
	}

	void addGenerateCharucoDiamond(std::string& charuco_folder) {
		const auto callback = [&]() {
			file_access::isDirectory(charuco_folder, true);
			for (int i = 0; i < 6; i++) {
				cv::Mat diamondImage;
				cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
				cv::aruco::drawCharucoDiamond(dictionary, cv::Vec4i((i * 4) + 0, (i * 4) + 1, (i * 4) + 2, (i * 4) + 3), 200, 120, diamondImage);

				imshow("board", diamondImage);
				auto filename = charuco_folder + "diamond_" + std::to_string(i) + ".png";

				try {
					cv::imwrite(filename, diamondImage);
				}
				catch (std::runtime_error & ex) {
					fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
					return 1;
				}
			}
		};

		addTopBarButton("Generate ChAruCo Diamonds", callback);
	}

	void addGenerateCharucoBoard(std::string& charuco_folder) {
		const auto callback = [&]() {
			file_access::isDirectory(charuco_folder, true);
			for (int i = 0; i < 6; i++) {

				cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
				cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(5, 5, 0.04, 0.02, dictionary);
				cv::Mat boardImage;
				board->draw(cv::Size(5000, 5000), boardImage, 10, 1);

				imshow("board", boardImage);
				auto filename = charuco_folder + "board_" + std::to_string(i) + ".png";

				try {
					cv::imwrite(filename, boardImage);
				}
				catch (std::runtime_error & ex) {
					fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
					return 1;
				}
			}
		};

		addTopBarButton("Generate ChAruCo Boards", callback);
	}
}