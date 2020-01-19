#pragma once
#ifndef _IMGUI_HELPERS_HEADER
#define _IMGUI_HELPERS_HEADER

#include <atomic>
#include <string>
#include <memory>
#include "CaptureDevice.hpp"
#include "Data.hpp"
#include "optimization/OptimizationProblem.hpp"
#include "Enums.hpp"
#include "Voxelgrid.hpp"
#include "camera.hpp"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_internal.h"

const char* glsl_version = "#version 330";

namespace vc::imgui {
	const unsigned int WINDOW_FLAGS = 0
		//| ImGuiWindowFlags_NoMove
		| ImGuiWindowFlags_AlwaysAutoResize
		;

	class ProgramGUI {
		vc::enums::RenderState* renderState;
		void (*calibrationCallback)();
		std::atomic_bool* calibrateCameras;
		vc::io::Camera* camera;

	public:
		bool showCoordinateSystem;
		
		ProgramGUI(vc::enums::RenderState* renderState, void (*calibrationCallback)(), std::atomic_bool* calibrateCameras, vc::io::Camera* camera) :
			renderState(renderState), calibrationCallback(calibrationCallback), calibrateCameras(calibrateCameras), camera(camera) {}

		void render() {
			ImGui::Begin("Program Info", nullptr, WINDOW_FLAGS);

			if (ImGui::TreeNode("View")) {
				for (int n = 0; n < (int)vc::enums::RenderState::COUNT; n++)
				{
					auto nn = static_cast<vc::enums::RenderState>(n);
					auto name = vc::enums::renderStateToName[nn];
					//std::stringstream ss;
					//ss << "View: " << name;
					auto value = (int)*renderState == n;
					if (ImGui::Selectable(name, value)) {
						*renderState = nn;
					}
				}
				ImGui::TreePop();
			}
						
			if (*renderState != vc::enums::RenderState::ONLY_COLOR && *renderState != vc::enums::RenderState::ONLY_DEPTH) {
				ImGui::Separator();
				bool checked = calibrateCameras->load();
				if (ImGui::Checkbox("Calibrate", &checked)) {
					calibrateCameras->store(checked);
					calibrationCallback();
				}

				ImGui::Separator();
				ImGui::Checkbox("Coordinate system", &showCoordinateSystem);

				ImGui::Separator();

				float width = ImGui::GetWindowWidth();
				ImGui::Text("Camera properties");
				ImGui::PushItemWidth(width * 0.25f);
				ImGui::InputFloat("X", &camera->Position.x);
				ImGui::SameLine(0.33f * width);
				ImGui::InputFloat("Y", &camera->Position.y);
				ImGui::SameLine(0.66f * width);
				ImGui::InputFloat("Z", &camera->Position.z);

				ImGui::PushItemWidth(width * 0.33f);
				ImGui::InputFloat("Yaw", &camera->Yaw);
				ImGui::SameLine(0.5f * width);
				ImGui::InputFloat("Pitch", &camera->Pitch);

				ImGui::PushItemWidth(width * 0.65f);
				ImGui::SliderFloat("Speed", &camera->MovementSpeed, 1.0f, 10.0f);
				ImGui::SliderFloat("Sensitivity", &camera->MouseSensitivity, 0.01f, 1.0f);
				ImGui::SliderFloat("Zoom", &camera->Zoom, 1.0f, 90.0f);

				if (ImGui::Button("Reset camera")) {
					*camera = vc::io::Camera(glm::vec3(0.0f, 0.0f, -1.0f));
				}
			}

			ImGui::Separator();
			ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
			ImGui::End();
		}
	};

	class PipelineGUI {
	private:
		std::shared_ptr<vc::capture::CaptureDevice> pipeline;
	
	public:
		float alpha = 1.0f;

		PipelineGUI(std::shared_ptr<vc::capture::CaptureDevice> pipeline) : pipeline(pipeline) {}

		void render() {
			ImGui::Begin(("Pointcloud " + pipeline->data->deviceName).c_str(), nullptr, WINDOW_FLAGS);
			ImGui::Text("Editable settings of pointclouds.");

			ImGui::SliderFloat("Alpha", &alpha, 0.0f, 1.0f);
			
			ImGui::End();
		}
	};

	class AllPipelinesGUI {
	private:
		std::vector<std::shared_ptr<vc::capture::CaptureDevice>>* pipelines;

	public:
		float overallAlpha = 1.0f;
		float rotationSpeed = 0.0f;
		std::vector<float> alphas;

		AllPipelinesGUI(std::vector<std::shared_ptr<vc::capture::CaptureDevice>>* pipelines) : 
			pipelines(pipelines) 
		{
			for (int i = 0; i < 4; i++)
			{
				alphas.emplace_back(overallAlpha);
			}
		}

		void render() {
			ImGui::Begin("Pointclouds", nullptr, WINDOW_FLAGS);
			ImGui::Text("Editable settings of all pointclouds.");

			if (ImGui::SliderFloat("Overall alpha", &overallAlpha, 0.0f, 1.0f)) {
				for (int i = 0; i < 4; i++)
				{
					alphas[i] = overallAlpha;
				}
			}

			for (int i = 0; i < pipelines->size(); i++)
			{
				ImGui::Separator();
				std::stringstream ss;
				ss << "Pipeline: " << (*pipelines)[i]->data->deviceName;
				ImGui::Text(ss.str().c_str());
				ss = std::stringstream();
				ss << "Alpha" << "##" << i;
				ImGui::SliderFloat(ss.str().c_str(), &alphas[i], 0.0f, 1.0f);
			}

			//ImGui::Separator();
			//ImGui::SliderFloat("Rotation speed", &rotationSpeed, -1.0f, 1.0f);
			//ImGui::SetWindowPos(ImVec2(5, 5), true);
			ImGui::End();
		}
	};

	class OptimizationProblemGUI {
	private:
		vc::optimization::OptimizationProblem* optimizationProblem;

	public:
		bool highlightMarkerCorners = true;

		OptimizationProblemGUI(vc::optimization::OptimizationProblem* optimizationProblem) :
			optimizationProblem(optimizationProblem) {}

		void render() {
			ImGui::Begin("Optimization", nullptr, WINDOW_FLAGS);
			ImGui::Text("Editable settings of the camera calibration.");

			ImGui::Checkbox("Highlight marker corners", &highlightMarkerCorners);
			if (ImGui::Button("Reset")) {
				optimizationProblem->reset();
			}
			ImGui::End();
		}
	};

	class VoxelgridGUI {
	private:
		vc::fusion::Voxelgrid* voxelgrid;
		const float truncationDistanceRange = 1.0f;
	public:
		bool renderVoxelgrid = true;
		bool fuse = false;
		bool marchingCubes = false;
		float truncationDistance = truncationDistanceRange / 2.0f;

		VoxelgridGUI(vc::fusion::Voxelgrid* voxelgrid) : voxelgrid(voxelgrid){}

		void render() {
			ImGui::Begin("Voxelgrid", nullptr, WINDOW_FLAGS);
			ImGui::Text("Editable settings of the voxelgrid.");

			ImGui::Checkbox("Render voxelgrid", &renderVoxelgrid);

			ImGui::Checkbox("Fuse", &fuse);

			ImGui::Checkbox("Marching cubes after frame", &marchingCubes);			

			ImGui::Separator();

			fuse = ImGui::SliderFloat("Truncation distance", &truncationDistance, 0, truncationDistanceRange);

			ImGui::End();
		}
	};

	ImGuiIO init(GLFWwindow* window, int window_width, int window_height) {
		static bool isInitialized = false;

			ImGui::CreateContext();
			ImGuiIO& io = ImGui::GetIO(); (void)io;
			io.DisplaySize = ImVec2(window_width, window_height);
			unsigned char* pixels;
			int width = 100;
			int height = 100;
			io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
			ImGui_ImplGlfw_InitForOpenGL(window, true);
			ImGui_ImplOpenGL3_Init(glsl_version);

			ImGui::StyleColorsLight();
			isInitialized = true;
			return io;
	}

	void startFrame(ImGuiIO* io, int window_width, int window_height) {
		io->DisplaySize = ImVec2(window_width, window_height);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
	}

	void render() {
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}

	void terminate() {
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}
}
#endif // !_IMGUI_HELPERS_HEADER