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

#include "imgui/imgui.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_internal.h"

const char* glsl_version = "#version 330";

namespace vc::imgui {
	class ProgramGUI {
		vc::enums::RenderState* renderState;
		void (*calibrationCallback)();
		std::atomic_bool* calibrateCameras;
		
	public:
		ProgramGUI(vc::enums::RenderState* renderState, void (*calibrationCallback)(), std::atomic_bool* calibrateCameras) :
			renderState(renderState), calibrationCallback(calibrationCallback), calibrateCameras(calibrateCameras) {}

		void render() {
			ImGui::Begin("Program Info");

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
			
			ImGui::Separator();

			bool checked = calibrateCameras->load();
			if (ImGui::Checkbox("Calibrate", &checked)) {
				calibrateCameras->store(checked);
				calibrationCallback();
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
			ImGui::Begin(("Pointcloud " + pipeline->data->deviceName).c_str());
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
			ImGui::Begin("Pointclouds");
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
			ImGui::Begin("Optimization");
			ImGui::Text("Editable settings of the camera calibration.");

			ImGui::Checkbox("Highlight marker corners", &highlightMarkerCorners);
			if (ImGui::Button("Reset")) {
				optimizationProblem->reset();
			}
			ImGui::End();
		}
	};

	void init(GLFWwindow* window, int* window_width, int* window_height) {
		static bool isInitialized = false;

		if (!isInitialized) {
			ImGui::CreateContext();
			ImGuiIO& io = ImGui::GetIO(); (void)io;
			io.DisplaySize = ImVec2(*window_width, *window_height);
			unsigned char* pixels;
			int width = 100;
			int height = 100;
			io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
			ImGui_ImplGlfw_InitForOpenGL(window, true);
			ImGui_ImplOpenGL3_Init(glsl_version);

			ImGui::StyleColorsLight();
			isInitialized = true;
		}
	}

	void startFrame() {
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
