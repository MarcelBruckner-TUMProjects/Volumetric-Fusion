#pragma region Includes
// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "stb_image.h"

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
// Include short list of convenience functions for rendering

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>

#include <map>
#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <filesystem>


#include <imgui.h>
#include "imgui_impl_glfw.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "Enums.hpp"
using namespace vc::enums;

#include "ImGuiHelpers.hpp"
#include "processing.hpp"

#include "Eigen/Dense"
#include "Settings.hpp"
#include "Data.hpp"
#include "CaptureDevice.hpp"
#include "Rendering.hpp"
#if APPLE
#include "FileAccess.hpp"
#include <glut.h>
#else
#include "VolumetricFusion/FileAccess.hpp"
#endif
#include <signal.h>


#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "camera.hpp"
#include "shader.hpp"

#pragma endregion

template<typename T, typename V>
std::vector<T> extractKeys(std::map<T, V> const& input_map);

template<typename T>
std::vector<T> findOverlap(std::vector<T> a, std::vector<T> b);

void my_function_to_handle_aborts(int signalNumber) {
	std::cout << "Something aborted" << std::endl;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;

int main(int argc, char* argv[]) try {
	
	vc::settings::FolderSettings folderSettings;
	folderSettings.recordingsFolder = "allCameras/";
	vc::settings::State state = vc::settings::State(CaptureState::PLAYING, RenderState::ONLY_COLOR);

	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	
	// glfw window creation
	// --------------------
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);

	// tell GLFW to capture our mouse
	//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	ImGui_ImplGlfw_Init(window, false);
	
	//vc::rendering::Rendering rendering(app, viewOrientation);

	rs2::context ctx; // Create librealsense context for managing devices
	//std::vector<vc::data::Data> datas;
	std::vector<std::shared_ptr<  vc::capture::CaptureDevice>> pipelines;
	std::vector<std::string> streamNames;

	if (state.captureState == CaptureState::RECORDING || state.captureState == CaptureState::STREAMING) {
		auto devices = ctx.query_devices();
		if (state.captureState == CaptureState::RECORDING) {
			if (devices.size() > 0) {
				vc::file_access::resetDirectory(folderSettings.recordingsFolder, true);
			}
		}
		int i = 0;
		for (auto&& device : ctx.query_devices())
		{
			if (state.captureState == CaptureState::RECORDING) {
				pipelines.emplace_back(std::make_shared < vc::capture::RecordingCaptureDevice>(ctx, device, folderSettings.recordingsFolder));
			}
			else if (state.captureState == CaptureState::STREAMING) {
				pipelines.emplace_back(std::make_shared < vc::capture::StreamingCaptureDevice>(ctx, device));
			}
			i++;
		}
	}
	else if (state.captureState == CaptureState::PLAYING) {
		std::vector<std::string> filenames = vc::file_access::listFilesInFolder(folderSettings.recordingsFolder);

		for (int i = 0; i < filenames.size() && i < 4; i++)
		{
			pipelines.emplace_back(std::make_shared < vc::capture::PlayingCaptureDevice>(ctx, filenames[i]));
		}
	}

	if (pipelines.size() <= 0) {
		throw(rs2::error("No device or file found!"));
	}
	while (streamNames.size() < 4) {
		streamNames.emplace_back("");
	}

	// Create a thread for getting frames from the device and process them
	// to prevent UI thread from blocking due to long computations.
	std::atomic_bool stopped(false);
	std::atomic_bool paused(false);

	// Create custom depth processing block and their output queues:
	/*std::map<int, rs2::frame_queue> depth_processing_queues;
	std::map<int, std::shared_ptr<rs2::processing_block>> depth_processing_blocks;*/

	// Calculated relative transformations between cameras per frame
	std::map<int, Eigen::MatrixXd> relativeTransformations = {
		{0, Eigen::MatrixXd::Identity(4,4)}
	};

	// Camera calibration thread
	std::thread calibrationThread;
	std::atomic_bool calibrateCameras = true;

	for (int i = 0; i < pipelines.size(); i++) {
		pipelines[i]->startPipeline();
		pipelines[i]->resumeThread();
		pipelines[i]->calibrate(calibrateCameras);
	}

#pragma region Camera Calibration Thread

	calibrationThread = std::thread([&pipelines, &stopped, &calibrateCameras, &relativeTransformations]() {
		while (!stopped) {
			if (!calibrateCameras) {
				continue;
			}

			for (int i = 1; i < pipelines.size(); i++) {
				{
					//if (!pipelines[0]->processing->hasMarkersDetected || !pipelines[i]->processing->hasMarkersDetected) {
					if (!pipelines[i]->processing->hasMarkersDetected) {
						continue;
					}

					Eigen::Matrix4d baseToMarkerTranslation = pipelines[i == 0 ? 1 : 0]->processing->translation;
					Eigen::Matrix4d baseToMarkerRotation = pipelines[i == 0 ? 1 : 0]->processing->rotation;

                    //Eigen::Matrix4d markerToRelativeTranslation = pipelines[i]->processing->translation.inverse();
                    //Eigen::Matrix4d markerToRelativeRotation = pipelines[i]->processing->rotation.inverse();
                    Eigen::Matrix4d markerToRelativeTranslation = pipelines[i]->processing->translation;
                    Eigen::Matrix4d markerToRelativeRotation = pipelines[i]->processing->rotation;

					//Eigen::Matrix4d relativeTransformation = markerToRelativeTranslation * markerToRelativeRotation * baseToMarkerRotation * baseToMarkerTranslation;
					Eigen::Matrix4d relativeTransformation = (
						//markerToRelativeTranslation * markerToRelativeRotation * baseToMarkerRotation * baseToMarkerTranslation
						(markerToRelativeTranslation * markerToRelativeRotation).inverse() * baseToMarkerTranslation * baseToMarkerRotation
					);

					relativeTransformations[i] = relativeTransformation;

				/*	std::stringstream ss;
					ss << "************************************************************************************" << std::endl;
					ss << "Devices " << i << ", " << i << std::endl << std::endl;
					ss << "Translations: " << std::endl << baseToMarkerTranslation << std::endl << markerToRelativeTranslation << std::endl << std::endl;
					ss << "Rotations: " << std::endl << baseToMarkerRotation << std::endl << markerToRelativeRotation << std::endl << std::endl;
					ss << "Combined: " << std::endl << relativeTransformation << std::endl;
					std::cout << ss.str();*/
				}
			}
		}
	});
#pragma endregion

#pragma region Main loop

	while (!glfwWindowShouldClose(window))
	{
		// per-frame time logic
		// --------------------
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		int width, height;
		glfwGetWindowSize(window, &width, &height);
		
		// Retina display (Mac OS) have double the pixel density
		int w2, h2;
		glfwGetFramebufferSize(window, &w2, &h2);
		const bool isRetinaDisplay = w2 == width * 2 && h2 == width * 2;

		const float aspect = 1.0f * width / height;

		// input
		// -----
		processInput(window);

		vc::rendering::startFrame(window);

		for (int i = 0; i < pipelines.size() && i < 4; ++i)
		{
			switch (state.renderState) {
			case RenderState::ONLY_COLOR:
			{
				int x = i % 2;
				int y = floor(i / 2);
				//std::cout << i << ": x=" << x << " - y=" << y << std::endl;
				if (pipelines[i]->data->filteredColorFrames) {
					pipelines[i]->rendering->renderOnlyColor(pipelines[i]->data->filteredColorFrames, x, y, aspect);
				}
				break;
			}
			}
		}
		
		

		int numPipelines = pipelines.size();


	/*	vc::imgui_helpers::initialize(streamNames, width, height);
		vc::imgui_helpers::addSwitchViewButton(state.renderState, calibrateCameras);
		if (vc::imgui_helpers::addPauseResumeToggle(paused)) {
			for (int i = 0; i < pipelines.size(); i++)
			{
				pipelines[i]->paused->store(paused);
			}
		}
		
		if (state.renderState != RenderState::ONLY_DEPTH) {
			if (vc::imgui_helpers::addCalibrateToggle(calibrateCameras)) {
				for (int i = 0; i < pipelines.size(); i++)
				{
					pipelines[i]->calibrateCameras->store(calibrateCameras);
				}
			}
		}

		vc::imgui_helpers::addGenerateCharucoDiamond(folderSettings.charucoFolder);
		vc::imgui_helpers::addGenerateCharucoBoard(folderSettings.charucoFolder);
		vc::imgui_helpers::finalize();*/

		//switch (state.renderState) {
		//case RenderState::COUNT:
		//case RenderState::MULTI_POINTCLOUD:
		//{
		//	// Draw the pointclouds
		//	for (int i = 0; i < pipelines.size() && i < 4; ++i)
		//	{
		//		if (pipelines[i]->data->colorizedDepthFrames && pipelines[i]->data->points) {
		//			// TODO MULTI POINTCLOUDS 
		//		}
		//	}
		//}
		//break;

		//case RenderState::CALIBRATED_POINTCLOUD:
		//{
		//	if (!calibrateCameras)
		//	if (isRetinaDisplay) {
		//		glViewport(0, 0, width * 2, height * 2);
		//	}
		//	else {
		//		glViewport(0, 0, width, height);
		//	}

		//	for (int i = 0; i < pipelines.size() && i < 4; ++i)
		//	{
		//		// TODO CALIBRATED POINTCLOUDS
		//	}
		//}
		//break;

		//case RenderState::ONLY_COLOR:
		//{
		//	for (int i = 0; i < pipelines.size() && i < 4; ++i)
		//	{
		//		if (pipelines[i]->data->filteredColorFrames != false) {
		//			// TODO ONLY COLOR
		//		}
		//	}

		//	break;
		//}

		//case RenderState::ONLY_DEPTH:
		//{
		//	for (int i = 0; i < pipelines.size() && i < 4; ++i)
		//	{
		//		if (pipelines[i]->data->filteredDepthFrames != false) {
		//			// TODO ONLY DEPTH
		//		}
		//	}

		//	break;
		//}
		//}

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
#pragma endregion

#pragma region Final cleanup

	stopped.store(true);
	for (int i = 0; i < pipelines.size(); i++) {
		pipelines[i]->stopThread();
		pipelines[i]->thread->join();
	}
	calibrationThread.join();
#pragma endregion
	glfwTerminate();
	//std::exit(EXIT_SUCCESS);
	return EXIT_SUCCESS;
}
#pragma region Error handling
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
#pragma endregion

#pragma region Helper functions

template<typename T, typename V>
std::vector<T> extractKeys(std::map<T, V> const& input_map) {
	std::vector<T> retval;
	for (auto const& element : input_map) {
		retval.emplace_back(element.first);
	}
	return retval;
}


template<typename T>
std::vector<T> findOverlap(std::vector<T> a, std::vector<T> b) {
	std::vector<T> c;

	for (T x : a) {
		if (std::find(b.begin(), b.end(), x) != b.end()) {
			c.emplace_back(x);
		}
	}

	return c;
}


// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	}
	if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.ProcessKeyboard(FORWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.ProcessKeyboard(BACKWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.ProcessKeyboard(LEFT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.ProcessKeyboard(RIGHT, deltaTime);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}


// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

	lastX = xpos;
	lastY = ypos;

	camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	camera.ProcessMouseScroll(yoffset);
}

#pragma endregion