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


#include "imgui.h"
#include "imgui_impl_glfw.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "Enums.hpp"
using namespace vc::enums;

//#include "ImGuiHelpers.hpp"
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
//#include <io.h>

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
void mouse_button_callback(GLFWwindow*, int button, int action, int mods);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int TOP_BAR_HEIGHT = 0;
const unsigned int SCR_HEIGHT = 600;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;

// mouse
bool mouseButtonDown[4] = { false, false, false, false };

vc::settings::State state = vc::settings::State(CaptureState::PLAYING, RenderState::MULTI_POINTCLOUD);

int main(int argc, char* argv[]) try {
	
	vc::settings::FolderSettings folderSettings;
	folderSettings.recordingsFolder = "allCameras/";

	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	// glfw window creation
	// --------------------
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Volumetric Capture", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetKeyCallback(window, key_callback);
	
	// tell GLFW to capture our mouse
	//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	//ImGui_ImplGlfw_Init(window, false);
	
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

	glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT - TOP_BAR_HEIGHT);
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
		
		//processInput(window);

		glm::mat4 model = glm::mat4(1.0f);
		glm::mat4 view = camera.GetViewMatrix();
		// pass projection matrix to shader (note that in this case it could change every frame)
		glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);

		vc::rendering::startFrame(window);

		for (int i = 0; i < pipelines.size() && i < 4; ++i)
		{
			int x = i % 2;
			int y = floor(i / 2);
			if(state.renderState == RenderState::ONLY_COLOR || state.renderState == RenderState::ONLY_DEPTH)
			{
				//std::cout << i << ": x=" << x << " - y=" << y << std::endl;
				if (state.renderState == RenderState::ONLY_COLOR && pipelines[i]->data->filteredColorFrames) {
					pipelines[i]->rendering->renderTexture(pipelines[i]->data->filteredColorFrames, x, y, aspect);
				}
				else if (state.renderState == RenderState::ONLY_DEPTH && pipelines[i]->data->colorizedDepthFrames) {
					pipelines[i]->rendering->renderTexture(pipelines[i]->data->colorizedDepthFrames, x, y, aspect);
				}
			}
			else if (state.renderState == RenderState::MULTI_POINTCLOUD && pipelines[i]->data->points && pipelines[i]->data->filteredColorFrames) {
				pipelines[i]->rendering->renderPointcloud(pipelines[i]->data->points, pipelines[i]->data->filteredColorFrames, model, view, projection, width, height, x, y);
			}
		}
		
		
		/*
		vc::imgui_helpers::initialize(width, height);
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
		vc::imgui_helpers::finalize();
		*/

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

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS || action == GLFW_REPEAT) {
		switch (key) {
		case GLFW_KEY_ESCAPE: {
			glfwSetWindowShouldClose(window, true);
			break;
		}
		case GLFW_KEY_1: {
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			break;
		}
		case GLFW_KEY_2: {
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			break;
		}
		case GLFW_KEY_C: {
			state.renderState = RenderState::ONLY_COLOR;
			break;
		}
		case GLFW_KEY_V: {
			state.renderState = RenderState::ONLY_DEPTH;
			break;
		}
		case GLFW_KEY_B: {
			state.renderState = RenderState::MULTI_POINTCLOUD;
			break;
		}
		case GLFW_KEY_W: {
			camera.ProcessKeyboard(FORWARD, deltaTime);
			break;
		}
		case GLFW_KEY_S: {
			camera.ProcessKeyboard(BACKWARD, deltaTime);
			break;
		}
		case GLFW_KEY_A: {
			camera.ProcessKeyboard(LEFT, deltaTime);
			break;
		}
		case GLFW_KEY_D: {
			camera.ProcessKeyboard(RIGHT, deltaTime);
			break;
		}
		}
	}
}


// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glfwMakeContextCurrent(window);
	glViewport(0, 0, width, height - 50);
}


// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (mouseButtonDown[GLFW_MOUSE_BUTTON_1]) {
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

		camera.ProcessMouseMovement(-xoffset, -yoffset);
	}
}

void mouse_button_callback(GLFWwindow*, int button, int action, int mods)
{
	if (action == GLFW_PRESS) {
		firstMouse = true;
	}
	mouseButtonDown[button] = action;
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	camera.ProcessMouseScroll(yoffset);
}

#pragma endregion
