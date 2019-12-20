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
#include <VolumetricFusion\Voxelgrid.hpp>
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
void processInput(GLFWwindow* window);
void setCalibration(bool calibrate);

// settings
const unsigned int SCR_WIDTH = 800 * 2;
const unsigned int TOP_BAR_HEIGHT = 0;
const unsigned int SCR_HEIGHT = 600 * 2 ;

std::vector<int> DEFAULT_COLOR_STREAM = { 640, 480 };
std::vector<int> DEFAULT_DEPTH_STREAM = { 640, 480 };

std::vector<int> CALIBRATION_COLOR_STREAM = { 1920, 1080 };
std::vector<int> CALIBRATION_DEPTH_STREAM = { 1280, 720 };

// camera
Camera camera(glm::vec3(0.0f, 0.0f, -1.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;
float MouseSensitivity = 0.1;
float Yaw = 0;
float Pitch = 0;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;

// mouse
bool mouseButtonDown[4] = { false, false, false, false };

vc::settings::State state = vc::settings::State(CaptureState::PLAYING, RenderState::CALIBRATED_POINTCLOUD);
std::vector<std::shared_ptr<  vc::capture::CaptureDevice>> pipelines;

bool visualizeCharucoResults = true;

bool renderVoxelgrid = false;
vc::fusion::Voxelgrid* voxelgrid;
std::atomic_bool calibrateCameras = false;
std::atomic_bool fuseFrames = false;
std::atomic_bool renderCoordinateSystem = false;

int main(int argc, char* argv[]) try {
	
	vc::settings::FolderSettings folderSettings;
	folderSettings.recordingsFolder = "recordings/allCameras/";

	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
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

	glad_set_post_callback([](const char* name, void* funcptr, int len_args, ...) {
		GLenum error_code;

		(void)funcptr;
		(void)len_args;

		error_code = glad_glGetError();

		if (error_code != GL_NO_ERROR) {
			// shut this up for a while
			fprintf(stderr, "ERROR %d in %s\n", error_code, name);
		}
	});

	voxelgrid = new vc::fusion::Voxelgrid();

	//ImGui_ImplGlfw_Init(window, false);
	
	//vc::rendering::Rendering rendering(app, viewOrientation);

	rs2::context ctx; // Create librealsense context for managing devices
	//std::vector<vc::data::Data> datas;
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
				pipelines.emplace_back(std::make_shared < vc::capture::RecordingCaptureDevice>(ctx, device, DEFAULT_COLOR_STREAM, DEFAULT_DEPTH_STREAM, folderSettings.recordingsFolder));
			}
			else if (state.captureState == CaptureState::STREAMING) {
				pipelines.emplace_back(std::make_shared < vc::capture::StreamingCaptureDevice>(ctx, device, DEFAULT_COLOR_STREAM, DEFAULT_DEPTH_STREAM));
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
	std::map<int, glm::mat4> relativeTransformations = {
		{0, glm::mat4(1.0f)},
		{1, glm::mat4(1.0f)},
		{2, glm::mat4(1.0f)},
		{3, glm::mat4(1.0f)},
	};

	std::thread calibrationThread;
	std::thread fusionThread;

	// Program state (mostly for recordings)
	struct {
		std::vector<int> highestFrameIds = { -1, -1, -1, -1 };
		bool allMarkersDetected = false;
		bool allPipelinesEnteredLooped = false;
	} programState;

	for (int i = 0; i < pipelines.size(); i++) {
		pipelines[i]->processing->visualize = visualizeCharucoResults;
	}

#pragma region Camera Calibration Thread

	setCalibration(calibrateCameras);
	calibrationThread = std::thread([&stopped, &programState, &relativeTransformations]() {
		while (!stopped) {
			if (!calibrateCameras) {
				continue;
			}

			int markersDetected = 0;
			int pipelinesEnteredLoop = 0;
			for (int i = 0; i < pipelines.size(); i++) {
				{
					if (!pipelines[i]->processing->hasMarkersDetected/* || relativeTransformations.count(i) != 0*/) {
						continue;
					}
					markersDetected++;

					programState.highestFrameIds[i] = MAX(pipelines[i]->processing->frameId, programState.highestFrameIds[i]);
					if (programState.highestFrameIds[i] > pipelines[i]->processing->frameId) {
						pipelinesEnteredLoop++;
					}

					glm::mat4 baseToMarkerTranslation = pipelines[0]->processing->translation;
					glm::mat4 baseToMarkerRotation = pipelines[0]->processing->rotation;

					if (i == 0) {
						//relativeTransformations[i] = glm::inverse(baseToMarkerTranslation);
						relativeTransformations[i] = glm::mat4(1.0f); 
						continue;
					}

					glm::mat4 markerToRelativeTranslation = pipelines[i]->processing->translation;
					glm::mat4 markerToRelativeRotation = pipelines[i]->processing->rotation;

					glm::mat4 relativeTransformation = (
						//glm::mat4(1.0f)

						//baseToMarkerTranslation * (markerToRelativeRotation) * (baseToMarkerRotation) * glm::inverse(markerToRelativeTranslation)
						//baseToMarkerTranslation * glm::inverse(markerToRelativeRotation) * (baseToMarkerRotation) * glm::inverse(markerToRelativeTranslation)
						//baseToMarkerTranslation * (markerToRelativeRotation) * glm::inverse(baseToMarkerRotation) * glm::inverse(markerToRelativeTranslation)
						//baseToMarkerTranslation * glm::inverse(markerToRelativeRotation) * glm::inverse(baseToMarkerRotation) * glm::inverse(markerToRelativeTranslation) 

						//baseToMarkerTranslation * (baseToMarkerRotation) * (markerToRelativeRotation) * glm::inverse(markerToRelativeTranslation)
						//baseToMarkerTranslation * glm::inverse((baseToMarkerRotation) * glm::inverse(markerToRelativeRotation)) * glm::inverse(markerToRelativeTranslation)
						///*baseToMarkerTranslation **/ glm::inverse(baseToMarkerRotation)* (markerToRelativeRotation)*glm::inverse(markerToRelativeTranslation) //######################################################################
						baseToMarkerTranslation * glm::inverse(baseToMarkerRotation) * (markerToRelativeRotation) * glm::inverse(markerToRelativeTranslation) //######################################################################
						//baseToMarkerTranslation * glm::inverse(baseToMarkerRotation) * glm::inverse(markerToRelativeRotation) * glm::inverse(markerToRelativeTranslation)

						//glm::inverse(markerToRelativeTranslation * markerToRelativeRotation) * baseToMarkerTranslation * baseToMarkerRotation
						//glm::inverse(markerToRelativeTranslation * markerToRelativeRotation) * baseToMarkerRotation * baseToMarkerTranslation
						//glm::inverse(markerToRelativeRotation * markerToRelativeTranslation) * baseToMarkerTranslation * baseToMarkerRotation
						//glm::inverse(markerToRelativeRotation * markerToRelativeTranslation) * baseToMarkerRotation * baseToMarkerTranslation

						//baseToMarkerTranslation * baseToMarkerRotation * glm::inverse(markerToRelativeTranslation * markerToRelativeRotation)
						//baseToMarkerRotation * baseToMarkerTranslation * glm::inverse(markerToRelativeTranslation * markerToRelativeRotation)
						//baseToMarkerTranslation * baseToMarkerRotation * glm::inverse(markerToRelativeRotation * markerToRelativeTranslation)
						//baseToMarkerRotation * baseToMarkerTranslation * glm::inverse(markerToRelativeRotation * markerToRelativeTranslation)

						//markerToRelativeTranslation * markerToRelativeRotation* glm::inverse(baseToMarkerTranslation * baseToMarkerRotation) 
						//markerToRelativeTranslation * markerToRelativeRotation* glm::inverse(baseToMarkerRotation * baseToMarkerTranslation) 
						//markerToRelativeRotation * markerToRelativeTranslation* glm::inverse(baseToMarkerTranslation * baseToMarkerRotation) 
						//markerToRelativeRotation * markerToRelativeTranslation* glm::inverse(baseToMarkerRotation * baseToMarkerTranslation) 

						//glm::inverse(baseToMarkerTranslation * baseToMarkerRotation) * markerToRelativeTranslation * markerToRelativeRotation
						//glm::inverse(baseToMarkerRotation * baseToMarkerTranslation) * markerToRelativeTranslation * markerToRelativeRotation
						//glm::inverse(baseToMarkerTranslation * baseToMarkerRotation) * markerToRelativeRotation * markerToRelativeTranslation
						//glm::inverse(baseToMarkerRotation * baseToMarkerTranslation) * markerToRelativeRotation * markerToRelativeTranslation
					);

					relativeTransformations[i] = relativeTransformation;
				}
			}
			programState.allMarkersDetected = markersDetected == pipelines.size();
			programState.allPipelinesEnteredLooped = pipelinesEnteredLoop == pipelines.size();

			if (programState.allMarkersDetected) {
				setCalibration(false);
				// start fusion thread logic
				fuseFrames.store(true);
			}
		}
	});
#pragma endregion



#pragma region Fusion Thread
	fusionThread = std::thread([&stopped, &programState, &relativeTransformations]() {
		const int maxIntegrations = 10;
		int integrations = 0;
		while (!stopped) {
			if (calibrateCameras || !fuseFrames) {
				continue;
			}

			for (int i = 0; i < pipelines.size(); i++) {
				// Only integrate frames with a valid transformation
				if (relativeTransformations.count(i) <= 0 || !pipelines[i]->data->points) {
					continue;
				}

				const rs2::vertex* vertices = pipelines[i]->data->points.get_vertices();

				auto pip = pipelines[i]->processing;
				voxelgrid->integrateFrameCPU(pipelines[i], relativeTransformations[i], i, pip->frameId);
			}

			integrations++;
			if (integrations >= maxIntegrations) {
				std::cout << "Fused " << (integrations * pipelines.size()) << " frames" << std::endl;
				fuseFrames.store(false);
				break;
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
		
		// -------------------------------------------------------------------------------
		processInput(window);

		glm::mat4 model = glm::mat4(1.0f);
		glm::mat4 view = camera.GetViewMatrix();
		glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
		//glm::mat4 projection = glm::ortho(0.0f, (float)SCR_WIDTH, 0.0f, (float)SCR_HEIGHT, 0.1f, 100.0f);
		vc::rendering::startFrame(window);

		for (int i = 0; i < pipelines.size() && i < 4; ++i)
		{
			int x = i % 2;
			int y = floor(i / 2);
				if (state.renderState == RenderState::ONLY_COLOR) {
					pipelines[i]->renderColor(x, y, aspect, width, height);
				}
				else if (state.renderState == RenderState::ONLY_DEPTH) {
					pipelines[i]->renderDepth(x, y, aspect, width, height);
				}
			else if (state.renderState == RenderState::MULTI_POINTCLOUD) {
					pipelines[i]->renderPointcloud(model, view, projection, width, height, x, y, relativeTransformations[i], renderCoordinateSystem);
					if (renderVoxelgrid) {
						voxelgrid->renderGrid(model, view, projection);
					}
				}
			else if (state.renderState == RenderState::CALIBRATED_POINTCLOUD) {
					pipelines[i]->renderAllPointclouds(model, view, projection, width, height, relativeTransformations[i], i, renderCoordinateSystem);
				}
		}
		if (renderVoxelgrid && state.renderState == RenderState::CALIBRATED_POINTCLOUD) {
			voxelgrid->renderGrid(model, view, projection);
		}
		if (state.renderState == RenderState::VOXELGRID) {
			voxelgrid->renderField(model, view, projection);
			
		}
		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
#pragma endregion

#pragma region Final cleanup

	stopped.store(true);
	for (int i = 0; i < pipelines.size(); i++) {
		pipelines[i]->stopPipeline();
	}
	calibrationThread.join();
	fusionThread.join();
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

void setCalibration(bool calibrate) {
	calibrateCameras.store(calibrate);
	for (int i = 0; i < pipelines.size(); i++) {
		if (calibrateCameras) {
			pipelines[i]->setResolutions(CALIBRATION_COLOR_STREAM, CALIBRATION_DEPTH_STREAM);
		}
		else {
			pipelines[i]->setResolutions(DEFAULT_COLOR_STREAM, DEFAULT_DEPTH_STREAM);
		}
		pipelines[i]->calibrate(calibrateCameras);
	}
}

bool isKeyPressed(GLFWwindow* window, int key) {
	return glfwGetKey(window, key) == GLFW_PRESS;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
	if (isKeyPressed(window, GLFW_KEY_W)) {
		camera.ProcessKeyboard(FORWARD, deltaTime);
	}
	if (isKeyPressed(window, GLFW_KEY_S)) {
		camera.ProcessKeyboard(BACKWARD, deltaTime);
	}
	if (isKeyPressed(window, GLFW_KEY_A)) {
		camera.ProcessKeyboard(LEFT, deltaTime);
	}
	if (isKeyPressed(window, GLFW_KEY_D)) {
		camera.ProcessKeyboard(RIGHT, deltaTime);
	}
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS || action == GLFW_REPEAT) {
		switch (key) {
		case GLFW_KEY_ESCAPE: {
			glfwSetWindowShouldClose(window, true);
			break;
		}
		case GLFW_KEY_8: {
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			break;
		}
		case GLFW_KEY_9: {
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			break;
		}
		case GLFW_KEY_1: {
			state.renderState = RenderState::ONLY_COLOR;
			break;
		}
		case GLFW_KEY_2: {
			state.renderState = RenderState::ONLY_DEPTH;
			break;
		}
		case GLFW_KEY_3: {
			state.renderState = RenderState::MULTI_POINTCLOUD;
			break;
		}
		case GLFW_KEY_4: {
			state.renderState = RenderState::CALIBRATED_POINTCLOUD;
			break;
		}
		case GLFW_KEY_5: {
			//state.renderState = RenderState::VOXELGRID;
			break;
		}
		case GLFW_KEY_V: {
			visualizeCharucoResults = !visualizeCharucoResults;
			for (auto pipe : pipelines) {
				pipe->processing->visualize = visualizeCharucoResults;
			}
			break;
		}
		case GLFW_KEY_G: {
			renderVoxelgrid = !renderVoxelgrid;
			break;
		}
		case GLFW_KEY_C: {
			setCalibration(!calibrateCameras);
			break;
		}
		case GLFW_KEY_L: {
			renderCoordinateSystem = !renderCoordinateSystem;
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

		float xoffset = lastX - xpos;
		float yoffset = ypos - lastY; // reversed since y-coordinates go from bottom to top
		//float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

		lastX = xpos;
		lastY = ypos;

		//processMouse(xoffset, yoffset);
		camera.ProcessMouseMovement(xoffset, yoffset);
	}
}
//
//void processMouse(float xoffset, float yoffset, GLboolean constrainPitch ) {
//	xoffset *= MouseSensitivity;
//	yoffset *= MouseSensitivity;
//
//	Yaw += xoffset;
//	Pitch += yoffset;
//
//	// Make sure that when pitch is out of bounds, screen doesn't get flipped
//	if (constrainPitch)
//	{
//		if (Pitch > 89.0f)
//			Pitch = 89.0f;
//		if (Pitch < -89.0f)
//			Pitch = -89.0f;
//	}
//	model = glm::mat4(1.0f);
//	model = glm::rotate(model, glm::radians(Pitch), glm::vec3(1.0f, 0.0f, 0.0f));
//	model = glm::rotate(model, glm::radians(Yaw), glm::vec3(0.0f, 1.0f, 0.0f));
//}

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
