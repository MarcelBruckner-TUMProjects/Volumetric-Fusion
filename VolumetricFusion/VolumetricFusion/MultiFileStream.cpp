#pragma region Includes
// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS	

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
#include <chrono>
#include <thread>
#include <atomic>
#include <filesystem>
#include <thread>

using namespace std::chrono_literals;

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
#include <glm/gtx/string_cast.hpp>

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
const unsigned int SCR_WIDTH = 800;
const unsigned int TOP_BAR_HEIGHT = 0;
const unsigned int SCR_HEIGHT = 600;

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

vc::settings::State state = vc::settings::State(CaptureState::PLAYING, RenderState::VOXELGRID);
std::vector<std::shared_ptr<  vc::capture::CaptureDevice>> pipelines;

bool visualizeCharucoResults = true;

bool renderVoxelgrid = false;
vc::fusion::Voxelgrid* voxelgrid;
std::atomic_bool calibrateCameras = true;
std::atomic_bool fuseFrames = false;
std::atomic_bool renderCoordinateSystem = false;

// disable compiler warning C4996
#include <pcl/visualization/cloud_viewer.h>
//#include <pcl/io/io.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>

int main(int argc, char* argv[]) try {
	
	vc::settings::FolderSettings folderSettings;
	//folderSettings.recordingsFolder = "recordings/static_scene_front/";
	folderSettings.recordingsFolder = "allCameras/";

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
		const int maxIntegrations = 5;
		int integrations = 0;
		while (!stopped) {
			if (calibrateCameras || !fuseFrames) {
				continue;
			}

			for (int i = 0; i < pipelines.size(); i++) {
				// Only integrate frames with a valid transformation
				if (relativeTransformations.count(i) <= 0) {
					continue;
				}

				auto p = pipelines[i];
				voxelgrid->integrateFrameCPU(p, relativeTransformations[i], i, p->processing->frameId);
			}

			integrations++;
			if (integrations >= maxIntegrations) {
				std::cout << "Fused " << (integrations * pipelines.size()) << " frames" << std::endl;
				fuseFrames.store(false);
				break;
			}

			{
				//using namespace std::literals::chrono_literals;
				//std::this_thread::sleep_for(2s);
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

		//projection = glm::mat4(
		//	glm::vec4(1.81f, 0.0f, 0.0f, 0.0f),
		//	glm::vec4(0.0f, 2.41f, 0.0f, 0.0f),
		//	glm::vec4(0.0f, 0.0f, -1.0f, -1.0f),
		//	glm::vec4(0.0f, 0.0f, -0.2, 0.0f)
		//);
		//view = glm::mat4(
		//	glm::vec4(-0.76f, 0.02f, -0.65, 0.0f),
		//	glm::vec4(0.0f, 1.0f, 0.04f, 0.0f),
		//	glm::vec4(0.65f, 0.03f, -0.76, 0.0f),
		//	glm::vec4(0.37f, 0.29f, -8.45, 1.0f)
		//);

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

		if (state.renderState == RenderState::PCL) {
			//const float r = 233, g = 233, b = 0;
			std::uint32_t rgb_color_green(0x00FF00);
			std::uint32_t rgb_color_red(0xFF0000);
			std::uint32_t rgb_color_blue(0x0000FF);

			// pipeline 0
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr0(new pcl::PointCloud<pcl::PointXYZRGB>);
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr0_cpy(new pcl::PointCloud<pcl::PointXYZRGB>);
			auto depthFrame0 = pipelines[0]->data->filteredDepthFrames;
			auto points0 = pipelines[0]->data->pointclouds.calculate(depthFrame0);
			const float* vertices_0 = reinterpret_cast<const float*>(points0.get_vertices());
			for (int i = 0; i < points0.size(); i += 3) {
				pcl::PointXYZRGB point;
				point.x = vertices_0[i + 0];
				point.y = vertices_0[i + 1];
				point.z = vertices_0[i + 2];
				point.rgb = *reinterpret_cast<float*>(&rgb_color_green);
				point_cloud_ptr0->points.push_back(point);
				pcl::PointXYZRGB point_cpy;
				point_cpy.x = vertices_0[i + 0];
				point_cpy.y = vertices_0[i + 1];
				point_cpy.z = vertices_0[i + 2];
				point_cpy.rgb = *reinterpret_cast<float*>(&rgb_color_blue);
				point_cloud_ptr0_cpy->points.push_back(point_cpy);
			}
			Eigen::Matrix4f transform0 = Eigen::Matrix4f::Identity();
			glm::mat4 rt0 = relativeTransformations[1];
			transform0(0, 0) = rt0[0][0]; transform0(0, 1) = rt0[0][1]; transform0(0, 2) = rt0[0][2]; transform0(0, 3) = rt0[0][3];
			transform0(1, 0) = rt0[1][0]; transform0(1, 1) = rt0[1][1]; transform0(1, 2) = rt0[1][2]; transform0(1, 3) = rt0[1][3];
			transform0(2, 0) = rt0[2][0]; transform0(2, 1) = rt0[2][1]; transform0(2, 2) = rt0[2][2]; transform0(2, 3) = rt0[2][3];
			transform0(3, 0) = rt0[3][0]; transform0(3, 1) = rt0[3][1]; transform0(3, 2) = rt0[3][2]; transform0(3, 3) = rt0[3][3];
			std::cout << "Transform1: " << transform0 << ", glm1: " << glm::to_string(rt0) << std::endl;
			//pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_transformed_ptr0(new pcl::PointCloud<pcl::PointXYZRGB>());
			//pcl::transformPointCloud(*point_cloud_ptr0, *point_cloud_ptr0, transform0);

			// pipeline 1
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr1(new pcl::PointCloud<pcl::PointXYZRGB>);
			auto depthFrame1 = pipelines[1]->data->filteredDepthFrames;
			auto points1 = pipelines[1]->data->pointclouds.calculate(depthFrame1);
			const float* vertices_1 = reinterpret_cast<const float*>(points1.get_vertices());
			for (int i = 0; i < points1.size(); i += 3) {
				pcl::PointXYZRGB point;
				point.x = vertices_1[i + 0];
				point.y = vertices_1[i + 1];
				point.z = vertices_1[i + 2];
				point.rgb = *reinterpret_cast<float*>(&rgb_color_red);
				point_cloud_ptr1->points.push_back(point);
			}
			Eigen::Matrix4f transform1 = Eigen::Matrix4f::Identity();
			glm::mat4 rt1 = relativeTransformations[1];
			transform1(0, 0) = rt1[0][0]; transform1(0, 1) = rt1[0][1]; transform1(0, 2) = rt1[0][2]; transform1(0, 3) = rt1[0][3];
			transform1(1, 0) = rt1[1][0]; transform1(1, 1) = rt1[1][1]; transform1(1, 2) = rt1[1][2]; transform1(1, 3) = rt1[1][3];
			transform1(2, 0) = rt1[2][0]; transform1(2, 1) = rt1[2][1]; transform1(2, 2) = rt1[2][2]; transform1(2, 3) = rt1[2][3];
			transform1(3, 0) = rt1[3][0]; transform1(3, 1) = rt1[3][1]; transform1(3, 2) = rt1[3][2]; transform1(3, 3) = rt1[3][3];
			std::cout << "Transform1: " << transform1 << ", glm1: " << glm::to_string(rt1) << std::endl;
			//pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_transformed_ptr1(new pcl::PointCloud<pcl::PointXYZRGB>());
			//pcl::transformPointCloud(*point_cloud_ptr1, *point_cloud_ptr1, transform1);

			// do icp
			pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
			icp.setInputSource(point_cloud_ptr0);
			icp.setInputTarget(point_cloud_ptr1);
	
			pcl::PointCloud<pcl::PointXYZRGB> point_cloud_final;
			std::cout << "Run icp ..." << std::endl;
			auto start_time = std::clock();
			icp.align(point_cloud_final);
			auto end_time = std::clock();
			auto icpConverged = icp.hasConverged();
			auto icpFitnessScore = icp.getFitnessScore();
			double elapsed_secs = double(end_time - start_time) / CLOCKS_PER_SEC;
			std::cout << "has converged: " << icp.hasConverged() << ", score: " << icpFitnessScore << ", time in s: " << elapsed_secs << std::endl;
			std::cout << icp.getFinalTransformation() << std::endl;

			pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
			viewer->setBackgroundColor(0.5f, 0.5f, 0.5f);
			
			auto point_cloud_final_ptr = point_cloud_final.makeShared();
			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_blue(point_cloud_ptr0_cpy);
			viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud_ptr0_cpy, rgb_blue, "pc 0");
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "pc 0");

			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_red(point_cloud_ptr1);
			viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud_ptr1, rgb_red, "pc 1");
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "pc 1");

			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_green(point_cloud_final_ptr);
			viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud_final_ptr, rgb_green, "final pc");
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "final pc");

			viewer->addCoordinateSystem(0.1);
			viewer->initCameraParameters();

			while (!viewer->wasStopped())
			{
				viewer->spinOnce(100);
				std::this_thread::sleep_for(100ms);
			}
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
	fuseFrames.store(!calibrate);
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
			state.renderState = RenderState::VOXELGRID;
			break;
		}
		case GLFW_KEY_7: {
			state.renderState = RenderState::PCL;
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
		case GLFW_KEY_M: {
			glm::mat4 view = camera.GetViewMatrix();
			glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
			std::cout << "glm::mat4 view = " << glm::to_string(view) << std::endl;
			std::cout << "glm::mat4 projection = " << glm::to_string(projection) << std::endl;
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

		//float xoffset = lastX - xpos;
		//float yoffset = ypos - lastY; // reversed since y-coordinates go from bottom to top
		float xoffset = xpos - lastX;
		float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

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
