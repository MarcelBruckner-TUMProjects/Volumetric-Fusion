#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <set>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <matrix_lib/matrix_structures/MatrixStructuresInclude.h>
#include <common_utils/data_structures/Grid2Interface.h>
#include <common_utils/timing/TimerCPU.h>
#include <common_utils/Common.h>

#include "heatmap_fusion/linmath.h"
#include "heatmap_fusion/GLShader.h"
#include "heatmap_fusion/RuntimeSettings.h"
#include "heatmap_fusion/IOManagement.h"
#include "heatmap_fusion/MeshProcessingCPU.h"
#include "heatmap_fusion/VirtualCamera.h"
#include "heatmap_fusion/ViewerInput.h"
#include "heatmap_fusion/BoundingBox.h"
#include "heatmap_fusion/VisualizationHelper.h"
#include "heatmap_fusion/Texture.h"
#include "heatmap_fusion/ImageProcessingCUDA.h"

using namespace heatmap_fusion;
using namespace matrix_lib;
using namespace std;

enum class VisualizationMode {
	SHAPE,
	POINTCLOUD,
	DEPTH_IMAGE,
	DEPTH_NORMALS
};


void testVisualization() {
	// Global settings.
	//std::string meshPath = "../data/mesh.ply";
	std::string depthIntrinsicsPath = "../data/depthIntrinsics_4.mat";
	std::string depthExtrinsicsPath = "../data/depthExtrinsics_4.mat";
	std::string colorIntrinsicsPath = "../data/colorIntrinsics_4.mat";
	std::string colorExtrinsicsPath = "../data/colorExtrinsics_4.mat";
	std::string colorImagePath = "../data/color000000_4.png";
	std::string depthImagePath = "../data/depth000000_4.png";

	std::string depthIntrinsicsPath2 = "../data/depthIntrinsics_5.mat";
	std::string depthExtrinsicsPath2 = "../data/depthExtrinsics_5.mat";
	std::string colorIntrinsicsPath2 = "../data/colorIntrinsics_5.mat";
	std::string colorExtrinsicsPath2 = "../data/colorExtrinsics_5.mat";
	std::string colorImagePath2 = "../data/color000000_5.png";
	std::string depthImagePath2 = "../data/depth000000_5.png";

	float depthMin = 0.2f;
	float depthMax = 8.0f;
	int screenWidth = 1280;
	int screenHeight = 960;

	// Explicitly set device
	cudaGLSetGLDevice(0);

	// Initialize window.
	GLFWwindow* window;

	glfwSetErrorCallback(ViewerInput::errorCallback);
	if (!glfwInit())
		exit(EXIT_FAILURE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

	window = glfwCreateWindow(screenWidth, screenHeight, "Visualization", NULL, NULL);
	if (!window) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	// Set event callbacks.
	glfwSetKeyCallback(window, ViewerInput::keyCallback);
	glfwSetMouseButtonCallback(window, ViewerInput::mouseButtonCallback);
	glfwSetScrollCallback(window, ViewerInput::scrollCallback);
	glfwSetCursorPosCallback(window, ViewerInput::cursorPosCallback);

	glfwMakeContextCurrent(window);
	gladLoadGL();
	glfwSwapInterval(1);

	// Load depth map.
	Array2<float> depthImageRaw = io_management::loadDepthImage(depthImagePath);
	Array2<float> depthImageRaw2 = io_management::loadDepthImage(depthImagePath2);

	int depthWidth = depthImageRaw.getDimX();
	int depthHeight = depthImageRaw.getDimY();
	int depthWidth2 = depthImageRaw2.getDimX();
	int depthHeight2 = depthImageRaw2.getDimY();

	Array2<uchar4> colorImageRaw = io_management::loadColorImage(colorImagePath);
	Array2<uchar4> colorImageRaw2 = io_management::loadColorImage(colorImagePath2);

	int colorWidth = colorImageRaw.getDimX();
	int colorHeight = colorImageRaw.getDimY();
	int colorWidth2 = colorImageRaw2.getDimX();
	int colorHeight2 = colorImageRaw2.getDimY();

	// Load template mesh.
	//TriMesh triMesh = io_management::loadMeshPly(meshPath);
	//mesh_proc::computeMeshNormals(triMesh.positions, triMesh.faceIndices, triMesh.normals);

	//VertexMesh vertexMesh = mesh_proc::convertMesh(triMesh);

	// Load intrinsics.
	Mat4f depthIntrinsicsProjection;
	Mat4f depthIntrinsicsProjection2;
	depthIntrinsicsProjection.loadMatrixFromFile(depthIntrinsicsPath);
	depthIntrinsicsProjection2.loadMatrixFromFile(depthIntrinsicsPath2);

	Mat4f colorIntrinsicsProjection;
	Mat4f colorIntrinsicsProjection2;
	colorIntrinsicsProjection.loadMatrixFromFile(colorIntrinsicsPath);
	colorIntrinsicsProjection2.loadMatrixFromFile(colorIntrinsicsPath2);

	VirtualCameraf camera{ Mat4f::identity(), depthIntrinsicsProjection, depthWidth, depthHeight, depthMin, depthMax };

	Mat3f depthIntrinsics = depthIntrinsicsProjection.getMatrix3x3();
	Mat3f depthIntrinsics2 = depthIntrinsicsProjection2.getMatrix3x3();
	
	Mat4f depthExtrinsics;
	Mat4f depthExtrinsics2;
	depthExtrinsics.loadMatrixFromFile(depthExtrinsicsPath);
	depthExtrinsics2.loadMatrixFromFile(depthExtrinsicsPath2);

	Mat4f colorExtrinsics;
	Mat4f colorExtrinsics2;
	colorExtrinsics.loadMatrixFromFile(colorExtrinsicsPath);
	colorExtrinsics2.loadMatrixFromFile(colorExtrinsicsPath2);

	// Initialize mesh visualization.
	//vertexMesh.positions.updateDeviceIfNeeded();
	//vertexMesh.normals.updateDeviceIfNeeded();

	//vertexMesh.positions.updateHostIfNeeded();
	//vertexMesh.normals.updateHostIfNeeded();

	//int nVertices = vertexMesh.positions.getSize();

	// Preprocess depth map.
	Texture2D_RGBA8UC colorImage;
	Texture2D_RGBA8UC colorImage2;
	colorImage.create(colorImageRaw);
	colorImage2.create(colorImageRaw2);

	Texture2D_32F depthImage;
	Texture2D_32F depthImage2;
	depthImage.create(depthImageRaw);
	depthImage2.create(depthImageRaw2);

	Texture2D_RGBA32F observationImage;
	Texture2D_RGBA32F observationImage2;
	observationImage.create(depthWidth, depthHeight);
	observationImage2.create(depthWidth2, depthHeight2);

	image_proc_gpu::unprojectDepthImage(depthImage, observationImage, depthWidth, depthHeight, depthIntrinsics, depthExtrinsics);
	image_proc_gpu::unprojectDepthImage(depthImage2, observationImage2, depthWidth2, depthHeight2, depthIntrinsics2, depthExtrinsics2);
	
	Texture2D_RGBA32F normalImage;
	Texture2D_RGBA32F normalImage2;
	normalImage.create(depthWidth, depthHeight);
	normalImage2.create(depthWidth2, depthHeight2);

	int normalKernelRadius = 2;
	float normalPrunningDistance = 0.05;
	image_proc_gpu::computeNormalImage(observationImage, normalImage, depthWidth, depthHeight, normalKernelRadius, normalPrunningDistance);
	image_proc_gpu::computeNormalImage(observationImage2, normalImage2, depthWidth2, depthHeight2, normalKernelRadius, normalPrunningDistance);


	// Compile shaders.
	ShaderPhong shaderPhong;
	shaderPhong.compile(Settings::get().s_shaderPath + "/phong.vert", Settings::get().s_shaderPath + "/phong.frag");

	ShaderPhong shaderPhongInstanced;
	shaderPhongInstanced.compile(Settings::get().s_shaderPath + "/phong_instanced.vert", Settings::get().s_shaderPath + "/phong_instanced.frag");

	ShaderTexture shaderDepthTexture;
	shaderDepthTexture.compile(Settings::get().s_shaderPath + "/depth_image.vert", Settings::get().s_shaderPath + "/depth_image.frag");

	ShaderTexture shaderNormalTexture;
	shaderNormalTexture.compile(Settings::get().s_shaderPath + "/normal_image.vert", Settings::get().s_shaderPath + "/normal_image.frag");

	ShaderPoints shaderPoints;
	shaderPoints.compile(Settings::get().s_shaderPath + "/points.vert", Settings::get().s_shaderPath + "/points.frag");

	ShaderPoints shaderPoints2;
	shaderPoints2.compile(Settings::get().s_shaderPath + "/points.vert", Settings::get().s_shaderPath + "/points.frag");

	// Create buffers.
	//MeshBuffers meshBuffers;
	//meshBuffers.create(vertexMesh.positions, vertexMesh.normals);

	TextureBuffers textureBuffers;
	textureBuffers.create();

	MemoryContainer<float4> validObservations;
	MemoryContainer<float4> validObservations2;
	image_proc_gpu::filterInvalidPoints(observationImage, depthWidth, depthHeight, validObservations);
	image_proc_gpu::filterInvalidPoints(observationImage2, depthWidth2, depthHeight2, validObservations2);

	PointcloudBuffers pointcloudBuffers;
	PointcloudBuffers pointcloudBuffers2;
	pointcloudBuffers.create(validObservations);
	pointcloudBuffers2.create(validObservations2);

	int nPoints = validObservations.getSize();
	int nPoints2 = validObservations2.getSize();
	validObservations.updateHostIfNeeded();
	validObservations2.updateHostIfNeeded();

	GLuint pointsBuffer;
	glGenBuffers(1, &pointsBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, pointsBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float4)* nPoints, validObservations.h(), GL_STATIC_DRAW);

	GLuint pointsBuffer2;
	glGenBuffers(1, &pointsBuffer2);
	glBindBuffer(GL_ARRAY_BUFFER, pointsBuffer2);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float4)* nPoints2, validObservations2.h(), GL_STATIC_DRAW);

	cout << "nValidObservations = " << validObservations.getSize() << endl;
	cout << "nValidObservations2 = " << validObservations2.getSize() << endl;

	// Set keyboard control parameters.
	float distanceStep{ 0.5f };
	float thetaStep = 0.5f;
	VirtualCameraf fixedCamera = camera;
	bool bRenderWireframe{ false };
	VisualizationMode visualizationMode{ VisualizationMode::POINTCLOUD };

	BoundingBox3f sceneBoundingBox(Vec3f(0.0, 0.0, 0.5), Vec3f(0.0, 0.0, 0.5));
	//for (int i = 0; i < nVertices; i++) {
	//	sceneBoundingBox.include(Vec3f(vertexMesh.positions.h(i)));
	//}

	// Start loop.
	
	while (!glfwWindowShouldClose(window)) {
	
		// Refresh viewport.
		float ratio;
		int width, height;
		mat4x4 m, p, mvp;
		glfwGetFramebufferSize(window, &width, &height);
		ratio = width / (float)height;

		glViewport(0, 0, width, height);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (bRenderWireframe) {
			glPolygonMode(GL_FRONT, GL_LINE);
			glPolygonMode(GL_BACK, GL_LINE);
		}

		Vec3f lightDirection = Vec3f(0.0, 1.0, -1.0);//camera.getLook();
		Vec3f eye = camera.getEye();
		Mat4f viewMatrix = camera.getView();
		Mat4f viewProjMatrix = camera.getViewProj();

		// Execute visualization.
		if (visualizationMode == VisualizationMode::SHAPE) {
			//glEnable(GL_DEPTH_TEST);

			//// Render the mesh.
			//shaderPhong.use(viewMatrix, viewProjMatrix, eye, lightDirection);
			//meshBuffers.bind();
			//glDrawArrays(GL_TRIANGLES, 0, nVertices);
		}
		else if (visualizationMode == VisualizationMode::POINTCLOUD) {
			{

				// Render points.
				Mat4f cameraPose = Mat4f::identity();

				Mat4f cameraPose1(0.84991388, -0.26177695, 0.45729556, 2.18669759,
					0.27087871, 0.96146821, 0.0469426, -4.09601347,
					-0.45196363, 0.08397447, 0.88807498, 18.49981854,
					0, 0, 0, 1);

				shaderPoints.use(viewMatrix * cameraPose, viewProjMatrix * cameraPose, colorIntrinsicsProjection, colorExtrinsics, colorWidth, colorHeight);

				// Points
				glBindBuffer(GL_ARRAY_BUFFER, pointsBuffer);
				glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
				glEnableVertexAttribArray(0);
				glVertexAttribDivisor(2, 1);

				// Texture.
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, colorImage.getTexture());
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				
				// Draw call.
				glDrawArrays(GL_POINTS, 0, nPoints);
				
				//Mat4f cameraPose2(-0.948254, -0.317345, -0.0102926, 0.252493
				//	, 0.312217, -0.937847, 0.151534, 0.138665
				//	, -0.0577416, 0.14048, 0.988398, -0.0946556
				//	, 0, 0, 0, 1);

				//Mat4f cameraPose2(-0.931374, 0.362238, -0.036413, 19.196568,
				//	-0.141143, -0.267075, 0.953284, 44.103508,
				//	0.335591, 0.893003, 0.299874, 35.351875,
				//	0, 0, 0, 1);

				Mat4f cameraPose2(0.89017749, -0.43662759, 0.13015521, 4.05189313,
					0.42247246, 0.89798885, 0.12301644, 2.59794436,
					-0.17059031, -0.05451947, 0.98383259, -15.57232251,
					0.0, 0.0, 0.0, 1.0);

				shaderPoints2.use(viewMatrix * cameraPose, viewProjMatrix * cameraPose, colorIntrinsicsProjection2, colorExtrinsics2, colorWidth2, colorHeight2);

				// Points
				glBindBuffer(GL_ARRAY_BUFFER, pointsBuffer2);
				glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
				glEnableVertexAttribArray(0);
				glVertexAttribDivisor(2, 1);

				// Texture.
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, colorImage2.getTexture());
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

				// Draw call.
				glDrawArrays(GL_POINTS, 0, nPoints2);
			}
		}
		else if (visualizationMode == VisualizationMode::DEPTH_IMAGE) {
			// Render depth texture.
			shaderDepthTexture.use();
			textureBuffers.bind(depthImage.getTexture());

			glDrawArrays(GL_TRIANGLE_FAN, 0, textureBuffers.getNumVertices());
		}
		else if (visualizationMode == VisualizationMode::DEPTH_NORMALS) {
			// Render normal texture.
			shaderNormalTexture.use();
			textureBuffers.bind(normalImage.getTexture());

			glDrawArrays(GL_TRIANGLE_FAN, 0, textureBuffers.getNumVertices());
		}

		// Swap buffer and check events.
			
		glfwSwapBuffers(window);
		glfwPollEvents();

		// Turn off wire-frame (if enabled).
		glPolygonMode(GL_FRONT, GL_FILL);
		glPolygonMode(GL_BACK, GL_FILL);

		// Turn off depth check (if enabled).
		glDisable(GL_DEPTH_TEST);

		// Check events.
		if (ViewerInput::get().isMouseMoved()) {
			// Rotate view around the object.
			Vec2i posCenter(std::round(width / 2.f), std::round(height / 2.f));
			Vec2f posDelta = ViewerInput::get().getMouse().pos - ViewerInput::get().getMousePrev().pos;

			float theta = thetaStep;
			if (ViewerInput::get().isKeyPressed(GLFW_KEY_RIGHT_SHIFT) || ViewerInput::get().isKeyPressed(GLFW_KEY_LEFT_SHIFT)) theta *= 0.1f;

			if (ViewerInput::get().getMouse().leftPressed) {
				camera.lookRight(theta * posDelta.x());
				camera.lookUp(theta * posDelta.y());

				Vec3f lookDir = camera.getLook();
				Vec3f sceneCenter = sceneBoundingBox.getCenter();
				Vec3f eye = sceneCenter - (sceneCenter - camera.getEye()).length() * lookDir;
				camera.translate(eye - camera.getEye());
			}
		}

		if (ViewerInput::get().isScrollChanged()) {
			float distance = distanceStep * 0.5f;
			if (ViewerInput::get().isKeyPressed(GLFW_KEY_RIGHT_SHIFT) || ViewerInput::get().isKeyPressed(GLFW_KEY_LEFT_SHIFT)) distance *= 0.05f;

			camera.move(distance * ViewerInput::get().getScrollChangeY());
		}

		if (ViewerInput::get().isKeyDown(GLFW_KEY_V)) {
			// Reset to fixed camera view.
			camera = fixedCamera;
		}

		if (ViewerInput::get().isKeyDown(GLFW_KEY_T)) {
			// Switch wireframe mode.
			bRenderWireframe = !bRenderWireframe;
		}

		// Switch visualization mode.
		if (ViewerInput::get().isKeyDown(GLFW_KEY_1)) visualizationMode = VisualizationMode::SHAPE;
		else if (ViewerInput::get().isKeyDown(GLFW_KEY_2)) visualizationMode = VisualizationMode::POINTCLOUD;
		else if (ViewerInput::get().isKeyDown(GLFW_KEY_3)) visualizationMode = VisualizationMode::DEPTH_IMAGE;
		else if (ViewerInput::get().isKeyDown(GLFW_KEY_4)) visualizationMode = VisualizationMode::DEPTH_NORMALS;

		// Reset change events.
		ViewerInput::get().resetChangeEvents();
	}

	// Cleanup.
	glfwDestroyWindow(window);
	glfwTerminate();
}


int main() {
	testVisualization();

	return 0;
}