#pragma once

#ifndef _CAPTURE_DEVICE_
#define _CAPTURE_DEVICE_

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <map>
#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include "data.hpp"
#include "processing.hpp"
#include "Rendering.hpp"

namespace vc::capture {
	/// <summary>
	/// Base class for capturing devices
	/// </summary>
	class CaptureDevice {
	public:
		rs2::config cfg;
		std::shared_ptr < vc::rendering::Rendering> rendering;
		std::shared_ptr < vc::processing::ChArUco> chArUco;
		std::shared_ptr < vc::data::Data> data;
		std::shared_ptr < vc::data::Camera> rgb_camera;
		std::shared_ptr < vc::data::Camera> depth_camera;

		std::shared_ptr<rs2::pipeline > pipeline;

		std::shared_ptr < std::atomic_bool> isPipelineRunning;
		std::shared_ptr < std::atomic_bool> isThreadRunning;
		std::shared_ptr < std::atomic_bool> paused;
		std::shared_ptr < std::atomic_bool> calibrateCameras;

		std::shared_ptr < std::thread> thread;

		void startPipeline() {
			this->pipeline->start(this->cfg);
			isPipelineRunning->store(true);
			setCameras();
			isThreadRunning->store(true);
			this->thread = std::make_shared<std::thread>(&vc::capture::CaptureDevice::captureThreadFunction, this);
			resumeThread();

			//this->data->setIntrinsics(this->pipeline->get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics());
		}

		void stopPipeline() { 
			stopThread();
			if (isPipelineRunning->load()) {
				this->pipeline->stop();
			}
			isPipelineRunning->store(false);
		}

		void pauseThread() {
			this->paused->store(true);
		}

		void resumeThread() {
			this->paused->store(false);
		}

		void stopThread() {
			if (isThreadRunning->load()) {
				this->isThreadRunning->store(false);
				if (thread) {
					thread->join();
				}
			}
		}

		void calibrate(bool calibrate) {
			this->calibrateCameras->store(calibrate);
		}

		void renderColor(const float pos_x, const float pos_y, const float aspect, const int viewport_width, const int viewport_height) {
			if (data->filteredColorFrames) {
				this->rendering->renderTexture(data->filteredColorFrames, pos_x, pos_y, aspect, viewport_width, viewport_height);
			}
		}

		void renderDepth(const float pos_x, const float pos_y, const float aspect, const int viewport_width, const int viewport_height) {
			if (data->colorizedDepthFrames) {
				this->rendering->renderTexture(data->colorizedDepthFrames, pos_x, pos_y, aspect, viewport_width, viewport_height);
			}
		}
		
		void renderAllPointclouds(glm::mat4 model, glm::mat4 view, glm::mat4 projection,
			const int viewport_width, const int viewport_height, glm::mat4 relativeTransformation, bool renderCoordinateSystem = false) {
			renderPointcloud(model, view, projection, viewport_width, viewport_height, -1, -1, relativeTransformation, renderCoordinateSystem);
		}

		void renderPointcloud(glm::mat4 model, glm::mat4 view, glm::mat4 projection,
			const int viewport_width, const int viewport_height, const int pos_x, const int pos_y, glm::mat4 relativeTransformation = glm::mat4(1.0f), bool renderCoordinateSystem = false) {
			if (data->filteredDepthFrames && data->filteredColorFrames) {
				rendering->renderPointcloud(data->filteredDepthFrames, data->filteredColorFrames, depth_camera, rgb_camera, model, view, projection,
					viewport_width, viewport_height, pos_x, pos_y, relativeTransformation, renderCoordinateSystem);
			}
		}

		void setResolutions(const std::vector<int> colorStream, const std::vector<int> depthStream, bool directResume = true) {
			pauseThread();
			stopPipeline();
			if (colorStream.size() == 2) {
				this->cfg.enable_stream(RS2_STREAM_COLOR, colorStream[0], colorStream[1], RS2_FORMAT_RGB8, 30);
			}
			if (depthStream.size() == 2) {
				this->cfg.enable_stream(RS2_STREAM_DEPTH, depthStream[0], depthStream[1], RS2_FORMAT_Z16, 30);
			}
			if (directResume) {
				startPipeline();
			}
		}

		CaptureDevice(CaptureDevice& other) {
			this->rendering = other.rendering;
			this->data = other.data;
			this->chArUco = other.chArUco;
			this->pipeline = other.pipeline;
			this->cfg = other.cfg;
			this->rgb_camera = other.rgb_camera;
			this->depth_camera = other.depth_camera;

			this->isPipelineRunning = other.isPipelineRunning;
			this->isThreadRunning = other.isThreadRunning;
			this->paused = other.paused;
			this->calibrateCameras = other.calibrateCameras;
			this->thread = other.thread;
		}

		CaptureDevice(rs2::context context) {
			this->rendering = std::make_shared<vc::rendering::Rendering>();
			this->data = std::make_shared<vc::data::Data>();
			this->chArUco = std::make_shared<vc::processing::ChArUco>();
			this->pipeline = std::make_shared<rs2::pipeline>(context);
			this->isPipelineRunning = std::make_shared<std::atomic_bool>(false);
			this->isThreadRunning = std::make_shared<std::atomic_bool>(false);
			this->paused = std::make_shared<std::atomic_bool>(true);
			this->calibrateCameras = std::make_shared<std::atomic_bool>(false);
			//setCameras();

		}

		void setCameras() {
			this->rgb_camera = std::make_shared<vc::data::Camera>(this->pipeline->get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics());
			this->depth_camera = std::make_shared<vc::data::Camera>(this->pipeline->get_active_profile().get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics(), 
				this->pipeline->get_active_profile().get_device().first<rs2::depth_sensor>().get_depth_scale());
		}

		void captureThreadFunction() {
			rs2::align alignToColor(RS2_STREAM_COLOR);
			chArUco->startCharucoProcessing(*rgb_camera);

			while (isThreadRunning->load() && isPipelineRunning->load()) //While application is running
			{
				while (paused->load()) {
					continue;
				}

				try {
					rs2::frameset frameset = pipeline->wait_for_frames(); // Wait for next set of frames from the camera

					frameset = alignToColor.process(frameset);

					rs2::frame depthFrame = frameset.get_depth_frame(); //Take the depth frame from the frameset
					if (!depthFrame) { // Should not happen but if the pipeline is configured differently
						return;       //  it might not provide depth and we don't want to crash
					}

					rs2::frame filteredDepthFrame = depthFrame; // Does not copy the frame, only adds a reference

					rs2::frame colorFrame = frameset.get_color_frame();

					if (calibrateCameras->load()) {
						// Send color frame for processing
						chArUco->charucoProcessingBlocks->invoke(colorFrame);
						// Wait for results
						colorFrame = chArUco->charucoProcessingQueues.wait_for_frame();
					}

					data->frameId = frameset.get_color_frame().get_frame_number();
					data->filteredColorFrames = colorFrame;

					// Apply filters.
					/*for (auto&& filter : data->filters) {
						filteredDepthFrame = filter->process(filteredDepthFrame);
					}*/

					// Push filtered & original data to their respective queues
					data->filteredDepthFrames = filteredDepthFrame;

					data->colorizedDepthFrames = data->colorizer.process(depthFrame);		// Colorize the depth frame with a color map

					//data->points = data->pointclouds.calculate(depthFrame);  // Generate pointcloud from the depth data
					//data->pointclouds.map_to(data->colorizedDepthFrames);      // Map the colored depth to the point cloud
				}
				catch (const std::exception & e) {
					std::stringstream stream;
					stream << "******************** THREAD ERROR *******************" << std::endl << e.what() << "****************************************************" << std::endl;
				}
			}
			stopPipeline();
		}
	};

	/// <summary>
	/// A capture device for streaming the live RGB-D data from the device.
	/// </summary>
	/// <seealso cref="CaptureDevice" />
	class StreamingCaptureDevice : public CaptureDevice {
	public:
		StreamingCaptureDevice(rs2::context context, rs2::device device, const std::vector<int> colorStream, const std::vector<int> depthStream) :
			CaptureDevice(context)
		{
			data->deviceName = device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);

			this->cfg.enable_device(data->deviceName);
			setResolutions(colorStream, depthStream, false);
		}
	};

	/// <summary>
	///  A capture device for streaming the live RGB-D data from the device and to record it to a file.
	/// </summary>
	/// <seealso cref="StreamingCaptureDevice" />
	class RecordingCaptureDevice : public StreamingCaptureDevice {
	public:
		RecordingCaptureDevice(rs2::context context, rs2::device device, const std::vector<int> colorStream, const std::vector<int> depthStream, std::string foldername) :
			StreamingCaptureDevice(context, device, colorStream, depthStream)
		{
			/*this->cfg.enable_device(data->deviceName);
			setResolutions(colorStream, depthStream);*/
			this->cfg.enable_record_to_file(foldername + data->deviceName + ".bag");
		}
	};

	/// <summary>
	///  A capture device for streaming the RGB-D data from a file.
	/// </summary>
	/// <seealso cref="CaptureDevice" />
	class PlayingCaptureDevice : public CaptureDevice {
	public:
		PlayingCaptureDevice(rs2::context context, std::string filename) :
			CaptureDevice(context)
		{
			data->deviceName = filename;

			this->cfg.enable_device_from_file(data->deviceName);
			this->cfg.enable_all_streams();
		}
	};
}

#endif