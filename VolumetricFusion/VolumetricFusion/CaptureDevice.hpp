#pragma once

#ifndef _CAPTURE_DEVICE_
#define _CAPTURE_DEVICE_

#include <map>
#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include "Data.hpp"
#include "PinholeCamera.hpp"
#include "Processing.hpp"
#include "Rendering.hpp"
#include "Utils.hpp"

#include "ceres/problem.h"
#include "ceres/solver.h"

namespace vc::capture {
	/// <summary>
	/// Base class for capturing devices
	/// </summary>
	class CaptureDevice {
	public:
		rs2::config cfg;
		rs2::pipeline_profile profile;
		std::shared_ptr < vc::rendering::Rendering> rendering;
		std::shared_ptr < vc::processing::ChArUco> chArUco;
		std::shared_ptr < vc::data::Data> data;
		std::shared_ptr < vc::camera::PinholeCamera> rgb_camera;
		std::shared_ptr < vc::camera::PinholeCamera> depth_camera;

		std::shared_ptr<rs2::pipeline > pipeline;
		std::shared_ptr < std::atomic_bool> paused;
		std::shared_ptr < std::atomic_bool> stopped;
		std::shared_ptr < std::atomic_bool> calibrateCameras;

		std::shared_ptr < std::thread> thread;
		
		bool startPipeline() {
			try {
				this->profile = this->pipeline->start(this->cfg);
				setCameras();
				resumeThread();

				//this->data->setIntrinsics(this->pipeline->get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics());
				return true;
			}
			catch (rs2::error & e) {
				std::cerr << e.what() << std::endl;
				return false;
			}
		}

		bool terminate() { 
			stopThread();
			try{
				this->pipeline->stop();
				return true;
			}
			catch (rs2::error & e) {
				return false;
			}
		}

		void pauseThread() {
			this->paused->store(true);
		}

		void resumeThread() {
			this->paused->store(false);
		}

		void stopThread() {
			stopped->store(true);
			if (thread) {
				thread->join();
			}
		}

		void calibrate(bool calibrate) {
			this->calibrateCameras->store(calibrate);
		}

		void renderColor(int pos_x,  int pos_y, const float aspect, const int viewport_width, const int viewport_height) {
			if (data->filteredColorFrames) {
				this->rendering->renderTexture(data->filteredColorFrames, pos_x, pos_y, aspect, viewport_width, viewport_height);
			}
		}

		void renderDepth(int pos_x, int pos_y, const float aspect, const int viewport_width, const int viewport_height) {
			if (data->colorizedDepthFrames) {
				this->rendering->renderTexture(data->colorizedDepthFrames, pos_x, pos_y, aspect, viewport_width, viewport_height);
			}
		}
		
		void renderPointcloud(glm::mat4 model, glm::mat4 view, glm::mat4 projection,
			Eigen::Matrix4d relativeTransformation, float alpha) {
			if (data->filteredDepthFrames && data->filteredColorFrames) {
				rendering->renderPointcloud(data->filteredDepthFrames, data->filteredColorFrames, depth_camera, rgb_camera, model, view, projection,
					relativeTransformation, alpha);
			}
		}

		bool setResolutions(const std::vector<int> colorStream, const std::vector<int> depthStream, bool directResume = true) {
			pauseThread();
			try{
				this->pipeline->stop();
			}
			catch (rs2::error & e) {
				std::cerr << "Pipeline already stopped" << std::endl << e.what() << std::endl;
			}
			if (colorStream.size() == 2) {
				this->cfg.enable_stream(RS2_STREAM_COLOR, colorStream[0], colorStream[1], RS2_FORMAT_RGB8, 30);
			}
			if (depthStream.size() == 2) {
				this->cfg.enable_stream(RS2_STREAM_DEPTH, depthStream[0], depthStream[1], RS2_FORMAT_Z16, 30);
			}
			if (directResume) {
				return startPipeline();
			}
			return true;
		}

		CaptureDevice(CaptureDevice& other) {
			this->rendering = other.rendering;
			this->data = other.data;
			this->chArUco = other.chArUco;
			this->pipeline = other.pipeline;
			this->cfg = other.cfg;
			this->rgb_camera = other.rgb_camera;
			this->depth_camera = other.depth_camera;

			this->paused = other.paused;
			this->calibrateCameras = other.calibrateCameras;
			this->thread = other.thread;
		}

		CaptureDevice(rs2::context context) : 
			rendering(std::make_shared<vc::rendering::Rendering>()),
			data(std::make_shared<vc::data::Data>()),
			chArUco(std::make_shared<vc::processing::ChArUco>()),
			pipeline(std::make_shared<rs2::pipeline>(context)),
			paused(std::make_shared<std::atomic_bool>(true)),
			stopped(std::make_shared<std::atomic_bool>(false)),
			calibrateCameras(std::make_shared<std::atomic_bool>(false)),
			depth_camera(std::make_shared<vc::camera::MockPinholeCamera>()),
			rgb_camera(std::make_shared<vc::camera::MockPinholeCamera>()),
			thread(std::make_shared<std::thread>(&vc::capture::CaptureDevice::captureThreadFunction, this))
		{
			//setCameras();
		}

		void setCameras() {
			bool tmpCalibrate = calibrateCameras->load();
			calibrateCameras->store(false);
			this->rgb_camera = std::make_shared<vc::camera::PinholeCamera>(this->pipeline->get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics());
			this->depth_camera = std::make_shared<vc::camera::PinholeCamera>(this->pipeline->get_active_profile().get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics(),
			this->pipeline->get_active_profile().get_device().first<rs2::depth_sensor>().get_depth_scale());
			chArUco->startCharucoProcessing(*rgb_camera);
			calibrateCameras->store(tmpCalibrate);
		}

		void captureThreadFunction() {

			while (!stopped->load()) //While application is running
			{
				while (paused->load()) {
					continue;
				}

				try {
					rs2::frameset frameset;
					if (!pipeline->poll_for_frames(&frameset)) {
						continue;
					}

					rs2::align alignToColor(RS2_STREAM_COLOR);
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

					rs2::colorizer colorizer;
					data->colorizedDepthFrames = colorizer.process(depthFrame);		// Colorize the depth frame with a color map

					//data->points = data->pointclouds.calculate(depthFrame);  // Generate pointcloud from the depth data
					//data->pointclouds.map_to(data->colorizedDepthFrames);      // Map the colored depth to the point cloud
				}
				catch (const rs2::error & e) {
					std::cerr << vc::utils::asHeader("RS2 - Thread error") << e.what() << std::endl;
				}
				catch (const std::exception & e) {
					std::cerr << vc::utils::asHeader("Thread error") << e.what() << std::endl;
				}
			}
			//terminate();
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