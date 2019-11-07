#pragma once

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <map>
#include <iostream>
#include <string>
#include <thread>
#include <atomic>

namespace vc::capture {
	class CaptureDevice {
	public:		
		std::shared_ptr < vc::data::Data >data;
		std::shared_ptr<rs2::pipeline > pipeline;
		rs2::config cfg;

		std::shared_ptr < std::atomic_bool> stopped;
		std::shared_ptr < std::atomic_bool> paused;
		std::shared_ptr < std::atomic_bool> calibrateCameras;
		
		void startPipeline() {
			this->pipeline->start(this->cfg);
		}

		CaptureDevice(CaptureDevice& other) {
			this->data = other.data;
			this->pipeline = other.pipeline;
			this->cfg = other.cfg;

			this->stopped = other.stopped;
			this->paused = other.paused;
			this->calibrateCameras = other.calibrateCameras;
		}
				
		CaptureDevice(rs2::context context) {
			this->data = std::make_shared<vc::data::Data>();
			this->pipeline = std::make_shared<rs2::pipeline>(context);
			this->stopped = std::make_shared<std::atomic_bool>(false);
			this->paused = std::make_shared<std::atomic_bool>(true);
			this->calibrateCameras = std::make_shared<std::atomic_bool>(false);
		}

		void captureThreadFunction() {
			{
				rs2::align alignToColor(RS2_STREAM_COLOR);

				while (!stopped->load()) //While application is running
				{
					if (paused->load()) {
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
							data->processing.charucoProcessingBlocks->invoke(colorFrame);
							// Wait for results
							colorFrame = data->processing.charucoProcessingQueues.wait_for_frame();
						}

						data->filteredColorFrames = colorFrame;

						// Apply filters.
						/*for (auto&& filter : data->filters) {
							filteredDepthFrame = filter->process(filteredDepthFrame);
						}*/

						// Push filtered & original data to their respective queues
						data->filteredDepthFrames = filteredDepthFrame;

						data->points = data->pointclouds.calculate(depthFrame);  // Generate pointcloud from the depth data
						data->colorizedDepthFrames = data->colorizer.process(depthFrame);		// Colorize the depth frame with a color map
						data->pointclouds.map_to(data->colorizedDepthFrames);      // Map the colored depth to the point cloud
					}
					catch (const std::exception & e) {
						std::stringstream stream;
						stream << "******************** THREAD ERROR *******************" << std::endl << e.what() << "****************************************************" << std::endl;
					}
				}
				pipeline->stop();
			}
		}
	};

	class StreamingCaptureDevice : public CaptureDevice {
	public:
		StreamingCaptureDevice(rs2::context context, rs2::device device) : 
		CaptureDevice(context)
		{
			data->deviceName = device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);

			this->cfg.enable_device(data->deviceName);
			this->cfg.enable_all_streams();
		}
	};

	class RecordingCaptureDevice : public StreamingCaptureDevice {
	public:
		RecordingCaptureDevice(rs2::context context, rs2::device device, std::string foldername) :
			StreamingCaptureDevice(context, device)
		{
			this->cfg.enable_device(data->deviceName);
			this->cfg.enable_all_streams();
			this->cfg.enable_record_to_file(foldername + data->deviceName + ".bag");
		}
	};

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