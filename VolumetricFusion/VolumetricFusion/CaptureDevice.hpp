#pragma once
#include <example.hpp>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <map>
#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include "ProcessingBlocks.hpp"

#include <opencv2\core\matx.hpp>

namespace vc {
	/// <summary>
	/// A capture device holding references to all neccessary pipelines and data.
	/// </summary>
	class CaptureDevice {
	public:
		/// <summary>
		/// The RealSense pipeline
		/// </summary>
		std::shared_ptr<rs2::pipeline> pipeline;

		/// <summary>
		/// The device name displayed in the views
		/// </summary>
		std::string deviceName;

		/// <summary>
		/// The configuration
		/// </summary>
		rs2::config config;

		/// <summary>
		/// The render frames. We'll keep track of the last frame of each stream available to make the presentation persistent	
		/// </summary>
		rs2::frame renderFrames;

		/// <summary>
		/// The texture
		/// </summary>
		texture texture;

		/// <summary>
		/// The colorizer
		/// </summary>
		rs2::colorizer colorizer;

		/// <summary>
		/// The filtered color frame
		/// </summary>
		rs2::frame filteredColorFrame;

		/// <summary>
		/// The filtered depth frames
		/// </summary>
		rs2::frame filteredDepthFrame;

		rs2::pointcloud pointcloud;
		rs2::points points;
		rs2::frame colorizedDepthFrame;

		/// <summary>
		/// The camera matrix
		/// </summary>
		cv::Matx33f cameraMatrix;

		/// <summary>
		/// The distortion coefficients
		/// </summary>
		std::vector<float> distortionCoefficients;

		/// <summary>
		/// The charuco identifier buffer
		/// </summary>
		std::map<unsigned long long, std::vector<int>> charucoIdBuffer;

		/// <summary>
		/// The rotation buffer
		/// </summary>
		std::map<unsigned long long, cv::Vec3d> rotationBuffer;

		/// <summary>
		/// The translation buffer
		/// </summary>
		std::map<unsigned long long, cv::Vec3d> translationBuffer;
		
		std::shared_ptr<std::atomic_bool> isStopped;
		std::shared_ptr<std::atomic_bool> isPaused;
		std::shared_ptr<std::atomic_bool> isCharucoProcessing;

		rs2::pipeline_profile start();
		void stop();

		void setIntrinsics();

		void renderOnlyColor(rect r);
		void renderOnlyDepth(rect r);
		void renderPointcloud(int i, glfw_state view_orientation, bool is_retina_display, int width, int height, int width_half, int height_half);

	protected:
		/// <summary>
		/// The intrinsics
		/// </summary>
		rs2_intrinsics intrinsics;

		/// <summary>
		/// The processing threads
		/// </summary>
		std::thread processingThread = std::thread([&]() {
			auto pipe = pipeline;

			rs2::align align_to_color(RS2_STREAM_COLOR);

			while (!*isStopped) //While application is running
			{
				while (*isPaused) {
					continue;
				}

				try {
					rs2::frameset data = pipe->wait_for_frames(); // Wait for next set of frames from the camera

					data = align_to_color.process(data);

					rs2::frame depth_frame = data.get_depth_frame(); //Take the depth frame from the frameset
					if (!depth_frame) { // Should not happen but if the pipeline is configured differently
						return;       //  it might not provide depth and we don't want to crash
					}

					rs2::frame filtered_depth_frame = depth_frame; // Does not copy the frame, only adds a reference

					rs2::frame color_frame = data.get_color_frame();

					//if (*isCharucoProcessing) {
					//	// Send color frame for processing
					//	charucoProcessingBlock.invoke(color_frame);
					//	// Wait for results
					//	color_frame = charucoProcessingQueue.wait_for_frame();
					//}

					filteredColorFrame = color_frame;

					//// Apply filters.
					//for (auto&& filter : filters) {
					//	filtered_depth_frame = filter->process(filtered_depth_frame);
					//}

					// Push filtered & original data to their respective queues
					filteredDepthFrame = filtered_depth_frame;

					points = pointcloud.calculate(depth_frame);  // Generate pointcloud from the depth data
					colorizedDepthFrame = colorizer.process(depth_frame);		// Colorize the depth frame with a color map
					pointcloud.map_to(colorizedDepthFrame);      // Map the colored depth to the point cloud
				}
				catch (const std::exception & e) {
					std::stringstream stream;
					stream << "******************** THREAD ERROR *******************" << std::endl << e.what() << "****************************************************" << std::endl;
				}
			}
			pipe->stop();
		});;

		/// <summary>
		/// The color processing queue
		/// </summary>
		rs2::frame_queue charucoProcessingQueue;
		
		/// <summary>
		/// The color processing block
		/// </summary>
		//rs2::processing_block charucoProcessingBlock = vc::processing_blocks::createColorProcessingBlock(vc::processing_blocks::charucoPoseEstimation);


		void setBools(std::shared_ptr<std::atomic_bool> isStopped,
			std::shared_ptr<std::atomic_bool> isPaused,
			std::shared_ptr<std::atomic_bool> isColorProcessing);

	};

	/// <summary>
	/// A capture device to stream the live RGB-D data from the cameras.
	/// </summary>
	/// <seealso cref="CaptureDevice" />
	class StreamingCaptureDevice : public CaptureDevice {
	public:
		StreamingCaptureDevice(rs2::context ctx, rs2::device device, 
			std::shared_ptr<std::atomic_bool> isStopped, 
			std::shared_ptr<std::atomic_bool> isPaused,
			std::shared_ptr<std::atomic_bool> isColorProcessing);
	};
	
	/// <summary>
	/// A capture device that additionally records the stream to a file.
	/// </summary>
	/// <seealso cref="StreamingCaptureDevice" />
	class RecordingCaptureDevice : public StreamingCaptureDevice {
	public:
		RecordingCaptureDevice(rs2::context ctx, rs2::device device, std::string recordingsFolder,
			std::shared_ptr<std::atomic_bool> isStopped,
			std::shared_ptr<std::atomic_bool> isPaused,
			std::shared_ptr<std::atomic_bool> isColorProcessing);
	};
	
	/// <summary>
	/// A capture device to stream the RGB-D data from a file.
	/// </summary>
	/// <seealso cref="CaptureDevice" />
	class PlayingCaptureDevice : public CaptureDevice {
	public:
		PlayingCaptureDevice(rs2::context ctx, std::string filename,
			std::shared_ptr<std::atomic_bool> isStopped,
			std::shared_ptr<std::atomic_bool> isPaused,
			std::shared_ptr<std::atomic_bool> isColorProcessing);
	};
}