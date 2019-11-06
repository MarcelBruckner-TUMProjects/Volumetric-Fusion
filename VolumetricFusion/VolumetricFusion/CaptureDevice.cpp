#pragma once
#include "CaptureDevice.hpp"

/// <summary>
/// Starts this instance.
/// </summary>
/// <returns></returns>
rs2::pipeline_profile vc::CaptureDevice::start() {
	return pipeline->start(config);
}

void vc::CaptureDevice::stop()
{
	processingThread.join();
}

/// <summary>
/// Sets the intrinsics.
/// </summary>
/// <param name="intrinsics">The intrinsics.</param>
void vc::CaptureDevice::setIntrinsics()
{
	this->intrinsics = pipeline->get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();;
	cameraMatrix = cv::Matx33f(
		intrinsics.fx, 0, intrinsics.ppx,
		0, intrinsics.fy, intrinsics.ppy,
		0, 0, 1
	);

	for (float c : intrinsics.coeffs) {
		distortionCoefficients.push_back(c);
	}
}

void vc::CaptureDevice::renderOnlyColor(rect r)
{
	texture.render(filteredColorFrame, r);
}

void vc::CaptureDevice::renderOnlyDepth(rect r)
{
	texture.render(colorizer.process(filteredDepthFrame), r);
}

void vc::CaptureDevice::renderPointcloud(int i, glfw_state view_orientation, bool is_retina_display, int width, int height, int width_half, int height_half)
{
	if (colorizedDepthFrame && points) {
		view_orientation.tex.upload(colorizedDepthFrame);   //  and upload the texture to the view (without this the view will be B&W)
		if (is_retina_display) {
			glViewport(width * (i % 2), height - (height * (i / 2)), width, height);
		}
		else {
			glViewport(width_half * (i % 2), height_half - (height_half * (i / 2)), width_half, height_half);
		}

		if (filteredColorFrame) {
			draw_pointcloud_and_colors(width_half, height_half, view_orientation, points, filteredColorFrame, 0.2f);
		}
		else {
			draw_pointcloud(width_half, height_half, view_orientation, points);
		}

		if (distortionCoefficients.size()) {
			draw_rectangle(width_half, height_half, 0, 0, 0, view_orientation);
		}
	}
}

/// <summary>
/// Sets the bools.
/// </summary>
/// <param name="isStopped">The is stopped.</param>
/// <param name="isPaused">The is paused.</param>
/// <param name="isColorProcessing">The is color processing.</param>
void vc::CaptureDevice::setBools(std::shared_ptr<std::atomic_bool> isStopped, std::shared_ptr<std::atomic_bool> isPaused, std::shared_ptr<std::atomic_bool> isColorProcessing)
{
	this->isStopped = isStopped;
	this->isPaused = isPaused;
	this->isCharucoProcessing = isColorProcessing;
}

/// <summary>
/// Initializes a new instance of the <see cref="StreamingCaptureDevice"/> class.
/// </summary>
/// <param name="context">The context.</param>
/// <param name="device">The device.</param>
vc::StreamingCaptureDevice::StreamingCaptureDevice(rs2::context context, rs2::device device,
	std::shared_ptr<std::atomic_bool> isStopped,
	std::shared_ptr<std::atomic_bool> isPaused,
	std::shared_ptr<std::atomic_bool> isColorProcessing) {

	setBools(isStopped, isPaused, isColorProcessing);
	this->pipeline = std::make_shared<rs2::pipeline>(context);
	this->deviceName = device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);

	config.enable_device(deviceName);
	config.enable_all_streams();

	//charucoProcessingBlock.start(charucoProcessingQueue);
}

/// <summary>
/// Initializes a new instance of the <see cref="RecordingCaptureDevice"/> class.
/// </summary>
/// <param name="context">The context.</param>
/// <param name="device">The device.</param>
/// <param name="recordingsFolder">The recordings folder.</param>
vc::RecordingCaptureDevice::RecordingCaptureDevice(rs2::context context, rs2::device device, std::string recordingsFolder,
	std::shared_ptr<std::atomic_bool> isStopped,
	std::shared_ptr<std::atomic_bool> isPaused,
	std::shared_ptr<std::atomic_bool> isColorProcessing) : vc::StreamingCaptureDevice(context, device, isStopped, isPaused, isColorProcessing)
{
	config.enable_record_to_file(recordingsFolder + deviceName + ".bag");
}

/// <summary>
/// Initializes a new instance of the <see cref="PlayingCaptureDevice"/> class.
/// </summary>
/// <param name="context">The context.</param>
/// <param name="filename">The filename.</param>
vc::PlayingCaptureDevice::PlayingCaptureDevice(rs2::context context, std::string filename,
	std::shared_ptr<std::atomic_bool> isStopped,
	std::shared_ptr<std::atomic_bool> isPaused,
	std::shared_ptr<std::atomic_bool> isColorProcessing)
{
	setBools(isStopped, isPaused, isColorProcessing);

	pipeline = std::make_shared<rs2::pipeline>(context);
	std::string deviceName = filename;

	config.enable_device_from_file(filename);
	config.enable_all_streams();
}