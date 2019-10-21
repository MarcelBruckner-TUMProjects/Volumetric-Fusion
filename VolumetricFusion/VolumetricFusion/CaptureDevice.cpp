#include "CaptureDevice.h"

CaptureDevice::CaptureDevice(rs2::context ctx, rs2::device device)
{ 
	serialNr = device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
    pipeline = rs2::pipeline(ctx);
	cfg.enable_device(device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
}

rs2::pipeline_profile CaptureDevice::start()
{
	return pipeline.start(cfg);
}

int CaptureDevice::stop()
{
	pipeline.stop();
	return 0;
}

rs2::pointcloud CaptureDevice::getPointcloud()
{
	return pointcloud;
}

rs2::points CaptureDevice::getPoints()
{
	return points;
}

void CaptureDevice::acquireFrame()
{
	frames = pipeline.wait_for_frames();
	rs2::video_frame colorFrame = getColorFrame();
	pointcloud.map_to(colorFrame);
	rs2::depth_frame depthFrame = getDepthFrame();
	points = pointcloud.calculate(depthFrame);
}

rs2::frameset CaptureDevice::getFrames()
{
	return frames;
}

rs2::video_frame CaptureDevice::getColorFrame()
{
	return frames.get_color_frame();
}

rs2::depth_frame CaptureDevice::getDepthFrame()
{
	return  frames.get_depth_frame();
}




