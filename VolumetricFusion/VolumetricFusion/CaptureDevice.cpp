#include "CaptureDevice.h"
#include <stdio.h>
#include <iostream>

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
	if (pipeline.poll_for_frames(&frames))
	{
		rs2::video_frame color_ = frames.get_color_frame();
		pointcloud.map_to(color_);
		rs2::depth_frame depth_ = frames.get_depth_frame();
		points = pointcloud.calculate(depth_);

		/*color = &color_;
		depth = &depth_;*/
	}

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

unsigned long long CaptureDevice::getFrameNumber()
{
	try {
		return frames.get_frame_number();
	}
	catch (const rs2::error & e)
	{
		std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
		return -2;
	}
	catch (const std::exception & e) {
		return -1;
	}
}

std::string CaptureDevice::getSerialNr()
{
	return serialNr;
}
