#pragma once

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

class CaptureDevice
{
private:
	const char* serialNr;
	rs2::pipeline pipeline;
	rs2::pointcloud pointcloud;
	rs2::points points;
	rs2::context ctx;
	rs2::config cfg;
	rs2::frameset frames;

public:
	CaptureDevice(rs2::context ctx, rs2::device device);
	rs2::pipeline_profile start();
	int stop();
	
	rs2::pointcloud getPointcloud();
	rs2::points getPoints();

	void acquireFrame();

	rs2::frameset getFrames();
	rs2::video_frame getColorFrame();
	rs2::depth_frame getDepthFrame();
};

