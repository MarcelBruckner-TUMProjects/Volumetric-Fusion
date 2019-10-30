#pragma once
#include <librealsense2\hpp\rs_processing.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace processing_blocks {

	namespace {
		/**
		Creates a processing block for our OpenCV needs.

		@param lambda a function that takes a reference to a cv::Mat as parameter:
				basic lambda syntax: processing_blocks::createProcessingBlock([](cv::Mat &image){...})
		*/
		template<typename F>
		rs2::processing_block createProcessingBlock(F& lambda, int imageDescriptor, int factor) {
			return rs2::processing_block(
				[=](rs2::frame f, rs2::frame_source& src)
			{
				// For each input frame f, do:

				const int w = f.as<rs2::video_frame>().get_width();
				const int h = f.as<rs2::video_frame>().get_height();

				// frame --> cv
				cv::Mat image(cv::Size(w, h), imageDescriptor, (void*)f.get_data(), cv::Mat::AUTO_STEP);
				// do some (silly) processing

				// Here the magic happens
				lambda(image);

				// Allocate new frame. Copy all missing data from f.
				// This assumes the output is same resolution and format
				// if not true, need to specify parameters explicitly
				auto res = src.allocate_video_frame(f.get_profile(), f);

				// copy from cv --> frame
				memcpy((void*)res.get_data(), image.data, w * h * factor);

				// Send the resulting frame to the output queue
				src.frame_ready(res);
			});
		}
	}

	template<typename F>
	rs2::processing_block createColorProcessingBlock(F& lambda) {
		// Don't bother the magic numbers, they describe the image channels
		return createProcessingBlock(lambda, CV_8UC3, 3);
	}

	template<typename F>
	rs2::processing_block createDepthProcessingBlock(F& lambda) {
		// Don't bother the magic numbers, they describe the image channels
		return createProcessingBlock(lambda, CV_8UC1, 2);
	}

	rs2::processing_block createEmpty()
	{
		return rs2::processing_block(
			[&](rs2::frame f, rs2::frame_source& src)
		{
			src.frame_ready(f);
		});
	}
}