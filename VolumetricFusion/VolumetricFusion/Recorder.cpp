#include "Recorder.h"
#include <iostream>
#include <stdint.h>

Recorder::Recorder(std::vector<CaptureDevice*> devices,  std::string baseDir)
{
	this->baseDir = baseDir;
	this->devices = devices;


	std::thread t(&Recorder::persist, this);
	t.detach();
}

Recorder::~Recorder()
{
}

void Recorder::addToQueue()
{
	if (!isRecording) {
		return;
	}

	for (auto device : devices) {
		try {
			rs2::points points = device->getPoints();
			rs2::video_frame* colorFrame = device->getColorFrame();
			std::string serialNr = device->getSerialNr();
			auto frameNr = device->getFrameNumber();
			std::string filename = baseDir + "\\" + serialNr + "\\" + std::to_string(frameNr) + ".ply";

			queue.push(PointsQueueInstance(&points, filename, colorFrame));
		}
		catch (const rs2::error & e)
		{
			std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
		}
		catch (const std::exception & e)
		{
			std::cerr << e.what() << std::endl;
		}
	}
}

void Recorder::test() {
	std::cout << "Test" << std::endl;
}

void Recorder::persist() {
	while (true) {
		try {
			if (queue.size() > 0) {
				PointsQueueInstance obj = queue.front();
				rs2::points* points = obj.points;
				rs2::video_frame* colorFrame = obj.texture;
				std::string filename = obj.filename;
				points->export_to_ply(filename, *colorFrame);
				queue.pop();
			}
		}
		catch (const rs2::error & e)
		{
			std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
		}
		catch (const std::exception & e)
		{
			std::cerr << e.what() << std::endl;
		}
	}
}


void Recorder::clearPersistentPointsFolder()
{
	std::experimental::filesystem::remove_all(this->baseDir);
	std::experimental::filesystem::create_directory(baseDir);

	for (auto device : devices) {
		std::experimental::filesystem::create_directory(baseDir + "\\" + device->getSerialNr()
);
	}
}

void Recorder::start()
{
	isRecording = true;
}

void Recorder::stop()
{
	isRecording = false;
}

