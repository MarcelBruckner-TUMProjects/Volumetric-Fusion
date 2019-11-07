#pragma once
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#include <string>
#include <vector>
#include "CaptureDevice.hpp"
#include <filesystem>
#include <cstdio>
#include <experimental/filesystem>
#include <queue>
#include <thread>


class PointsQueueInstance {
public:
	rs2::points* points;
	std::string filename;
	rs2::video_frame* texture;

	PointsQueueInstance(rs2::points* points, std::string filename, rs2::video_frame* texture) {
		this->points = points;
		this->filename = filename;
		this->texture = texture;
	}
};

class Recorder
{
private:	
	std::string baseDir;
	std::vector<CaptureDevice*> devices;
	std::queue<PointsQueueInstance> queue;
	std::thread persistThread;
	bool isRecording = false;
	void test();
	void persist();

public:
	Recorder(std::vector<CaptureDevice*> devices, std::string baseDir = "C:\\Users\\Marcel Bruckner\\Documents\\Volumetric-Fusion\\points");
	~Recorder();
	void addToQueue();
	void clearPersistentPointsFolder();
	void start();
	void stop();
};

