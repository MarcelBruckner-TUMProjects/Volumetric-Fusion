#pragma once
#include <string>
#include <sys/stat.h>

#include <filesystem>
#include <direct.h>
namespace fs = std::filesystem;

namespace file_access {
	bool exists(const std::string& name) {
		struct stat buffer;
		return (stat(name.c_str(), &buffer) == 0);
	}

	bool isDirectory(std::string path, bool createIfNot=false) {
		bool isDir = exists(path) && std::filesystem::is_directory(path);

		if (!isDir && createIfNot) {
			_mkdir(path.c_str());
			isDir = true;
		}

		return isDir;
	}

	template<typename F>
	void iterateFilesInFolder(std::string path, F& lambda, bool createIfNot = false) {
		if (!isDirectory(path, createIfNot)) {
			return;
		}
		for (const auto& entry : fs::directory_iterator(path)) {
			lambda(entry);
		}
	}

	bool hasEnding(std::string const& fullString, std::string const& ending) {
		if (fullString.length() >= ending.length()) {
			return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
		}
		else {
			return false;
		}
	}

	std::vector<std::string> listFilesInFolder(std::string folder, std::string filterExtension = std::string(".bag"), bool createIfNot = false, bool sorted=true) {
		std::vector<std::string> filenames;		
		iterateFilesInFolder(folder, [&folder, &filenames](const auto& entry) {
			auto path = entry.path();
			filenames.push_back(folder + path.filename().string());
		}, createIfNot);

		std::vector<std::string> filtered;
		std::copy_if(filenames.begin(), filenames.end(), std::back_inserter(filtered), [&filterExtension](std::string filename) {return hasEnding(filename, filterExtension); });

		if (sorted) {
			std::sort(filtered.begin(), filtered.end());
		}
		return filtered;
	}
	   
	void resetFolder(std::string path) {
		iterateFilesInFolder(path, [&](const auto& entry) {fs::remove(entry.path()); }, true);
	}

}
