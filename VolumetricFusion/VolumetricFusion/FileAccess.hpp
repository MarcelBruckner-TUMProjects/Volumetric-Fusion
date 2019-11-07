#pragma once
#include <string>
#include <sys/stat.h>

#include <filesystem>
namespace fs = std::filesystem;

namespace vc::file_access {
	bool exists(const std::string& name) {
		struct stat buffer;
		return (stat(name.c_str(), &buffer) == 0);
	}

	bool isDirectory(std::string path, bool createIfNot=false) {
		bool isDir = exists(path) && std::filesystem::is_directory(path);

		if (!isDir && createIfNot) {
			isDir = fs::create_directories(path);
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
		auto callback = [&folder, &filenames](const auto& entry) {
            auto path = entry.path();
            filenames.push_back(folder + path.filename().string());
        };
		iterateFilesInFolder(folder, callback, createIfNot);

		std::vector<std::string> filtered;
		std::copy_if(filenames.begin(), filenames.end(), std::back_inserter(filtered), [&filterExtension](std::string filename) {return hasEnding(filename, filterExtension); });

		if (sorted) {
			std::sort(filtered.begin(), filtered.end());
		}
		return filtered;
	}
	   
	void resetDirectory(std::string path, bool createIfNot = false) {
		if (!isDirectory(path, createIfNot)) {
			return;
		}
	    auto callback = [&](const auto& entry) {
	        fs::remove(entry.path());
	    };
		iterateFilesInFolder(path, callback, true);
	}

}
