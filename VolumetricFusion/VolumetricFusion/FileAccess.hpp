#pragma once
#include <string>
#include <sys/stat.h>

#include <filesystem>
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
}
