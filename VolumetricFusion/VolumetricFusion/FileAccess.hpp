#pragma once
#include <string>
#include <sys/stat.h>

namespace file_access {
	bool exists_test(const std::string& name) {
		struct stat buffer;
		return (stat(name.c_str(), &buffer) == 0);
	}
}
