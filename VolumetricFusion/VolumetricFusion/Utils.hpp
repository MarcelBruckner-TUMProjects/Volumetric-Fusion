#pragma once

#ifndef _UTILS_HEADER
#define _UTILS_HEADER

#include <vector>
#include <string>
#include <sstream>

#include <iostream>
#include <chrono>
#include <thread>

namespace vc::utils {
	void sleepFor(unsigned long milliseconds) {
		using namespace std::chrono_literals;
		std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
	}
	
	void sleepFor(std::string message, unsigned long milliseconds) {
		using namespace std::chrono_literals;
		std::cout << message << std::endl;
		std::cout << "Sleeping" << std::endl;
		std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
		std::cout << "Awaken" << std::endl;
	}

	std::string toString(glm::vec4 vec) {
		return std::to_string(vec.x) + ", " + std::to_string(vec.y) + ", " + std::to_string(vec.z) + ", " + std::to_string(vec.w) + "\n";
	}

	std::string toString(glm::vec4 baseFramePoint, glm::vec4 relativeFramePoint) {
		return toString(baseFramePoint) + " (base)\n" + toString(relativeFramePoint) + " (relative)\n";
	}

	template <typename T>
	std::string toString(std::string header, std::vector<std::vector<T>> translations, std::vector<std::vector<T>> rotations) {
		std::stringstream ss;
		ss << header << std::endl;

		ss << "Translations: " << toString(translations) << std::endl;
		ss << "   Rotations: " << toString(rotations) << std::endl;

		ss << std::endl << "********************************************************************************" << std::endl << std::endl;

		return ss.str();
	}


	template <typename T>
	std::string toString(std::vector<T> v, std::string valueDelimiter = ", ") {
		std::stringstream ss;

		for (auto& value : v)
		{
			ss << value << valueDelimiter;
		}
		std::string intermediate = ss.str();
		intermediate.erase(intermediate.length() - valueDelimiter.length(), intermediate.length());

		return intermediate;
	}

	template <typename T>
	std::string toString(std::vector<std::vector<T>> vec, std::string valueDelimiter = ", ", std::string pipeDelimiter = " | ") {
		std::stringstream ss;
		
		for (auto& v : vec)
		{
			ss << toString(v);
			ss << pipeDelimiter;
		}
		std::string intermediate = ss.str();
		intermediate.erase(intermediate.length() - pipeDelimiter.length(), intermediate.length());
		
		return intermediate;
	}

	std::string toString(std::string header, std::vector<Eigen::Matrix4d> b) {
		std::stringstream ss;
		ss << header << std::endl;
		
		for(auto & m : b)
		{
			ss << m << std::endl << std::endl;
		}
		
		return ss.str();
	}

	std::string toString(std::string header, Eigen::Matrix4d b) {
		std::stringstream ss;
		ss << header << std::endl;

		ss << b << std::endl << std::endl;

		return ss.str();
	}
}

#endif // !_UTILS_HEADER
