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
	std::string asHeader(std::string header) {
		std::stringstream ss;

		ss << std::endl;
		ss << "*****************************************************************************************************" << std::endl;
		ss << header << std::endl;
		ss << "*****************************************************************************************************" << std::endl;

		return ss.str();
	}

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
	
	template <typename T>
	std::string toString(std::string header, std::vector<std::vector<T>> translations, std::vector<std::vector<T>> rotations) {
		std::stringstream ss;
		ss << asHeader(header) << std::endl;

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
		ss << asHeader(header) << std::endl;

		for (auto& m : b)
		{
			ss << m << std::endl << std::endl;
		}

		return ss.str();
	}

	std::string toString(std::string header, Eigen::Matrix4d b) {
		std::stringstream ss;
		ss << asHeader(header) << std::endl;

		ss << b << std::endl << std::endl;

		return ss.str();
	}

	std::string toString(std::string header, Eigen::Vector4d b) {
		std::stringstream ss;
		ss << asHeader(header) << std::endl;

		ss << b << std::endl << std::endl;

		return ss.str();
	}
}

#endif // !_UTILS_HEADER
