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

	template <typename T>
	std::string toString(T* matrix, int rows = 4, int columns = 4, std::string name = "") {
		std::stringstream ss;
		std::string delimiter = " ";

		for (int row = 0; row < rows; row++)
		{
			ss << name << ": ";
			for (int column = 0; column < columns; column++)
			{
				auto m = matrix[row + column * rows];
				ss << m << delimiter;
			}
			std::string intermediate = ss.str();
			intermediate.erase(intermediate.length() - delimiter.length(), intermediate.length());

			ss = std::stringstream();
			ss << intermediate;
			ss << std::endl;
		}
		ss << std::endl;

		return ss.str();
	}

	template <typename T, typename U>
	std::string toString(T* matrix1, U* matrix2, int rows = 4, int columns = 4, std::string name = "") {
		std::stringstream ss;
		std::string delimiter = " ";

		for (int row = 0; row < rows; row++)
		{
			ss << name << ": ";
			for (int column = 0; column < columns; column++)
			{
				auto m = matrix1[row + column * rows];
				ss << m << delimiter;
			}
			ss << "--> ";
			for (int column = 0; column < columns; column++)
			{
				auto m = matrix2[row + column * rows];
				ss << m << delimiter;
			}
			std::string intermediate = ss.str();
			intermediate.erase(intermediate.length() - delimiter.length(), intermediate.length());

			ss = std::stringstream();
			ss << intermediate;
			ss << std::endl;
		}
		ss << std::endl;

		return ss.str();
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
	std::string toString(std::vector<std::vector<T>> vec) {
		std::stringstream ss;
		std::string valueDelimiter = ", ";
		std::string pipeDelimiter = " | ";

		for (auto& v : vec)
		{
			for (auto& value : v)
			{
				ss << std::to_string(value) << valueDelimiter;
			}
			std::string intermediate = ss.str();
			intermediate.erase(intermediate.length() - valueDelimiter.length(), intermediate.length());

			ss = std::stringstream();
			ss << intermediate;

			ss << pipeDelimiter;
		}
		std::string intermediate = ss.str();
		intermediate.erase(intermediate.length() - pipeDelimiter.length(), intermediate.length());
		
		return intermediate;
	}
}

#endif // !_UTILS_HEADER
