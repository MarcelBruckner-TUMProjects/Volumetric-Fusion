#pragma once

#ifndef _UTILS_HEADER
#define _UTILS_HEADER

#include <vector>
#include <string>
#include <sstream>

#include <iostream>
#include <chrono>
#include <thread>

#include "Structs.hpp"

#define NAME_AND_VALUE(name) vc::utils::nameAndValue(#name, (name))

namespace vc::utils {

	const int NUM_THREADS = -1000;
	
	std::string asHeader(std::string header) {
		std::stringstream ss;

		ss << std::endl;
		ss << "*****************************************************************************************************" << std::endl;
		ss << header << std::endl;
		ss << "*****************************************************************************************************" << std::endl;

		return ss.str();
	}

	template <typename T>
	std::string nameAndValue(char* name, T value) {
		std::stringstream ss;
		ss << asHeader(name) << value << std::endl;
		return ss.str();
	}

	void sleepFor(std::string message, long milliseconds, bool verbose = false) {
		if (milliseconds < 0) {
			return;
		}
		using namespace std::chrono_literals;
		std::stringstream ss;
		ss << vc::utils::asHeader(message) << std::endl;
		ss << "Sleeping" << std::endl;
		std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
		ss << "Awaken" << std::endl;

		if (verbose) {
			std::cout << ss.str();
		}
	}
	
	template <typename T>
	std::string toString(std::string header, std::vector<std::vector<T>> translations, std::vector<std::vector<T>> rotations) {
		std::stringstream ss;
		ss << asHeader(header);

		ss << "Translations: " << toString(translations) << std::endl;
		ss << "   Rotations: " << toString(rotations) << std::endl;

		ss << std::endl << "********************************************************************************" << std::endl << std::endl;

		return ss.str();
	}

	template <typename T>
	std::string toString(std::vector<T> v, std::string valueDelimiter = ", ") {
		if (v.empty()) {
			return "";
		}

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
	std::string toString(std::string header, std::vector<T> v, std::string valueDelimiter = ", ") {
		return asHeader(header) + toString(v, valueDelimiter);
	}

	template <typename T>
	std::string toString(std::vector<std::vector<T>> vec, std::string valueDelimiter = ", ", std::string pipeDelimiter = " | ") {
		if (vec.empty()) {
			return "";
		}
		
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

	template <typename T>
	std::string toString(std::string header, std::vector<std::vector<T>> vec, std::string valueDelimiter = ", ", std::string pipeDelimiter = " | ") {
		return asHeader(header) + toString(vec, valueDelimiter, pipeDelimiter);
	}

	std::string toString(std::string header, std::vector<Eigen::Matrix4d> b) {
		std::stringstream ss;
		ss << asHeader(header);

		for (auto& m : b)
		{
			ss << m << std::endl << std::endl;
		}

		return ss.str();
	}

	std::string toString(std::string header, Eigen::Matrix3d b) {
		std::stringstream ss;
		ss << asHeader(header);

		ss << b << std::endl << std::endl;

		return ss.str();
	}

	std::string toString(std::string header, Eigen::Matrix4d b) {
		std::stringstream ss;
		ss << asHeader(header);

		ss << b << std::endl << std::endl;

		return ss.str();
	}

	std::string toString(glm::vec4 pos) {
		std::stringstream ss;

			ss << pos[0] << " | " << pos[1] << " | " << pos[2] << " | " << pos[3] << std::endl;

		return ss.str();
	}

	std::string toString(vc::fusion::Triangle* b) {
		std::stringstream ss;

		for (auto& pos : b->pos)
		{
			ss << pos[0] << " | " << pos[1] << " | " << pos[2] << " | " << pos[3] << std::endl;
		}

		return ss.str();
	}

	std::string toString(std::string header, vc::fusion::Triangle* b) {
		std::stringstream ss;
		ss << asHeader(header);

		for (auto& pos : b->pos)
		{
			ss << pos[0] << " | " << pos[1] << " | " << pos[2] << " | " << pos[3] << std::endl;
		}
		ss << std::endl;

		return ss.str();
	}

	std::string toString(std::string header, Eigen::Vector4d b) {
		std::stringstream ss;
		ss << asHeader(header);

		ss << b << std::endl << std::endl;

		return ss.str();
	}

	std::string toString(std::string header, std::map<unsigned long long, Eigen::Vector4d> b) {
		std::stringstream ss;
		ss << asHeader(header);

		for (auto& m : b)
		{
			ss << m.first << ":" << std::endl << m.second << std::endl;
		}

		return ss.str();
	}

	template <typename T>
	bool contains(std::vector<T> v, T x) {
		return std::find(v.begin(), v.end(), x) != v.end();
	}

	template<typename T, typename V>
	std::vector<T> extractKeys(std::map<T, V> const& input_map) {
		std::vector<T> retval;
		for (auto const& element : input_map) {
			retval.emplace_back(element.first);
		}
		return retval;
	}


	template<typename T>
	std::vector<T> findOverlap(std::vector<T> a, std::vector<T> b) {
		std::vector<T> c;

		for (T x : a) {
			if (contains(b, x)) {
				c.emplace_back(x);
			}
		}

		return c;
	}

	template <typename T, typename V>
	std::map<T, V> filter(std::map<T, V> map, std::vector<T> keys) {
		std::map<T, V> filtered;

		for (auto& key : keys)
		{
			filtered[key] = map[key];
		}

		return filtered;
	}

	template <typename T>
	std::string toString(std::string header, T value) {
		std::stringstream ss;
		ss << header << std::endl << value;
		return asHeader(ss.str());
	}

	bool isValid(vc::fusion::Triangle* triangle) {
		return !(
			triangle->pos[0].x == 0 && triangle->pos[0].w == 0 && triangle->pos[0].z == 0 && triangle->pos[0].y == 0 &&
			triangle->pos[1].x == 0 && triangle->pos[1].w == 0 && triangle->pos[1].z == 0 && triangle->pos[1].y == 0 &&
			triangle->pos[2].x == 0 && triangle->pos[2].w == 0 && triangle->pos[2].z == 0 && triangle->pos[2].y == 0);
	}

	bool isValid(glm::vec4 v) {
		for (int i = 0; i < 4; i++)
		{
			if (std::abs(v[i]) > 10e2) {
				return false;
			}
		}
		return true;
	}

}

#endif // !_UTILS_HEADER
