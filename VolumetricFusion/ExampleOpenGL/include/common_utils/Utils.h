#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

using std::cout;
using std::endl;
using std::vector;

namespace common_utils {

	/**
		* Returns a string, where the given 'number' is padded with zeros, such that the total string width is
		* equal to 'width'.
		*/
	inline std::string zeroPaddedNumber(int number, int width) {
		std::stringstream ss;
		ss << std::setw(width) << std::setfill('0') << number;
		return ss.str();
	}

	/**
		* Splits an input string by a given delimiter. It either expepts an iterator as a parameter, or
		* returns a vector of string segments.
		*/
	template<typename Out>
	inline void split(const std::string &s, char delim, Out result) {
		std::stringstream ss;
		ss.str(s);
		std::string item;
		while (std::getline(ss, item, delim)) {
			*(result++) = item;
		}
	}
	inline vector<std::string> split(const std::string &s, char delim) {
		vector<std::string> elems;
		split(s, delim, std::back_inserter(elems));
		return elems;
	}

	/**
		* Not implemented exception.
		*/
	class NotImplemented : public std::logic_error {
	public:
		NotImplemented() : std::logic_error("Function not implemented") {};
	};

} // namespace common_utils