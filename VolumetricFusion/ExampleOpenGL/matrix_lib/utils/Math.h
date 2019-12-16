#pragma once
#include <common_utils/Common.h>

namespace matrix_lib {
	namespace math_proc {
		
		static const double PI = 3.1415926535897932384626433832795028842;
		static const float PIf = 3.14159265358979323846f;

		CPU_AND_GPU inline float degreesToRadians(float x) {
			return x * (PIf / 180.0f);
		}

		CPU_AND_GPU inline float radiansToDegrees(float x) {
			return x * (180.0f / PIf);
		}

		CPU_AND_GPU inline double degreesToRadians(double x) {
			return x * (PI / 180.0);
		}

		CPU_AND_GPU inline double radiansToDegrees(double x) {
			return x * (180.0 / PI);
		}

		template<class T>
		CPU_AND_GPU long int floor(T x) {
			return (long int)std::floor(x);
		}

		template<class T>
		CPU_AND_GPU long int ceil(T x) {
			return (long int)std::ceil(x);
		}

		template<class T>
		CPU_AND_GPU bool floatEqual(T v0, T v1, T eps = (T)0.000001) {
			return (std::abs(v0 - v1) <= eps);
		}

		using std::round;
		using std::abs;
		using std::sqrt;
		using std::max;
		using std::min;

	} // namespace math_proc
} // namespace matrix_lib