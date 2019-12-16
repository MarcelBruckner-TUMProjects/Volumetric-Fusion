#pragma once
#include "TimerCPU.h"

#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#endif

#if defined(__linux__)
#include <sys/time.h>
#endif

namespace common_utils {

	double TimerCPU::getTime() const {
		#if defined(_WIN32) || defined(_WIN64)
		unsigned __int64 pf;
		QueryPerformanceFrequency((LARGE_INTEGER *)&pf);
		double freq_ = 1.0 / (double)pf;

		unsigned __int64 val;
		QueryPerformanceCounter((LARGE_INTEGER *)&val);
		return (val)* freq_;
		#endif // defined(_WIN32) || defined(_WIN64)

		#if defined(__linux__)
		struct timeval timevalue;
		gettimeofday(&timevalue, nullptr);
		return (double)((long)timevalue.tv_sec) + (double)((long)timevalue.tv_usec) / 1000000.0;
		#endif // defined(__linux__)
	}

} // namespace common_utils


