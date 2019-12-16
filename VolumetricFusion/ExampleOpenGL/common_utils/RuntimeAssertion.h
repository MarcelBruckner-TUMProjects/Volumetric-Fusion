#pragma once
#include "Common.h"

/**
 * Method prints out the assertion error message, and exits.
 */
NO_INLINE void handleRuntimeAssertFailure(bool expressionVal, const std::string& expressionStr, const std::string& file, const std::string& functionWithLineNb, const std::string& message);
CPU_AND_GPU inline void handleRuntimeAssertFailureDevice(bool expressionVal, const char* expressionStr, const char* file, const char* function, unsigned lineNb, const char* message) {
	if (!expressionVal) {
		std::ostringstream outputStream;
		fprintf(stderr, "Assertion '%s' failed in file '%s' at %s:%d with message: %s\n", expressionStr, file, function, lineNb, message);

		#ifdef _MSC_VER
		__debugbreak();
		#endif

		exit(-1);
	}
}

#if (!defined(NDEBUG) || defined(ENABLE_ASSERTIONS))
	#if !defined(__CUDA_ARCH__)
	#define runtime_assert(EXPRESSION, MESSAGE) if(!(EXPRESSION)) handleRuntimeAssertFailure(EXPRESSION, std::string(#EXPRESSION), std::string(__FILE__), FUNCTION_LINE_STRING, MESSAGE); ; 
	#else
	#define runtime_assert(EXPRESSION, MESSAGE)
	//#define runtime_assert(EXPRESSION, MESSAGE) if(!(EXPRESSION)) handleRuntimeAssertFailureDevice(EXPRESSION, #EXPRESSION, __FILE__, __FUNCTION__, __LINE__, MESSAGE); 
	#endif
#else
	#define runtime_assert(EXPRESSION, MESSAGE)
#endif