#pragma once
#include "RuntimeAssertion.h"
#include <iostream>

void handleRuntimeAssertFailure(bool expressionVal, const std::string& expressionStr, const std::string& file, const std::string& functionWithLineNb, const std::string& message) {
	if (!expressionVal) {
		std::ostringstream outputStream;
		outputStream << "Assertion '" << expressionStr << "' failed in file '" << file << "' at " << functionWithLineNb;
		if (!message.empty()) outputStream << " with message: " << message;

		std::cerr << outputStream.str() << std::endl;
		
		#ifdef _MSC_VER
		__debugbreak();
		#endif

		exit(-1);
	}
}