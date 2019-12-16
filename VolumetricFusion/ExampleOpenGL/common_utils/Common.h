#pragma once

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

#ifndef SAFE_DELETE
#define SAFE_DELETE(p)       { if (p) { delete (p);     (p)=nullptr; } }
#endif

#ifndef SAFE_DELETE_ARRAY
#define SAFE_DELETE_ARRAY(p) { if (p) { delete[] (p);   (p)=nullptr; } }
#endif

#ifndef FLT_EPSILON
#define FLT_EPSILON ((float)1.19209290E-07F)
#endif

// To use timers to time execution, you should define TIME_EXECUTION. It might make
// your execution slower.
//#define TIME_EXECUTION

// For compilation of CUDA related code, you should define COMPILE_CUDA.
//#define COMPILE_CUDA

// To debug all CUDA kernel calls, you need to define CUDA_ERROR_CHECK flag. It will
// bring down the performance though.
//#define CUDA_ERROR_CHECK

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas.h>
#include <cusparse.h>
#include <iostream>
#endif

#ifdef __CUDACC__
#define CPU_AND_GPU __host__  __device__ // For CUDA host and device code.
#else
#define CPU_AND_GPU
#endif

#ifdef __CUDACC__
#define CONSTANT __constant__  // For CUDA host and device code.
#else
#define CONSTANT
#endif

// Macro for forcing a function to NOT be inlined and function macro, telling the current executing
// function.
#ifdef __linux__
#define NO_INLINE __attribute__((noinline))
#define __FUNCTION__ __func__
#else
#define NO_INLINE __declspec(noinline)
#endif

#define FUNCTION_LINE_STRING (std::string(__FUNCTION__) + ":" + std::to_string(__LINE__))

// Macros to catch CUDA errors in CUDA runtime calls.

#ifdef __CUDACC__
// CUDA error checking.
#define CUDA_SAFE_CALL( err ) __cudaSafeCall( err, std::string(__FILE__), FUNCTION_LINE_STRING )

inline void __cudaSafeCall(cudaError err, const std::string& file, const std::string& functionWithLineNb) {
	#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err) {
		std::ostringstream outputStream;
		outputStream << "cudaSafeCall() failed in file '" << file << "' at " << functionWithLineNb << " with message: " << cudaGetErrorString(err);

		std::cerr << outputStream.str() << std::endl;
		exit(-1);
	}
	#endif

	return;
}

// For debugging.
#define CUDA_CHECK_ERROR() __cudaCheckError( std::string(__FILE__), FUNCTION_LINE_STRING )   

inline void __cudaCheckError(const std::string& file, const std::string& functionWithLineNb) {
	#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		std::ostringstream outputStream;
		outputStream << "cudaCheckError() with sync failed in file '" << file << "' at " << functionWithLineNb << " with message: " << cudaGetErrorString(err);

		std::cerr << outputStream.str() << std::endl;
		exit(-1);
	}
	#endif

	return;
}

// CUBLAS error checking.
#define CUBLAS_SAFE_CALL( err ) __cublasSafeCall( err, std::string(__FILE__), FUNCTION_LINE_STRING )

inline const char *_cublasGetErrorEnum(cublasStatus_t error) {
	switch (error) {
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";

	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";

	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";

	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";

	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";

	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";

	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";

	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";

	case CUBLAS_STATUS_NOT_SUPPORTED:
		return "CUBLAS_STATUS_NOT_SUPPORTED";

	case CUBLAS_STATUS_LICENSE_ERROR:
		return "CUBLAS_STATUS_LICENSE_ERROR";
	}

	return "<unknown>";
}

inline void __cublasSafeCall(cublasStatus_t err, const std::string& file, const std::string& functionWithLineNb) {
	if (CUBLAS_STATUS_SUCCESS != err) {
		std::ostringstream outputStream;
		outputStream << "CUBLAS error in file '" << file << "' at " << functionWithLineNb << " with message: " << _cublasGetErrorEnum(err);

		std::cerr << outputStream.str() << std::endl;
		exit(-1);
	}
}

// CUSPARSE error checking.
#define CUSPARSE_SAFE_CALL( err ) __cusparseSafeCall(err, std::string(__FILE__), FUNCTION_LINE_STRING)

inline const char *_cusparseGetErrorEnum(cusparseStatus_t error) {
	switch (error) {
	case CUSPARSE_STATUS_SUCCESS:
		return "CUSPARSE_STATUS_SUCCESS";

	case CUSPARSE_STATUS_NOT_INITIALIZED:
		return "CUSPARSE_STATUS_NOT_INITIALIZED";

	case CUSPARSE_STATUS_ALLOC_FAILED:
		return "CUSPARSE_STATUS_ALLOC_FAILED";

	case CUSPARSE_STATUS_INVALID_VALUE:
		return "CUSPARSE_STATUS_INVALID_VALUE";

	case CUSPARSE_STATUS_ARCH_MISMATCH:
		return "CUSPARSE_STATUS_ARCH_MISMATCH";

	case CUSPARSE_STATUS_MAPPING_ERROR:
		return "CUSPARSE_STATUS_MAPPING_ERROR";

	case CUSPARSE_STATUS_EXECUTION_FAILED:
		return "CUSPARSE_STATUS_EXECUTION_FAILED";

	case CUSPARSE_STATUS_INTERNAL_ERROR:
		return "CUSPARSE_STATUS_INTERNAL_ERROR";

	case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

	case CUSPARSE_STATUS_ZERO_PIVOT:
		return "CUSPARSE_STATUS_ZERO_PIVOT";
	}

	return "<unknown>";
}

inline void __cusparseSafeCall(cusparseStatus_t err, const std::string& file, const std::string& functionWithLineNb) {
	if (CUSPARSE_STATUS_SUCCESS != err) {
		std::ostringstream outputStream;
		outputStream << "CUSPARSE error in file '" << file << "' at " << functionWithLineNb << " with message: " << _cusparseGetErrorEnum(err);

		std::cerr << outputStream.str() << std::endl;
		exit(-1);
	}
}

#else
// Just to make the code highlighting work also in CPU only mode.
#define CUDA_SAFE_CALL( x ) x
#define CUDA_CHECK_ERROR( ) 
#define CUBLAS_SAFE_CALL( x ) x
#define CUSPARSE_SAFE_CALL( x ) x
#endif

namespace common_utils {
	
} // namespace common_utils