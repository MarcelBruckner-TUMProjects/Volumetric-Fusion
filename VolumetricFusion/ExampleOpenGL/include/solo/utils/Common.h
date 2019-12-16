#pragma once
#include <common_utils/Common.h>

// If you want to debug the optimization time, you can define the following flag.
// But be aware that it will slow down some parts of the optimization, since it needs
// to block the synchronize with the GPU.
//#define TIME_EXECUTION

// For linear system solving using GPU, the COMPILE_CUDA flag should be defined.
//#define COMPILE_CUDA

// For constraint evaluation on the GPU, the USE_GPU_EVALUTION flag should be defined.
//#define USE_GPU_EVALUATION

// If we use GPU evaluation  of constraints, we need to add COMPILE_CUDA flag as well.
#ifdef USE_GPU_EVALUATION
#ifndef COMPILE_CUDA
#define COMPILE_CUDA
#endif
#endif

namespace solo {
	
} // namespace solo