#pragma once

// Important: This file can only be included in exactly one .cu file, otherwise the methods would get compiled 
// multiple times.

#include <common_utils/IncludeImplCUDA.hpp>
#include <common_utils/memory_managment/MemoryWrapper.h>

#include "IncludeCUDA.h"
#include "solo/linear_solvers/SparsePCGSolverCompleteGPU_ImplCUDA.hpp"
#include "solo/linear_solvers/SparsePCGSolverSequentialGPU_ImplCUDA.hpp"
#include "solo/linear_solvers/LossComputationGPU_ImplCUDA.hpp"
#include "solo/linear_solvers/SparsePCGSolverAtomicGPU_ImplCUDA.hpp"
#include "solo/optimization_algorithms/IndexProcessing_ImplCUDA.hpp"
#include "solo/constraint_evaluation/ParameterProcessing_ImplCUDA.hpp"

// Important: If you use some new memory block(even only on CPU), it needs to be added here, because
// otherwise the destructor won't be compiled.
template class common_utils::MemoryContainer<int>;		// Used in index vectors.
template class common_utils::MemoryWrapper<int>;		// Used in index matrices.
template class common_utils::MemoryContainer<float>;	// When we compile CPU solvers, we want to compile both float and double versions. 
template class common_utils::MemoryContainer<double>;	// Also used in parameter/residual/Jacobian matrices.

// TODO: Remove!
template class common_utils::MemoryContainer<float*>;	// When we compile parameter manager, the pointers point to the locations of parameters 
template class common_utils::MemoryContainer<double*>;	// (float or double).

// For using the index lists, stored in GPU/device memory.
template void solo::memory_proc::copyMemory<int>(int* outputDataPtr, const int* inputDataPtr, unsigned size, Type2Type<MemoryTypeCUDA>);