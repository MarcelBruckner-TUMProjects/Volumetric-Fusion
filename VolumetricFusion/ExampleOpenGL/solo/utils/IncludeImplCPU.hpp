#pragma once

// Important: This file can only be included in exactly one .cpp file, otherwise the methods would get compiled 
// multiple times.

#include <common_utils/IncludeImplCPU.hpp>

#include "IncludeCPU.h"
#include "solo/linear_solvers/DenseQRSolverCPU_Impl.hpp"
#include "solo/linear_solvers/SparsePCGSolverCPU_Impl.hpp"
#include "solo/linear_solvers/LossComputationCPU_Impl.hpp"
#include "solo/optimization_algorithms/IndexProcessing_ImplCPU.hpp"
#include "solo/constraint_evaluation/ParameterProcessing_ImplCPU.hpp"