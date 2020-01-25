#pragma once
#ifndef _CERES_OPTIMIZATION_HEADER
#define _CERES_OPTIMIZATION_HEADER

#include "CharacteristicPoints.hpp"
#include "OptimizationProblem.hpp"
#include "PointCorrespondenceError.hpp"
#include "ReprojectionError.hpp"

namespace vc::optimization {

	class CeresOptimizationProblem : public OptimizationProblem {
	
	private:

	public:

		CeresOptimizationProblem(bool verbose = false, long sleepDuration = -1l) 
			:OptimizationProblem(verbose, sleepDuration) {

		}

		virtual bool solveErrorFunction() = 0;
		
		virtual void initialize() = 0;

	};
}

#endif