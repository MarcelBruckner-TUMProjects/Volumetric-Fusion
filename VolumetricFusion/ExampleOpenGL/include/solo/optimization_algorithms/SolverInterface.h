#pragma once
#include "Settings.h";
#include "AlgorithmFactory.h";
#include "SolverProcessing.h";
#include "solo/linear_solvers/LinearSolverFactory.h";
#include "Status.h"
#include "GradientChecker.h"
#include "solo/data_structures/ParamVector.h"

namespace solo {
	
	/**
	 * Executes the optimization algorithm, given the optimization settings and a list of optimization
	 * constraints (objects of type Constraint). At least one constraint needs to be added.
	 * When the optimization is complete, it writes the parameter solutions to the memory locations, given 
	 * at the initialization of constraints.
	 * @param	settings	Solver settings
	 * @returns The final status of the optimization
	 */
	template<typename FloatType, typename ParamStorageType, typename ...Constraints>
	inline Status Solve(const Settings& settings, const ParamVector<FloatType, ParamStorageType>& paramVector, Constraints&&... constraints) {
		// Check that all constraints request the same accuracy (have the FloatType type).
		solver_proc::checkFloatTypeMatch(Type2Type<FloatType>(), std::forward<Constraints>(constraints)...);

		// Initialize linear solver.
		auto linearSolver = LinearSolverFactory<FloatType>::get(settings.linearSolverType, settings);

		// Initialize optimization algorithm.
		auto algorithm = AlgorithmFactory<FloatType, ParamStorageType, Constraints...>::get(settings.algorithmType, settings, *linearSolver);

		// Execute the optimization algorithm.
		return algorithm->execute(paramVector, std::forward<Constraints>(constraints)...);
	}


	/**
	 * Computes gradients for all constraints with both a given method and numerical derivation.
	 * It compares the results and prints any mismatched partial derivatives to the standard 
	 * output.
	 * @param	eps		Maximum allowed absolute difference between numerical and 
	 *					actual derivatives
	 * @returns True if all partial derivative match, otherwise false
	 */
	template<typename FloatType, typename ParamStorageType, typename ...Constraints>
	inline bool CheckGradients(double eps, const ParamVector<FloatType, ParamStorageType>& paramVector, Constraints&&... constraints) {
		// Check that all constraints request the same accuracy (have the FloatType type).
		solver_proc::checkFloatTypeMatch(Type2Type<FloatType>(), std::forward<Constraints>(constraints)...);

		// Run gradient checker.
		return GradientChecker<FloatType, ParamStorageType>::check(eps, paramVector, std::forward<Constraints>(constraints)...);
	}
} // namespace solo