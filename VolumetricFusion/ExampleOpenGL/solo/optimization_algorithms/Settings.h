#pragma once

namespace solo {
	
	enum class LinearSolverType {
		SPARSE_PCG_CPU,
		SPARSE_PCG_COMPLETE_GPU,
		SPARSE_PCG_SEQUENTIAL_GPU,
		SPARSE_PCG_ATOMIC_GPU,
		DENSE_QR_CPU,
		DENSE_QR_GPU
	};

	enum class AlgorithmType {
		GAUSS_NEWTON,
		LEVENBERG_MARQUARDT
	};

	/**
	 * Settings class cointains all the parameters that you can change to influence the solver behaviour.
	 * It needs to be added to the Solve() method in order to start the problem optimization.
	 */
	class Settings {
	public:
		LinearSolverType linearSolverType{ LinearSolverType::DENSE_QR_CPU };
		AlgorithmType algorithmType{ AlgorithmType::GAUSS_NEWTON };
		unsigned algorithmMaxNumIterations{ 10 };
		unsigned linearSolverMaxNumIterations{ 100 };
		float initialLMWeight{ 1e-20f };

		// Copies the residuals to the vector that is returned with status.
		// The residuals are evaluted only for active constraints.
		bool bEvaluateResiduals{ false }; 

		// Copies the jacobian L2 norms (one for each residual instance) to the vector that is returned 
		// with status. The jacobian L2 norms are evaluted only for active constraints.
		bool bEvaluateJacobians{ false };

		// If all parameter pointers are given in the same order as they appear in memory, then you can
		// turn this option to true and a more efficient version of the algorithm can run.
		bool bOrderedParameters{ false }; 

		// If enabled, it will check the termination criteria of the current algorithm (e.g. Gauss-Newton)
		// after each iteration, and exit the optimization early if the criteria is satisfied before the
		// maximum number of iterations is reached.
		bool bEnableAlgorithmEarlyStop{ false };
		float gradientTolerance{ 1e-10 };
		float functionTolerance{ 1e-10 };

		// If enabled, it will check the Q-termination criteria for the PCG linear solver.
		// (x^T A x - xPrev^T A xPrev) < eps * (x^T A x)
		bool bUseQStoppingCriteria{ false };
		float qTolerance{ 0.0001 };

		// The PCG solver always checks the R-termination criteria, for numerical stability.
		// |Ax - b| < eps * |b|
		float rTolerance{ std::numeric_limits<float>::epsilon() };

		// If enabled, all parameters are checked whether they are present in the optimization problem, and
		// non-present parameters (not in any of the index matrices) are completely excluded from optimization.
		// If you are sure all your parameters are present in optimization, you should disable it, since it
		// will accelerate the optimization considerably.
		bool bCheckParamUsage{ true };

		// If your index matrices are stored on the GPU, it makes sense to use GPU computation of index
		// vector, since then no copy to the CPU/host memory is needed (unless we also check for parameter
		// usage, since copy to CPU/host memory is needed then).
		bool bUseGPUForIndexVectorComputation{ false };
	};

} // namespace solo