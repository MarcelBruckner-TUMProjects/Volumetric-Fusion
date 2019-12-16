#pragma once
#include "Solo/data_structures/JacobianMatrix.h"
#include "Solo/data_structures/ResidualVector.h"
#include "Solo/optimization_algorithms/Settings.h"

namespace solo {

	struct TerminationParameters {
		double gradientTolerance{ 1e-10 };
		bool bUseGNEarlyStopping{ false };
		double qTolerance{ 0.0001 };
		double rTolerance{ std::numeric_limits<float>::epsilon() };
		bool bUsePCGEarlyStopping{ false };

		TerminationParameters(float gradientTolerance_, bool bUseGNEarlyStopping_, float qTolerance_, float rTolerance_, bool bUsePCGEarlyStopping_) :
			gradientTolerance{ gradientTolerance_ },
			bUseGNEarlyStopping{ bUseGNEarlyStopping_ },
			qTolerance{ qTolerance_ },
			rTolerance{ rTolerance_ },
			bUsePCGEarlyStopping{ bUsePCGEarlyStopping_ }
		{ }
	};

	struct TerminationCriteria {
		double maxGradientNorm{ -1.0 };
		bool bStopEarly{ false };
	};

	template<typename FloatType>
	class LinearSolver {
	public:
		virtual ~LinearSolver() = default;

		/**
		 * Solves the linear system (J^T J + lambda I) x = -J^T r, where r is a residual vector and J is a 
		 * Jacobian matrix.
		 * The r and J pairs are given for every optimization constraint separately, but the index vectors
		 * (with unique parameter indices for each Jacobian element and starting row indices for each residual) 
		 * are given for all constraints (concatenated together). 
		 * The index vectors are stored in CPU memory, at least at the first iteration. The solver is responsible
		 * to copy them to the GPU memory, if necessary. Special care needs to be taken also about the system
		 * components, which could be stored in CPU or GPU (if we use GPU constraint evaluation) memory.
		 * The computed system solution is written into the dense matrix solution, which can be stored in either
		 * CPU (host) or GPU (device) memory. The solver needs to signal the used memory with the usage of update
		 * flags (e.g., if GPU memory is used, after the computation the GPU memory should be set to updated, and 
		 * the CPU memory should not be set as updated).
		 * @return The squared loss, i.e. the sum of squared residual components (for all constraints).
		 */
		virtual FloatType solve(
			vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& systemComponents, 
			DenseMatrix<int>& columnIndexVector,
			DenseMatrix<int>& rowIndexVector,
			unsigned nParameters,
			FloatType lambda,
			TerminationCriteria& terminationCriteria,
			int& nIterations,
			DenseMatrix<FloatType>& solution
		) = 0;

		virtual LinearSolverType getType() const = 0;
	};

} // namespace solo