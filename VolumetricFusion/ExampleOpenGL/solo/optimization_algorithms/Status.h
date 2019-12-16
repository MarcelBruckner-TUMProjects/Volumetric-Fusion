#pragma once
#include "Solo/utils/IncludeLib.h"

namespace solo {
	
	/**
	 * It stores information about a particular optimization iteration.
	 */
	struct IterationInfo {
		int iterationNum{ -1 };
		int nLinearSolverIterations{ 0 };
		double loss{ -1.0 };
		double lossChange{ -1.0 };
		double maxGradientNorm{ -1.0 };

		IterationInfo(int iterationNum_, int nLinearSolverIterations_, double loss_, double lossChange_, double maxGradientNorm_) :
			iterationNum{ iterationNum_ },
			nLinearSolverIterations{ nLinearSolverIterations_ },
			loss{ loss_ },
			lossChange{ lossChange_ },
			maxGradientNorm{ maxGradientNorm_ }
		{ }

		static void print(const vector<IterationInfo>& iterationInfos) {
			cout << "  Iter.  |  Optimization loss" << endl;
			for (const auto& iterationInfo : iterationInfos) {
				cout << std::setw(7) << std::right << iterationInfo.iterationNum << "  |  "
					<< std::setw(15) << std::right << std::setprecision(6) << iterationInfo.loss << endl;
			}
		}

		static void printDetail(const vector<IterationInfo>& iterationInfos) {
			cout << "  Iter.  |  Optimization loss  |     Loss change     |  Max. gradient norm  |  Num. LS iter.  " << endl;
			for (const auto& iterationInfo : iterationInfos) {
				cout << std::setw(7) << std::right << iterationInfo.iterationNum << "  |  "
					<< std::setw(15) << std::right << std::setprecision(6) << iterationInfo.loss << "    |  "
					<< std::setw(15) << std::right << std::setprecision(6) << iterationInfo.lossChange << "    |  "
					<< std::setw(15) << std::right << std::setprecision(6) << iterationInfo.maxGradientNorm << "     |  "
					<< std::setw(7) << std::right << iterationInfo.nLinearSolverIterations << endl;
			}
		}

		static void printCeres(const vector<IterationInfo>& iterationInfos) {
			cout << "  Iter.  |  Optimization loss" << endl;
			for (const auto& iterationInfo : iterationInfos) {
				cout << std::setw(7) << std::right << iterationInfo.iterationNum << "  |  "
					<< std::setw(15) << std::right << std::setprecision(6) << (0.5 * iterationInfo.loss * iterationInfo.loss) << endl;
			}
		}

		static void printDetailCeres(const vector<IterationInfo>& iterationInfos) {
			cout << "  Iter.  |  Optimization loss  |     Loss change     |  Max. gradient norm  |  Num. LS iter.  " << endl;
			for (const auto& iterationInfo : iterationInfos) {
				cout << std::setw(7) << std::right << iterationInfo.iterationNum << "  |  "
					<< std::setw(15) << std::right << std::setprecision(6) << (0.5 * iterationInfo.loss * iterationInfo.loss) << "    |  "
					<< std::setw(15) << std::right << std::setprecision(6) << iterationInfo.lossChange << "    |  "
					<< std::setw(15) << std::right << std::setprecision(6) << iterationInfo.maxGradientNorm << "     |  "
					<< std::setw(7) << std::right << iterationInfo.nLinearSolverIterations << endl;
			}
		}
	};


	/**
	 * It stores time statistics for the optimization.
	 */
	struct TimeReport {
		double constraintEvaluation{ -1.0 };
		double indexVectorComputation{ -1.0 };
		double linearSolver{ -1.0 };
		double parameterUpdate{ -1.0 };
		double residualEvaluation{ -1.0 };
		double totalTime{ -1.0 };

		static void print(const TimeReport& timeReport) {
			cout << std::setw(40) << std::left << "Index computation (s): " << std::setprecision(6) << timeReport.indexVectorComputation << endl;
			cout << std::setw(40) << std::left << "Constraint evaluation/iteration (s): " << std::setprecision(6) << timeReport.constraintEvaluation << endl;
			cout << std::setw(40) << std::left << "Linear system solving/iteration (s): " << std::setprecision(6) << timeReport.linearSolver << endl;
			cout << std::setw(40) << std::left << "Parameter update/iteration (s): " << std::setprecision(6) << timeReport.parameterUpdate << endl;
			cout << std::setw(40) << std::left << "Residual evaluation/iteration (s): " << std::setprecision(6) << timeReport.residualEvaluation << endl;
			cout << std::setw(40) << std::left << "Total time (s): " << std::setprecision(6) << timeReport.totalTime << endl;
		}
	};


	/**
	 * It stores memory statistics for the optimization.
	 */
	struct MemoryReport {
		size_t localData{ 0 };
		size_t globalData{ 0 };
		size_t parameterMatrix{ 0 };
		size_t indexVectors{ 0 };
		size_t residualVector{ 0 };
		size_t jacobianMatrix{ 0 };

		static void print(const MemoryReport& memoryReport) {
			cout << std::setw(40) << std::left << "Local data (MB): " << std::setprecision(6) << (float(memoryReport.localData) / 1000000) << endl;
			cout << std::setw(40) << std::left << "Global data (MB): " << std::setprecision(6) << (float(memoryReport.globalData) / 1000000) << endl;
			cout << std::setw(40) << std::left << "Parameter matrix (MB): " << std::setprecision(6) << (float(memoryReport.parameterMatrix) / 1000000) << endl;
			cout << std::setw(40) << std::left << "Index vectors (MB): " << std::setprecision(6) << (float(memoryReport.indexVectors) / 1000000) << endl;
			cout << std::setw(40) << std::left << "Residual vector (MB): " << std::setprecision(6) << (float(memoryReport.residualVector) / 1000000) << endl;
			cout << std::setw(40) << std::left << "Jacobian matrix (MB): " << std::setprecision(6) << (float(memoryReport.jacobianMatrix) / 1000000) << endl;
		}
	};


	/**
	 * It stores information about the optimization problem.
	 */
	struct ProblemInfo {
		size_t parameterNum{ 0 };
		size_t residualNum{ 0 };
		bool bEarlyStop{ false };

		static void print(const ProblemInfo& problemInfo) {
			cout << std::setw(20) << std::left << "# parameters: " << std::setprecision(6) << problemInfo.parameterNum << endl;
			cout << std::setw(20) << std::left << "# residuals: " << std::setprecision(6) << problemInfo.residualNum << endl;

			if (problemInfo.bEarlyStop) cout << "Optimization problem was stopped early!" << endl;
		}
	};


	/**
	 * It stores the (last) residual evaluations for one constraint.
	 * The value is the sum of squared residuals for each residual dimension.
	 */
	using ResidualEvaluation = vector<double>;

	/**
	 * It stores the (last) jacobian evaluations for one constraint.
	 * The value is the sum of squared jacobian values for each parameter
	 * and residual dimension.
	 */
	using JacobianEvaluation = vector<double>;

	/**
	 * It summarizes the optimization process. 
	 * It includes information about every optimization iteration and the time statistics for main 
	 * code parts.
	 */
	struct Status {
		vector<IterationInfo> iterationInfo;
		ProblemInfo problemInfo;
		TimeReport timeReport;
		MemoryReport memoryReport;
		vector<ResidualEvaluation> residualEvaluations; // One evaluation vector for each (active!) constraint.
		vector<JacobianEvaluation> jacobianEvaluations; // One evaluation vector for each (active!) constraint.
	};

} // namespace solo