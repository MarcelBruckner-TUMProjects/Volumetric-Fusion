#pragma once
#include <common_utils/timing/TimerCPU.h>

#include "Algorithm.h"
#include "Settings.h"
#include "IndexComputation.h"
#include "SolverProcessing.h"
#include "Solo/constraint_evaluation/ConstraintEvaluator.h"
#include "Solo/linear_solvers/LinearSolver.h"
#include "Solo/linear_solvers/LossComputation.h"
#include "Solo/constraint_evaluation/ParameterManager.h"

namespace solo {

	template<typename FloatType, typename ParamStorageType, typename ...Constraints>
	class GaussNewtonAlgorithm : public Algorithm<FloatType, ParamStorageType, Constraints...> {
	public:
		GaussNewtonAlgorithm(const Settings& settings, LinearSolver<FloatType>& linearSolver) : 
			m_settings{ settings },
			m_linearSolver{ linearSolver }
		{ }

		/**
		 * Interface implementation.
		 */
		Status execute(ParamVector<FloatType, ParamStorageType> paramVector, Constraints&&... constraints) override {
			common_utils::TimerCPU timerTotal;
			common_utils::TimerCPU timerLocal;

			Status status;

			// Compute index vector.
			timerLocal.restart();
			DenseMatrix<int> columnIndexVector, rowIndexVector;
			vector<int> indexMapping;
			unsigned nParameters{ 0 };
			
			if (m_linearSolver.getType() == LinearSolverType::SPARSE_PCG_ATOMIC_GPU)
				nParameters = IndexComputation::computeIndexVectors(
					Type2Type<ColumnWiseStorage>(), m_settings.bCheckParamUsage, m_settings.bUseGPUForIndexVectorComputation, 
					columnIndexVector, rowIndexVector, indexMapping, std::forward<Constraints>(constraints)...
				);
			else
				nParameters = IndexComputation::computeIndexVectors(
					Type2Type<RowWiseStorage>(), m_settings.bCheckParamUsage, m_settings.bUseGPUForIndexVectorComputation, 
					columnIndexVector, rowIndexVector, indexMapping, std::forward<Constraints>(constraints)...
				);
			
			status.timeReport.indexVectorComputation = timerLocal.getElapsedTime();

			// Initialize a wrapper around parameter vector.
			DenseMatrixWrapper<FloatType> paramVectorWrapper{ paramVector.getData(), paramVector.getSize(), 1, Type2Type<ParamStorageType>() };

			// Execute iterative optimization.
			double avgTimeConstraintEvaluation{ 0.0 };
			double avgTimeLinearSolver{ 0.0 };
			double avgTimeParameterUpdate{ 0.0 };
			double avgTimeResidualEvaluation{ 0.0 };

			// Create the residual vectors and Jacobian matrix placeholders.
			const unsigned nUsedConstraints = solver_proc::computeNumUsedConstraints(std::forward<Constraints>(constraints)...);
			if (nUsedConstraints == 0) {
				cout << "No constraints are used in the optimization, therefore no optimization solving is run." << endl;
				return status;
			}

			vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>> residualsAndJacobians(nUsedConstraints);
			vector<ResidualVector<FloatType>> residualsAfter(nUsedConstraints);

			// Check for early stop of the algorithm, if requested.
			TerminationCriteria terminationCriteria;

			unsigned nIteration = 0;
			while (nIteration < m_settings.algorithmMaxNumIterations) {
				// Evaluate residuals and Jacobian for each constraint.
				timerLocal.restart();
				evaluateConstraints(paramVectorWrapper, 0, residualsAndJacobians, std::forward<Constraints>(constraints)...);
				avgTimeConstraintEvaluation += timerLocal.getElapsedTime();

				// Run the system solver to obtain an update vector.
				timerLocal.restart();
				int nSolverIterations{ 0 };
				DenseMatrix<FloatType> increment;
				FloatType squaredLoss = m_linearSolver.solve(
					residualsAndJacobians, columnIndexVector, rowIndexVector, nParameters, FloatType(m_settings.initialLMWeight), terminationCriteria, nSolverIterations, increment
				);
				avgTimeLinearSolver += timerLocal.getElapsedTime();

				// Update the parameter values.
				timerLocal.restart();
				ParameterManager::updateParameters(increment, indexMapping, paramVectorWrapper);
				avgTimeParameterUpdate += timerLocal.getElapsedTime();

				// Compute the final residuals (to check the function termination criteria).
				timerLocal.restart();
				evaluateResiduals(paramVectorWrapper, 0, residualsAfter, std::forward<Constraints>(constraints)...);
				FloatType squaredLossAfter = LossComputation::compute(residualsAfter);
				avgTimeResidualEvaluation += timerLocal.getElapsedTime();

				// Check the function tolerance termination criteria.
				FloatType lossChange(-1.0);
				if (isFunctionToleranceReached(squaredLoss, squaredLossAfter, m_settings.functionTolerance, lossChange) && m_settings.bEnableAlgorithmEarlyStop) {
					terminationCriteria.bStopEarly = true;
				}

				// Add the current iteration info.
				status.iterationInfo.emplace_back(IterationInfo{ 
					int(nIteration), nSolverIterations, std::sqrt(double(squaredLoss)), lossChange, terminationCriteria.maxGradientNorm
				});
				nIteration++;

				// When we are in the last iteration, we store the residuals to the status, if requested.
				if (m_settings.bEvaluateResiduals && (nIteration >= m_settings.algorithmMaxNumIterations || terminationCriteria.bStopEarly)) {
					storeEvaluatedResiduals(residualsAndJacobians, status);
				}

				// When we are in the last iteration, we store the jacobians to the status, if requested.
				if (m_settings.bEvaluateJacobians && (nIteration >= m_settings.algorithmMaxNumIterations || terminationCriteria.bStopEarly)) {
					storeEvaluatedJacobians(residualsAndJacobians, status);
				}

				if (terminationCriteria.bStopEarly) break;
			}

			if (nIteration > 0) {
				avgTimeConstraintEvaluation /= nIteration;
				avgTimeLinearSolver /= nIteration;
				avgTimeParameterUpdate /= nIteration;
				avgTimeResidualEvaluation /= nIteration;
			}

			// Store time report.
			status.timeReport.constraintEvaluation = avgTimeConstraintEvaluation;
			status.timeReport.linearSolver = avgTimeLinearSolver;
			status.timeReport.parameterUpdate = avgTimeParameterUpdate;
			status.timeReport.residualEvaluation = avgTimeResidualEvaluation;
			status.timeReport.totalTime = timerTotal.getElapsedTime();

			// Compute memory report and store system information.
			computeMemoryReportAndProblemInfo(status.memoryReport, status.problemInfo, std::forward<Constraints>(constraints)...);
			status.memoryReport.indexVectors = columnIndexVector.getByteSize() + rowIndexVector.getByteSize();
			status.problemInfo.parameterNum = nParameters;
			status.problemInfo.bEarlyStop = terminationCriteria.bStopEarly;

			return status;
		}

	private:
		const Settings& m_settings;
		LinearSolver<FloatType>& m_linearSolver;

		/**
		 * Helper method for recursively evaluating constraints.
		 */
		template<typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks, typename ...OtherConstraints>
		static void evaluateConstraints(
			DenseMatrixWrapper<FloatType>& paramVector,
			unsigned constraintIdx, 
			vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& residualsAndJacobians, 
			Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>& constraint, 
			OtherConstraints&&... otherConstraints
		) {
			using CostFunctionSignature = typename Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>::CostFunctionInterfaceSignature;
			
			unsigned nextConstraintIdx = constraintIdx;

			if (constraint.isUsedInOptimization()) {
				TIME_CPU_START(GaussNewtonAlgorithm_evaluateConstraint);

				// Evaluate the residual vector and Jacobian matrix.
				auto& resultPair = residualsAndJacobians[constraintIdx];
				ConstraintEvaluator::computeResidualsAndJacobian(constraint, paramVector, resultPair.first, resultPair.second);

				TIME_CPU_STOP(GaussNewtonAlgorithm_evaluateConstraint);

				nextConstraintIdx += 1;
			}

			// Recursively evaluate all the other constraints.
			evaluateConstraints(paramVector, nextConstraintIdx, residualsAndJacobians, std::forward<OtherConstraints>(otherConstraints)...);
		}

		static void evaluateConstraints(
			DenseMatrixWrapper<FloatType>& paramVector,
			unsigned constraintIdx, 
			vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& residualsAndJacobian
		) { }

		/**
		 * Helper method for recursively evaluating residuals.
		 */
		template<typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks, typename ...OtherConstraints>
		static void evaluateResiduals(
			DenseMatrixWrapper<FloatType>& paramVector,
			unsigned constraintIdx,
			vector<ResidualVector<FloatType>>& residuals,
			Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>& constraint,
			OtherConstraints&&... otherConstraints
		) {
			using CostFunctionSignature = typename Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>::CostFunctionInterfaceSignature;

			unsigned nextConstraintIdx = constraintIdx;

			if (constraint.isUsedInOptimization()) {
				TIME_CPU_START(GaussNewtonAlgorithm_evaluateConstraint);

				// Evaluate the residual vector and Jacobian matrix.
				auto& residualsPerConstraint = residuals[constraintIdx];
				ConstraintEvaluator::computeResiduals(constraint, paramVector, residualsPerConstraint);

				TIME_CPU_STOP(GaussNewtonAlgorithm_evaluateConstraint);

				nextConstraintIdx += 1;
			}

			// Recursively evaluate all the other constraints.
			evaluateResiduals(paramVector, nextConstraintIdx, residuals, std::forward<OtherConstraints>(otherConstraints)...);
		}

		static void evaluateResiduals(
			DenseMatrixWrapper<FloatType>& paramVector,
			unsigned constraintIdx,
			vector<ResidualVector<FloatType>>& residuals
		) { }

		/**
		 * Stores the constraint residuals in the Status object that is returned to the user.
		 */
		void storeEvaluatedResiduals(vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& residualsAndJacobians, Status& status) {
			// Copy the residuals to return them to the user.
			const unsigned nResiduals = residualsAndJacobians.size();
			if (nResiduals == 0) return;

			status.residualEvaluations.reserve(nResiduals);

			for (auto&& residualsAndJacobian : residualsAndJacobians) {
				auto& residualsOfConstraint = residualsAndJacobian.first;

				// If the residuals are in the device (GPU) memory, we need to copy them to the host first.
#				ifdef COMPILE_CUDA
				auto& residualsContainer = residualsOfConstraint.getContainer();
				if (residualsContainer.isUpdatedDevice() && !residualsContainer.isUpdatedHost()) {
					residualsContainer.copyDeviceToHost();
					residualsContainer.setUpdated(true, true);
				}
#				endif

				const int nResidualsOfConstraint = residualsOfConstraint.getNumResiduals();
				const int residualDim = residualsOfConstraint.getResidualDim();
				vector<double> copiedResiduals(nResidualsOfConstraint);

				ResidualVectorInterface<FloatType, MemoryTypeCPU> iResidualsOfConstraint{ residualsOfConstraint };

#				pragma omp parallel for
				for (int i = 0; i < nResidualsOfConstraint; ++i) {
					// If the residual has a dimension > 1, we just store the sum of squares in the 
					// evaluated residual.
					double residualNormSquared{ 0.f };
					for (int j = 0; j < residualDim; ++j) {
						residualNormSquared += std::pow(double(iResidualsOfConstraint(i, j)), 2.0);
					}
					copiedResiduals[i] = residualNormSquared;
				}

				status.residualEvaluations.emplace_back(std::move(copiedResiduals));
			}
		}

		/**
		 * Stores the constraint jacobians in the Status object that is returned to the user.
		 */
		void storeEvaluatedJacobians(vector<std::pair<ResidualVector<FloatType>, JacobianMatrix<FloatType>>>& residualsAndJacobians, Status& status) {
			// Copy the jacobians to return them to the user.
			const unsigned nJacobians = static_cast<unsigned>(residualsAndJacobians.size());
			if (nJacobians == 0) return;

			status.jacobianEvaluations.reserve(nJacobians);

			for (auto&& residualsAndJacobian : residualsAndJacobians) {
				auto& jacobiansOfConstraint = residualsAndJacobian.second;

				// If the residuals are in the device (GPU) memory, we need to copy them to the host first.
#				ifdef COMPILE_CUDA
				auto& jacobiansContainer = jacobiansOfConstraint.getContainer();
				if (jacobiansContainer.isUpdatedDevice() && !jacobiansContainer.isUpdatedHost()) {
					jacobiansContainer.copyDeviceToHost();
					jacobiansContainer.setUpdated(true, true);
				}
#				endif

				const unsigned nJacobiansOfConstraint = jacobiansOfConstraint.getNumResiduals();
				const unsigned residualDim = jacobiansOfConstraint.getResidualDim();
				const unsigned paramDim = jacobiansOfConstraint.getParamDim();
				vector<double> copiedJacobians(nJacobiansOfConstraint);

				JacobianMatrixInterface<FloatType, MemoryTypeCPU> iJacobiansOfConstraint{ jacobiansOfConstraint };

#				pragma omp parallel for
				for (int i = 0; i < nJacobiansOfConstraint; ++i) {
					// We just store the sum of squares in the evaluated jacobian (for each residual and parameter
					// dimension).
					double jacobianNormSquared{ 0.f };
					for (int residualId = 0; residualId < residualDim; ++residualId) {
						for (int paramId = 0; paramId < paramDim; paramId++) {
							jacobianNormSquared += std::pow(double(iJacobiansOfConstraint(i, residualId, paramId)), 2.0);
						}
					}
					copiedJacobians[i] = jacobianNormSquared;
				}

				status.jacobianEvaluations.emplace_back(std::move(copiedJacobians));
			}
		}

		/**
		 * It recursively computes memory sizes of optimization structures.
		 */
		template<typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks, typename ...OtherConstraints>
		static void computeMemoryReportAndProblemInfo(
			MemoryReport& memoryReport, ProblemInfo& problemInfo, 
			Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>& constraint, OtherConstraints&&... otherConstraints
		) {
			using CostFunctionSignature = typename Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>::CostFunctionInterfaceSignature;

			if (constraint.isUsedInOptimization()) {
				memoryReport.localData += constraint.getLocalDataByteSize();
				memoryReport.globalData += constraint.getGlobalDataByteSize();

				const unsigned nResiduals = constraint.getNumResiduals();
				memoryReport.parameterMatrix += nResiduals * GetTotalParamDim<CostFunctionSignature>::value * sizeof(FloatType);
				memoryReport.residualVector += nResiduals * GetResidualDim<CostFunctionSignature>::value * sizeof(FloatType);
				memoryReport.jacobianMatrix += nResiduals * GetResidualDim<CostFunctionSignature>::value * GetTotalParamDim<CostFunctionSignature>::value * sizeof(FloatType);

				problemInfo.residualNum += nResiduals * GetResidualDim<CostFunctionSignature>::value;
			}

			computeMemoryReportAndProblemInfo(memoryReport, problemInfo, std::forward<OtherConstraints>(otherConstraints)...);
		}

		static void computeMemoryReportAndProblemInfo(MemoryReport& memoryReport, ProblemInfo& problemInfo) { }

		/**
		 * Checks function tolerance termination criteria.
		 */
		static bool isFunctionToleranceReached(FloatType squaredLossPrevious, FloatType squaredLossCurrent, double functionTolerance, FloatType& lossChange) {
			FloatType lossPrevious = squaredLossPrevious;//std::sqrt(squaredLossPrevious);
			FloatType lossCurrent = squaredLossCurrent;//std::sqrt(squaredLossCurrent);
			
			/// Relative error.
			//lossChange = lossPrevious - lossCurrent;
			//
			//const double threshold = functionTolerance * lossPrevious;

			//if (std::abs(lossChange) > threshold) {
			//	return false;
			//}
			//else {
			//	return true;
			//}

			/// Absolute error.
			//lossChange = lossPrevious - lossCurrent;

			//const double threshold = functionTolerance;

			//if (std::abs(lossChange) > threshold) {
			//	return false;
			//}
			//else {
			//	return true;
			//}

			/// Combination of relative and absolute error.
			lossChange = lossPrevious - lossCurrent;

			const double threshold = functionTolerance * (lossPrevious + 1.0);

			if (std::abs(lossChange) > threshold) {
				return false;
			}
			else {
				return true;
			}
		}
	};

} // namespace solo