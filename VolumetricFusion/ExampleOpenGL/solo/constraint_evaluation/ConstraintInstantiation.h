#pragma once
#include "Solo/data_structures/MemoryInstantiation.h"

/**
 * Include the right header for constraint evaluation.
 */
#ifdef USE_GPU_EVALUATION
#include "ConstraintProcessingGPU.h"
#else
#include "ConstraintProcessingCPU.h"
#endif

/**
 * Preprocessor helpers for instantiation a Constraint type in CUDA object, enabling to specialize the evaluation
 * method and compile the code with custom cost functions.
 */
#define PREPARE_CONSTRAINT(ConstraintType) \
	template class MemoryInstantiation< ConstraintType >;

#ifdef USE_GPU_EVALUATION
#define COMPILE_CONSTRAINT_GPU(ConstraintType) \
	template void solo::ConstraintProcessingGPU::evaluate<typename solo::ExtractFloatType< ConstraintType >::type, ConstraintType, solo::OnlyResidualEvaluation>(\
		ConstraintType & constraint, \
		solo::DenseMatrixWrapper<typename solo::ExtractFloatType< ConstraintType >::type>& paramVector, \
		solo::DenseMatrixWrapper<int>& indexMatrix, \
		solo::ResidualVector<typename solo::ExtractFloatType< ConstraintType >::type>& residuals, \
		solo::JacobianMatrix<typename solo::ExtractFloatType< ConstraintType >::type>& jacobian, \
		common_utils::Type2Type<solo::OnlyResidualEvaluation>, \
		common_utils::Type2Type<solo::ConstraintEvaluationMode> \
	); \
	template void solo::ConstraintProcessingGPU::evaluate<typename solo::ExtractFloatType< ConstraintType >::type, ConstraintType, solo::ResidualAndJacobianEvaluation>(\
		ConstraintType & constraint, \
		solo::DenseMatrixWrapper<typename solo::ExtractFloatType< ConstraintType >::type>& paramVector, \
		solo::DenseMatrixWrapper<int>& indexMatrix, \
		solo::ResidualVector<typename solo::ExtractFloatType< ConstraintType >::type>& residuals, \
		solo::JacobianMatrix<typename solo::ExtractFloatType< ConstraintType >::type>& jacobian, \
		common_utils::Type2Type<solo::ResidualAndJacobianEvaluation>, \
		common_utils::Type2Type<solo::ConstraintEvaluationMode> \
	); \
	template void solo::ConstraintProcessingGPU::evaluate<typename solo::ExtractFloatType< ConstraintType >::type, ConstraintType, solo::ResidualAndJacobianEvaluation>(\
		ConstraintType & constraint, \
		solo::DenseMatrixWrapper<typename solo::ExtractFloatType< ConstraintType >::type>& paramVector, \
		solo::DenseMatrixWrapper<int>& indexMatrix, \
		solo::ResidualVector<typename solo::ExtractFloatType< ConstraintType >::type>& residuals, \
		solo::JacobianMatrix<typename solo::ExtractFloatType< ConstraintType >::type>& jacobian, \
		common_utils::Type2Type<solo::ResidualAndJacobianEvaluation>, \
		common_utils::Type2Type<solo::GradientCheckMode> \
	);
#else
#define COMPILE_CONSTRAINT_GPU(ConstraintType)
#endif

#ifdef USE_GPU_EVALUATION
#define COMPILE_CONSTRAINT_CPU(ConstraintType)
#else
#define COMPILE_CONSTRAINT_CPU(ConstraintType) \
	template void solo::ConstraintProcessingCPU::evaluate<typename solo::ExtractFloatType< ConstraintType >::type, ConstraintType, solo::OnlyResidualEvaluation>(\
		ConstraintType & constraint, \
		solo::DenseMatrixWrapper<typename solo::ExtractFloatType< ConstraintType >::type>& paramVector, \
		solo::DenseMatrixWrapper<int>& indexMatrix, \
		solo::ResidualVector<typename solo::ExtractFloatType< ConstraintType >::type>& residuals, \
		solo::JacobianMatrix<typename solo::ExtractFloatType< ConstraintType >::type>& jacobian, \
		common_utils::Type2Type<solo::OnlyResidualEvaluation>, \
		common_utils::Type2Type<solo::ConstraintEvaluationMode> \
	); \
	template void solo::ConstraintProcessingCPU::evaluate<typename solo::ExtractFloatType< ConstraintType >::type, ConstraintType, solo::ResidualAndJacobianEvaluation>(\
		ConstraintType & constraint, \
		solo::DenseMatrixWrapper<typename solo::ExtractFloatType< ConstraintType >::type>& paramVector, \
		solo::DenseMatrixWrapper<int>& indexMatrix, \
		solo::ResidualVector<typename solo::ExtractFloatType< ConstraintType >::type>& residuals, \
		solo::JacobianMatrix<typename solo::ExtractFloatType< ConstraintType >::type>& jacobian, \
		common_utils::Type2Type<solo::ResidualAndJacobianEvaluation>, \
		common_utils::Type2Type<solo::ConstraintEvaluationMode> \
	); \
	template void solo::ConstraintProcessingCPU::evaluate<typename solo::ExtractFloatType< ConstraintType >::type, ConstraintType, solo::ResidualAndJacobianEvaluation>(\
		ConstraintType & constraint, \
		solo::DenseMatrixWrapper<typename solo::ExtractFloatType< ConstraintType >::type>& paramVector, \
		solo::DenseMatrixWrapper<int>& indexMatrix, \
		solo::ResidualVector<typename solo::ExtractFloatType< ConstraintType >::type>& residuals, \
		solo::JacobianMatrix<typename solo::ExtractFloatType< ConstraintType >::type>& jacobian, \
		common_utils::Type2Type<solo::ResidualAndJacobianEvaluation>, \
		common_utils::Type2Type<solo::GradientCheckMode> \
	);
#endif


#define INSTANTIATE_CONSTRAINT_CPU(ConstraintType) \
	COMPILE_CONSTRAINT_CPU( ConstraintType )

#define INSTANTIATE_CONSTRAINT_GPU(ConstraintType) \
	PREPARE_CONSTRAINT( ConstraintType ) \
	COMPILE_CONSTRAINT_GPU( ConstraintType )