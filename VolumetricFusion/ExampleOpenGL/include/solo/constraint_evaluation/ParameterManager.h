#pragma once
#include "Constraint.h"
#include "ParameterProcessing.h"
#include "solo/data_structures/DenseMatrixWrapper.h"

namespace solo {
	
	/**
	 * ParameterManager takes cares for loading and storing of optimization parameters.
	 */
	class ParameterManager {
	public:
		/**
		 * Updates the optimization parameters with the current increment vector. If we filtered out unused
		 * parameters, we need to use computed index mapping to convert from increment parameters to output
		 * parameters.
		 * We always copy to the memory type that the parameter vector was provided on (i.e. if parameter
		 * vector was provided on device memory, we will update only device memory).
		 */
		template<typename FloatType>
		static void updateParameters(
			DenseMatrix<FloatType>& increment, 
			const vector<int>& indexMapping,
			DenseMatrixWrapper<FloatType>& paramVector
		) {
			runtime_assert(increment.cols() == 1, "The increment vector should have only one column.");

			if (paramVector.getWrapper().wrappedMemoryType() == MemoryType::CPU_MEMORY) {
				// We copy the memory from device to host, if necessary.
#				ifdef COMPILE_CUDA
				auto& incrementContainer = increment.getContainer();
				if (incrementContainer.isUpdatedDevice() && !incrementContainer.isUpdatedHost()) {
					incrementContainer.copyDeviceToHost();
					incrementContainer.setUpdated(true, true);
				}
#				endif

				updateParameterVectorCPU(increment, indexMapping, paramVector);
			}
#			ifdef COMPILE_CUDA
			else if (paramVector.getWrapper().wrappedMemoryType() == MemoryType::CUDA_MEMORY) {
				// We copy the memory from host to device, if necessary.
				auto& incrementContainer = increment.getContainer();
				if (incrementContainer.isUpdatedHost() && !incrementContainer.isUpdatedDevice()) {
					incrementContainer.copyHostToDevice();
					incrementContainer.setUpdated(true, true);
				}

				updateParameterVectorGPU(increment, indexMapping, paramVector);
			}
#			endif
			else {
				runtime_assert(false, "Unsupported input parameter vector storage type. Perhaps you need to define COMPILE_CUDA flag.");
			}		
		}
	};

} // namespace solo