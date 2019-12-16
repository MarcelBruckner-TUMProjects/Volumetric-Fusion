#pragma once
#include "Solo/utils/Common.h"
#include "Solo/constraint_evaluation/Constraint.h"

namespace solo {
	
	/**
	 * Takes care for explicit instantiation of memory types that need to be compiled with CUDA compiler.
	 * In order to compile the framework while using GPU evaluation of constraints, some parts of it need
	 * to be compiled with CUDA compiler. One part are all the memory data structures present in the GPU
	 * evaluation code.
	 */
	template<typename ConstraintType>
	class MemoryInstantiation {
	public:
		using FloatType = typename ExtractFloatType<ConstraintType>::type;
		using LocalData = typename ExtractLocalData<ConstraintType>::type;
		using GlobalData = typename ExtractGlobalData<ConstraintType>::type;

		void instantiate() const {
			LocalData	localData;
			GlobalData	globalData;
 		}
	};

} // namespace solo