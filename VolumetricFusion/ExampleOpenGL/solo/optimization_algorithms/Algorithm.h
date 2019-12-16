#pragma once
#include "Status.h"
#include "solo/data_structures/ParamVector.h"

namespace solo {
	
	template<typename FloatType, typename ParamStorageType, typename ...Constraints>
	class Algorithm {
	public:
		virtual ~Algorithm() = default;

		/**
		 * Executes the optimization algorithm, given all optimization constraints.
		 */
		virtual Status execute(ParamVector<FloatType, ParamStorageType> paramVector, Constraints&&... constraints) = 0;
	};

} // namespace solo