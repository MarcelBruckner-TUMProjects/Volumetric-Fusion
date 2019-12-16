#pragma once
#include "PoseAliases.h"
#include "PoseOperations.h"

namespace matrix_lib {

	/**
	 * Pose-to-pose multiplication.
	 */
	template<typename T>
	CPU_AND_GPU RigidPose<T> operator*(RigidPose<T> lhs, const RigidPose<T>& rhs) {
		lhs *= rhs;
		return lhs;
	}

	template<typename T>
	CPU_AND_GPU AffinePose<T> operator*(AffinePose<T> lhs, const AffinePose<T>& rhs) {
		lhs *= rhs;
		return lhs;
	}

} // namespace matrix_lib
