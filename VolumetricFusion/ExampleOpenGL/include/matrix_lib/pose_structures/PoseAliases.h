#pragma once
#include "RigidPose.h"
#include "SO3wT.h"
#include "AffinePose.h"
#include "AffineIncrement.h"

namespace matrix_lib {

	/**
	 * Alias names for different pose types.
	 */
	// Rigid pose.
	template <typename T>
	using RigidPose = Pose<T, PoseType::QUATERNION>;

	typedef RigidPose<float> RigidPosef;
	typedef RigidPose<double> RigidPosed;

	// Affine pose.
	template <typename T>
	using AffinePose = Pose<T, PoseType::AFFINE>;

	typedef AffinePose<float> AffinePosef;
	typedef AffinePose<double> AffinePosed;

	// SO3wT pose increment.
	template <typename T>
	using SO3wT = PoseIncrement<T, PoseType::SO3wT>;

	typedef SO3wT<float> SO3wTf;
	typedef SO3wT<double> SO3wTd;

	// Affine pose increment.
	template <typename T>
	using AffineIncrement = PoseIncrement<T, PoseType::AFFINE>;

	typedef AffineIncrement<float> AffineIncrementf;
	typedef AffineIncrement<double> AffineIncrementd;

} // namespace matrix_lib
