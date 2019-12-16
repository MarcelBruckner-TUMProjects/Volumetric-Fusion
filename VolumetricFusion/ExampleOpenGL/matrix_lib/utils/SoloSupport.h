#pragma once
#include "SoloIncludeCPU.h"
#include "matrix_lib/matrix_structures/Vec2.h"
#include "matrix_lib/matrix_structures/Vec3.h"
#include "matrix_lib/matrix_structures/Vec4.h"
#include "matrix_lib/matrix_structures/VecX.h"
#include "matrix_lib/matrix_structures/Mat2.h"
#include "matrix_lib/matrix_structures/Mat3.h"
#include "matrix_lib/matrix_structures/Mat4.h"
#include "matrix_lib/pose_structures/UnitQuaternion.h"
#include "matrix_lib/pose_structures/SO3.h"
#include "matrix_lib/pose_structures/AffinePose.h"
#include "matrix_lib/pose_structures/AffineIncrement.h"
#include "matrix_lib/pose_structures/RigidPose.h"
#include "matrix_lib/pose_structures/SO3wT.h"
#include "matrix_lib/pose_structures/PoseAliases.h"

// Implementation of Solo methods for BaseDeform classes.
template<typename T>
struct solo::BaseTypeHelper<matrix_lib::Vec2<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::Vec3<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::Vec4<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T, unsigned N>
struct solo::BaseTypeHelper<matrix_lib::VecX<T, N>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::Mat2<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::Mat3<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::Mat4<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::UnitQuaternion<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::SO3<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::RigidPose<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::SO3wT<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::AffinePose<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};

template<typename T>
struct solo::BaseTypeHelper<matrix_lib::AffineIncrement<T>> {
	using type = typename solo::BaseTypeHelper<T>::type;
};


template<typename T>
struct solo::ResultTypeHelper<matrix_lib::Vec2<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::Vec3<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::Vec4<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T, unsigned N>
struct solo::ResultTypeHelper<matrix_lib::VecX<T, N>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::Mat2<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::Mat3<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::Mat4<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::UnitQuaternion<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::SO3<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::RigidPose<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::SO3wT<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::AffinePose<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};

template<typename T>
struct solo::ResultTypeHelper<matrix_lib::AffineIncrement<T>> {
	using type = typename solo::ResultTypeHelper<T>::type;
};