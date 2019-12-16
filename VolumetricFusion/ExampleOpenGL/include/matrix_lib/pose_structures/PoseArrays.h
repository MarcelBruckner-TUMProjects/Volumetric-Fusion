#pragma once
#include "matrix_lib/utils/LibIncludeCPU.h"
#include "matrix_lib/matrix_structures/VecArrays.h"

namespace matrix_lib {
	
	/**
	 * SO3wT pose (SO3 rotation + translation).
	 */
	template<bool isInterface>
	struct ASO3wTf : public SoA<
		isInterface,
		typename TL<
			float4,	// Axis-angle (SO3) rotation
			float4	// Translation
		>::type
	> {
		CPU_AND_GPU float4* rotation() { return t()[I<0>()]; }
		CPU_AND_GPU const float4* rotation() const { return t()[I<0>()]; }
		CPU_AND_GPU float4* translation() { return t()[I<1>()]; }
		CPU_AND_GPU const float4* translation() const { return t()[I<1>()]; }
	};

	using iASO3wTf = ASO3wTf<true>;
	using sASO3wTf = ASO3wTf<false>;


	/**
	 * Quaternion pose (unit quaternion rotation + translation).
	 */
	template<bool isInterface>
	struct AQuaternionPosef : public SoA<
		isInterface,
		typename TL<
			float4,	// Unit quaternion rotation
			float4	// Translation
		>::type
	> {
		CPU_AND_GPU float4* rotation() { return t()[I<0>()]; }
		CPU_AND_GPU const float4* rotation() const { return t()[I<0>()]; }
		CPU_AND_GPU float4* translation() { return t()[I<1>()]; }
		CPU_AND_GPU const float4* translation() const { return t()[I<1>()]; }
	};

	using iAQuaternionPosef = AQuaternionPosef<true>;
	using sAQuaternionPosef = AQuaternionPosef<false>;


	/**
	 * Affine pose (3x3 affine matrix + translation).
	 */
	template<bool isInterface>
	struct AAffinePosef : public SoA<
		isInterface,
		typename TL<
			AMat3f<isInterface>,	// Affine 3x3 matrix
			float4					// Translation
		>::type
	> {
		CPU_AND_GPU AMat3f<isInterface>& rotation() { return t()[I<0>()]; }
		CPU_AND_GPU const AMat3f<isInterface>& rotation() const { return t()[I<0>()]; }
		CPU_AND_GPU float4* translation() { return t()[I<1>()]; }
		CPU_AND_GPU const float4* translation() const { return t()[I<1>()]; }
	};

	using iAAffinePosef = AAffinePosef<true>;
	using sAAffinePosef = AAffinePosef<false>;

} // namespace matrix_lib
