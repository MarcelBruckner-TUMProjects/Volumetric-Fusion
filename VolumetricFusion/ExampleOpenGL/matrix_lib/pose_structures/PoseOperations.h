#pragma once
#include "matrix_lib/matrix_structures/MatrixOperations.h"
#include "PoseType.h"

namespace matrix_lib {

	/**
	 * Rotates the 3D point with rotation in SO3 (axis-angle) notation (i.e.
	 * 3D vector).
	 */
	template<typename R, typename P>
	CPU_AND_GPU auto rotatePointWithSO3(const R& rotation, const P& point) {
		using T = typename BaseType<R, P>::type;
		using RType = typename ResultType<R, P>::type;

		const auto omega = makeTuple(rotation[I<0>()], rotation[I<1>()], rotation[I<2>()]);
		const auto p = makeTuple(point[I<0>()], point[I<1>()], point[I<2>()]);
		const auto theta2 = dot3(omega, omega);

		Tuple<typename TL<RType, RType, RType>::type> rotatedPoint;
		if (theta2 > T(FLT_EPSILON)) {
			// Away from zero, use the Rodrigues' formula
			//
			//   result = point costheta +
			//            (w x point) * sintheta +
			//            w (w . point) (1 - costheta)
			//
			// We want to be careful to only evaluate the square root if the
			// norm of the angle_axis vector is greater than zero. Otherwise
			// we get a division by zero.

			const auto theta = sqrt(theta2);
			const auto cosTheta = cos(theta);
			const auto sinTheta = sin(theta);
			const auto thetaInverse = T(1) / theta;

			const auto w = scale3(omega, thetaInverse);
			const auto wCrossPoint = cross3(w, p);
			const auto tmp = dot3(w, p) * (T(1) - cosTheta);

			auto rotatedPointTuple = add3(
				add3(
					scale3(p, cosTheta),
					scale3(wCrossPoint, sinTheta)
				),
				scale3(w, tmp)
			);

			rotatedPoint[I<0>()] = rotatedPointTuple[I<0>()];
			rotatedPoint[I<1>()] = rotatedPointTuple[I<1>()];
			rotatedPoint[I<2>()] = rotatedPointTuple[I<2>()];
		}
		else {
			// Near zero, the first order Taylor approximation of the rotation
			// matrix R corresponding to a vector w and angle w is
			//
			//   R = I + hat(w) * sin(theta)
			//
			// But sintheta ~ theta and theta * w = angle_axis, which gives us
			//
			//  R = I + hat(w)
			//
			// and actually performing multiplication with the point pt, gives us
			// R * pt = pt + w x pt.
			//
			// Switching to the Taylor expansion near zero provides meaningful
			// derivatives when evaluated using Jets.

			const auto wCrossPoint = cross3(omega, p);
			auto rotatedPointTuple = add3(p, wCrossPoint);

			rotatedPoint[I<0>()] = rotatedPointTuple[I<0>()];
			rotatedPoint[I<1>()] = rotatedPointTuple[I<1>()];
			rotatedPoint[I<2>()] = rotatedPointTuple[I<2>()];
		}

		return rotatedPoint;
	}


	/**
	 * Rotates the 3D point with rotation given as unit quaternion notation (i.e.
	 * 4D vector (real, imag0, imag1, imag2)).
	 */
	template<typename R, typename P>
	CPU_AND_GPU auto rotatePointWithUnitQuaternion(const R& rotation, const P& point) {
		using T = typename BaseType<R, P>::type;

		// p = 0 + px * i + py * j + pz * k
		// pRotated = q * p * qInverse
		const auto p0 = point[I<0>()];
		const auto p1 = point[I<1>()];
		const auto p2 = point[I<2>()];

		const auto real = rotation[I<0>()];
		const auto imag0 = rotation[I<1>()];
		const auto imag1 = rotation[I<2>()];
		const auto imag2 = rotation[I<3>()];

		const auto t2 = real * imag0;
		const auto t3 = real * imag1;
		const auto t4 = real * imag2;
		const auto t5 = -imag0 * imag0;
		const auto t6 = imag0 * imag1;
		const auto t7 = imag0 * imag2;
		const auto t8 = -imag1 * imag1;
		const auto t9 = imag1 * imag2;
		const auto t1 = -imag2 * imag2;

		return makeTuple(
			T(2) * ((t8 + t1) * p0 + (t6 - t4) * p1 + (t3 + t7) * p2) + p0,
			T(2) * ((t4 + t6) * p0 + (t5 + t1) * p1 + (t9 - t2) * p2) + p1,
			T(2) * ((t7 - t3) * p0 + (t2 + t9) * p1 + (t5 + t8) * p2) + p2
		);
	}


	/**
	 * Converts a unit quaternion 4D vector to 3x3 rotation matrix (with row-wise memory storage).
	 */
	template<typename P>
	CPU_AND_GPU auto convertUnitQuaternionToMatrix(const P& q) {
		using T = typename BaseType<P>::type;

		auto real = q[I<0>()];
		auto imag0 = q[I<1>()];
		auto imag1 = q[I<2>()];
		auto imag2 = q[I<3>()];

		return makeTuple(
			real * real + imag0 * imag0 - imag1 * imag1 - imag2 * imag2,
			T(2) * (imag0 * imag1 - real * imag2),
			T(2) * (imag0 * imag2 + real * imag1),
			T(2) * (imag0 * imag1 + real * imag2),
			real * real - imag0 * imag0 + imag1 * imag1 - imag2 * imag2,
			T(2) * (imag1 * imag2 - real * imag0),
			T(2) * (imag0 * imag2 - real * imag1),
			T(2) * (imag1 * imag2 + real * imag0),
			real * real - imag0 * imag0 - imag1 * imag1 + imag2 * imag2
		);
	}


	/**
	 * Converts a 3x3 rotation matrix (with row-wise memory storage) to a unit quaternion 4D vector.
	 */
	template<typename M>
	CPU_AND_GPU auto convertMatrixToUnitQuaternion(const M& m) {
		using T = typename BaseType<M>::type;
		using RType = typename ResultType<M>::type;

		auto m00 = m[I<0>()];	auto m01 = m[I<1>()];	auto m02 = m[I<2>()];
		auto m10 = m[I<3>()];	auto m11 = m[I<4>()];	auto m12 = m[I<5>()];
		auto m20 = m[I<6>()];	auto m21 = m[I<7>()];	auto m22 = m[I<8>()];

		auto tr = m00 + m11 + m22;

		RType qw, qx, qy, qz;
		if (tr > T(0)) {
			auto S = sqrt(tr + T(1.0)) * T(2.0); // S=4*qw 
			qw = T(0.25) * S;
			qx = (m21 - m12) / S;
			qy = (m02 - m20) / S;
			qz = (m10 - m01) / S;
		}
		else if ((m00 > m11) && (m00 > m22)) {
			auto S = sqrt(T(1.0) + m00 - m11 - m22) * T(2.0); // S=4*qx 
			qw = (m21 - m12) / S;
			qx = T(0.25) * S;
			qy = (m01 + m10) / S;
			qz = (m02 + m20) / S;
		}
		else if (m11 > m22) {
			auto S = sqrt(T(1.0) + m11 - m00 - m22) * T(2.0); // S=4*qy
			qw = (m02 - m20) / S;
			qx = (m01 + m10) / S;
			qy = T(0.25) * S;
			qz = (m12 + m21) / S;
		}
		else {
			auto S = sqrt(T(1.0) + m22 - m00 - m11) * T(2.0); // S=4*qz
			qw = (m10 - m01) / S;
			qx = (m02 + m20) / S;
			qy = (m12 + m21) / S;
			qz = T(0.25) * S;
		}

		return makeTuple(qw, qx, qy, qz);
	}


	/**
	 * Converts a unit quaternion 4D vector to an SO3 3D vector (axis-angle).
	 */
	template<typename P>
	CPU_AND_GPU auto convertUnitQuaternionToSO3(const P& q) {
		using T = typename BaseType<P>::type;
		using RType = typename ResultType<P>::type;

		auto qw = q[I<0>()];
		auto qx = q[I<1>()];
		auto qy = q[I<2>()];
		auto qz = q[I<3>()];

		auto sinSquaredTheta = qx * qx + qy * qy + qz * qz;

		// For quaternions representing non-zero rotation, the conversion
		// is numerically stable.
		RType wx, wy, wz;
		if (sinSquaredTheta > T(0)) {
			auto sinTheta = sqrt(sinSquaredTheta);
			auto cosTheta = qw;

			// If cosTheta is negative, theta is greater than pi/2, which
			// means that angle for the angleAxis vector which is 2 * theta
			// would be greater than pi.
			//
			// While this will result in the correct rotation, it does not
			// result in a normalized angle-axis vector.
			//
			// In that case we observe that 2 * theta ~ 2 * theta - 2 * pi,
			// which is equivalent saying
			//
			//   theta - pi = atan(sin(theta - pi), cos(theta - pi))
			//              = atan(-sin(theta), -cos(theta))
			//
			if (cosTheta < T(0)) {
				auto twoTheta = T(2.0) * atan2(-sinTheta, -cosTheta);
				auto k = twoTheta / sinTheta;

				wx = qx * k;
				wy = qy * k;
				wz = qz * k;
			}
			else {
				auto twoTheta = T(2.0) * atan2(sinTheta, cosTheta);
				auto k = twoTheta / sinTheta;

				wx = qx * k;
				wy = qy * k;
				wz = qz * k;
			}
		}
		else {
			// For zero rotation, sqrt() will produce NaN in the derivative since
			// the argument is zero.  By approximating with a Taylor series,
			// and truncating at one term, the value and first derivatives will be
			// computed correctly when Ceres Jets are used.
			const T k(2.0);
			wx = qx * k;
			wy = qy * k;
			wz = qz * k;
		}

		return makeTuple(wx, wy, wz);
	}


	/**
	 * Converts an SO3 3D vector (axis-angle) to a unit quaternion 4D vector.
	 */
	template<typename P>
	CPU_AND_GPU auto convertSO3ToUnitQuaternion(const P& omega) {
		using T = typename BaseType<P>::type;
		using RType = typename ResultType<P>::type;

		auto theta2 = dot3(omega, omega);

		RType qw, qx, qy, qz;
		if (theta2 > T(0.0)) {
			// For points not at the origin, the full conversion is numerically stable.
			auto theta = sqrt(theta2);
			auto halfTheta = theta * T(0.5);
			auto k = sin(halfTheta) / theta;

			qw = cos(halfTheta);
			qx = omega[I<0>()] * k;
			qy = omega[I<1>()] * k;
			qz = omega[I<2>()] * k;
		}
		else {
			// At the origin, sqrt() will produce NaN in the derivative since
			// the argument is zero.  By approximating with a Taylor series,
			// and truncating at one term, the value and first derivatives will be
			// computed correctly when Ceres Jets are used.
			const T k{ 0.5 };

			qw = T(1.0);
			qx = omega[I<0>()] * k;
			qy = omega[I<1>()] * k;
			qz = omega[I<2>()] * k;
		}

		return makeTuple(qw, qx, qy, qz);
	}


	/**
	 * Converts a SO3 3D vector to 3x3 rotation matrix (with row-wise memory storage).
	 */
	template<typename P>
	CPU_AND_GPU auto convertSO3ToMatrix(const P& omega) {
		using T = typename BaseType<P>::type;
		using RType = typename ResultType<P>::type;

		// We use the Rodrigues' formula for exponential map.
		// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

		auto theta2 = dot3(omega, omega);
		auto omegaHat = makeTuple(
			T(0), -omega[I<2>()], omega[I<1>()],
			omega[I<2>()], T(0), -omega[I<0>()],
			-omega[I<1>()], omega[I<0>()], T(0)
		);

		auto id3x3 = makeTuple(
			T(1), T(0), T(0),
			T(0), T(1), T(0),
			T(0), T(0), T(1)
		);

		// We need to be careful to not divide by zero. 
		Tuple<typename AddElements<9, RType, NullType>::type> rotationMatrix;
		if (theta2 > T(FLT_EPSILON)) {
			// Rodrigues' formula.
			const auto theta = sqrt(theta2);
			auto m = add3x3(
				id3x3,
				add3x3(
					scale3x3(omegaHat, (sin(theta) / theta)),
					scale3x3(mm3x3(omegaHat, omegaHat), ((T(1) - cos(theta)) / theta2))
				)
			);
			assign3x3(m, rotationMatrix);
		}
		else {
			// If theta squared is too small, we use only a first order approximation of the exponential formula.
			// R = exp(omega) = I + omega_hat + omega_hat^2 / 2! + ...
			auto m = add3x3(id3x3, omegaHat);
			assign3x3(m, rotationMatrix);
		}

		return rotationMatrix;
	}


	/**
	 * Converts 3x3 rotation matrix (with row-wise memory storage) to an SO3 3D vector.
	 */
	template<typename P>
	CPU_AND_GPU auto convertMatrixToSO3(const P& m) {
		return convertUnitQuaternionToSO3(convertMatrixToUnitQuaternion(m));
	}


	/**
	 * Basic quaternion operations.
	 * The assumed form of quaternion is a 4D vector (w, x, y, z), where w is a real component.
	 */
	template<typename Q0, typename Q1>
	CPU_AND_GPU auto addQuat(const Q0& q0, const Q1& q1) {
		return makeTuple(
			q0[I<0>()] + q1[I<0>()],
			q0[I<1>()] + q1[I<1>()],
			q0[I<2>()] + q1[I<2>()],
			q0[I<3>()] + q1[I<3>()]
		);
	}

	template<typename Q0, typename Q1>
	CPU_AND_GPU auto mulQuat(const Q0& q0, const Q1& q1) {
		return makeTuple(
			q0[I<0>()] * q1[I<0>()] - q0[I<1>()] * q1[I<1>()] - q0[I<2>()] * q1[I<2>()] - q0[I<3>()] * q1[I<3>()],
			q0[I<0>()] * q1[I<1>()] + q0[I<1>()] * q1[I<0>()] + q0[I<2>()] * q1[I<3>()] - q0[I<3>()] * q1[I<2>()],
			q0[I<0>()] * q1[I<2>()] - q0[I<1>()] * q1[I<3>()] + q0[I<2>()] * q1[I<0>()] + q0[I<3>()] * q1[I<1>()],
			q0[I<0>()] * q1[I<3>()] + q0[I<1>()] * q1[I<2>()] - q0[I<2>()] * q1[I<1>()] + q0[I<3>()] * q1[I<0>()]
		);
	}

	template<typename S, typename Q>
	CPU_AND_GPU auto scaleQuat(const S& s, const Q& q) {
		return makeTuple(
			s*q[I<0>()],
			s*q[I<1>()],
			s*q[I<2>()],
			s*q[I<3>()]
		);
	}

	template<typename Q>
	CPU_AND_GPU auto quatNorm(const Q& q) {
		return sqrt(dot4(q, q));
	}

	template<typename Q>
	CPU_AND_GPU auto normalizeQuat(const Q& q) {
		using T = typename BaseType<Q>::type;
		using RType = typename ResultType<Q>::type;

		auto lengthSq = dot4(q, q);

		Tuple<typename TL<RType, RType, RType, RType>::type> normalizedQuat;
		if (lengthSq > T(FLT_EPSILON)) {
			auto lengthInv = T(1) / sqrt(lengthSq);
			normalizedQuat[I<0>()] = lengthInv * q[I<0>()];
			normalizedQuat[I<1>()] = lengthInv * q[I<1>()];
			normalizedQuat[I<2>()] = lengthInv * q[I<2>()];
			normalizedQuat[I<3>()] = lengthInv * q[I<3>()];
		}
		else {
			// Near zero, we just set it to identity by default (stop any gradient propagation).
			normalizedQuat[I<0>()] = T(1);
			normalizedQuat[I<1>()] = T(0);
			normalizedQuat[I<2>()] = T(0);
			normalizedQuat[I<3>()] = T(0);
		}

		return normalizedQuat;
	}

	template<typename Q>
	CPU_AND_GPU auto conjugateQuat(const Q& q) {
		return makeTuple(q[I<0>()], -q[I<1>()], -q[I<2>()], -q[I<3>()]);
	}

	/**
	 * Basic dual quaternion operations.
	 * The assumed form of dual quaternion is an 8D vector with two quaternions, first for rotation
	 * and second for translation.
	 * The assumed form of dual number is a 2D vector.
	 */
	template<typename DQ0, typename DQ1>
	CPU_AND_GPU auto addDualQuat(const DQ0& q0, const DQ1& q1) {
		auto q0_0 = makeTuple(q0[I<0>()], q0[I<1>()], q0[I<2>()], q0[I<3>()]);
		auto q0_1 = makeTuple(q0[I<4>()], q0[I<5>()], q0[I<6>()], q0[I<7>()]);
		auto q1_0 = makeTuple(q1[I<0>()], q1[I<1>()], q1[I<2>()], q1[I<3>()]);
		auto q1_1 = makeTuple(q1[I<4>()], q1[I<5>()], q1[I<6>()], q1[I<7>()]);

		auto quat0 = addQuat(q0_0, q1_0);
		auto quat1 = addQuat(q0_1, q1_1);

		return makeTuple(
			quat0[I<0>()], quat0[I<1>()], quat0[I<2>()], quat0[I<3>()],
			quat1[I<0>()], quat1[I<1>()], quat1[I<2>()], quat1[I<3>()]
		);
	}

	template<typename DQ0, typename DQ1>
	CPU_AND_GPU auto mulDualQuat(const DQ0& q0, const DQ1& q1) {
		auto q0_0 = makeTuple(q0[I<0>()], q0[I<1>()], q0[I<2>()], q0[I<3>()]);
		auto q0_1 = makeTuple(q0[I<4>()], q0[I<5>()], q0[I<6>()], q0[I<7>()]);
		auto q1_0 = makeTuple(q1[I<0>()], q1[I<1>()], q1[I<2>()], q1[I<3>()]);
		auto q1_1 = makeTuple(q1[I<4>()], q1[I<5>()], q1[I<6>()], q1[I<7>()]);

		auto quat0 = mulQuat(q0_0, q1_0);
		auto quat1 = addQuat(
			mulQuat(q0_1, q1_0),
			mulQuat(q0_0, q1_1)
		);

		return makeTuple(
			quat0[I<0>()], quat0[I<1>()], quat0[I<2>()], quat0[I<3>()],
			quat1[I<0>()], quat1[I<1>()], quat1[I<2>()], quat1[I<3>()]
		);
	}

	template<typename DN, typename DQ>
	CPU_AND_GPU auto scaleDualQuat(const DN& dualNumber, const DQ& dualQuaternion) {
		auto s0 = dualNumber[I<0>()];
		auto s1 = dualNumber[I<1>()];
		auto q0 = makeTuple(dualQuaternion[I<0>()], dualQuaternion[I<1>()], dualQuaternion[I<2>()], dualQuaternion[I<3>()]);
		auto q1 = makeTuple(dualQuaternion[I<4>()], dualQuaternion[I<5>()], dualQuaternion[I<6>()], dualQuaternion[I<7>()]);

		auto quat0 = scaleQuat(s0, q0);
		auto quat1 = addQuat(
			scaleQuat(s0, q1),
			scaleQuat(s1, q0)
		);

		return makeTuple(
			quat0[I<0>()], quat0[I<1>()], quat0[I<2>()], quat0[I<3>()],
			quat1[I<0>()], quat1[I<1>()], quat1[I<2>()], quat1[I<3>()]
		);
	}

	template<typename DN>
	CPU_AND_GPU auto reciprocalDualNumber(const DN& dualNumber) {
		auto s0 = dualNumber[I<0>()];
		auto s1 = dualNumber[I<1>()];

		return makeTuple(1.0f / s0, -s1 / (s0*s0));
	}

	template<typename DQ>
	CPU_AND_GPU auto dualQuatNorm(const DQ& dualQuaternion) {
		auto q0 = makeTuple(dualQuaternion[I<0>()], dualQuaternion[I<1>()], dualQuaternion[I<2>()], dualQuaternion[I<3>()]);
		auto q1 = makeTuple(dualQuaternion[I<4>()], dualQuaternion[I<5>()], dualQuaternion[I<6>()], dualQuaternion[I<7>()]);

		// The following dual number is returned.
		auto a0 = quatNorm(q0);
		auto a1 = dot4(q0, q1) / a0;

		return makeTuple(a0, a1);
	}

	template<typename DQ>
	CPU_AND_GPU auto normalizeDualQuat(const DQ& dualQuaternion) {
		return scaleDualQuat(reciprocalDualNumber(dualQuatNorm(dualQuaternion)), dualQuaternion);
	}

	template<typename DQ>
	CPU_AND_GPU auto conjugateDualQuat(const DQ& dualQuaternion) {
		auto q0 = makeTuple(dualQuaternion[I<0>()], dualQuaternion[I<1>()], dualQuaternion[I<2>()], dualQuaternion[I<3>()]);
		auto q1 = makeTuple(dualQuaternion[I<4>()], dualQuaternion[I<5>()], dualQuaternion[I<6>()], dualQuaternion[I<7>()]);

		auto q0Conjugated = conjugateQuat(q0);
		auto q1Conjugated = conjugateQuat(q1);

		return makeTuple(
			q0Conjugated[I<0>()], q0Conjugated[I<1>()], q0Conjugated[I<2>()], q0Conjugated[I<3>()],
			q1Conjugated[I<0>()], q1Conjugated[I<1>()], q1Conjugated[I<2>()], q1Conjugated[I<3>()]
		);
	}

	template<typename R, typename Tr>
	CPU_AND_GPU auto createDualQuat(const R& rotationQuaternion, const Tr& translationVec) {
		using T = typename BaseType<R, Tr>::type;

		auto rotPart = makeTuple(
			rotationQuaternion[I<0>()], rotationQuaternion[I<1>()], rotationQuaternion[I<2>()], rotationQuaternion[I<3>()],
			T(0), T(0), T(0), T(0)
		);
		auto vecPart = makeTuple(
			T(1), T(0), T(0), T(0),
			T(0), T(0.5)*translationVec[I<0>()], T(0.5)*translationVec[I<1>()], T(0.5)*translationVec[I<2>()]
		);

		return mulDualQuat(vecPart, rotPart);
	}

	template<typename DQ>
	CPU_AND_GPU auto convertDualQuatToAffine(const DQ& dualQuaternion) {
		using T = typename BaseType<DQ>::type;

		auto dualQuaternionNormalized = normalizeDualQuat(dualQuaternion);

		auto q0 = makeTuple(dualQuaternionNormalized[I<0>()], dualQuaternionNormalized[I<1>()], dualQuaternionNormalized[I<2>()], dualQuaternionNormalized[I<3>()]);
		auto q1 = makeTuple(dualQuaternionNormalized[I<4>()], dualQuaternionNormalized[I<5>()], dualQuaternionNormalized[I<6>()], dualQuaternionNormalized[I<7>()]);

		auto r00 = T(1) - T(2) * q0[I<2>()] * q0[I<2>()] - T(2) * q0[I<3>()] * q0[I<3>()];
		auto r01 = T(2) * q0[I<1>()] * q0[I<2>()] - T(2) * q0[I<3>()] * q0[I<0>()];
		auto r02 = T(2) * q0[I<1>()] * q0[I<3>()] + T(2) * q0[I<2>()] * q0[I<0>()];
		auto r10 = T(2) * q0[I<1>()] * q0[I<2>()] + T(2) * q0[I<3>()] * q0[I<0>()];
		auto r11 = T(1) - T(2) * q0[I<1>()] * q0[I<1>()] - T(2) * q0[I<3>()] * q0[I<3>()];
		auto r12 = T(2) * q0[I<2>()] * q0[I<3>()] - T(2) * q0[I<1>()] * q0[I<0>()];
		auto r20 = T(2) * q0[I<1>()] * q0[I<3>()] - T(2) * q0[I<2>()] * q0[I<0>()];
		auto r21 = T(2) * q0[I<2>()] * q0[I<3>()] + T(2) * q0[I<1>()] * q0[I<0>()];
		auto r22 = T(1) - T(2) * q0[I<1>()] * q0[I<1>()] - T(2) * q0[I<2>()] * q0[I<2>()];


		auto vecPartQuat = scaleQuat(T(2), mulQuat(q1, conjugateQuat(q0)));
		auto t0 = vecPartQuat[I<1>()];
		auto t1 = vecPartQuat[I<2>()];
		auto t2 = vecPartQuat[I<3>()];

		return makeTuple(
			r00, r01, r02,
			r10, r11, r12,
			r20, r21, r22,
			t0, t1, t2
		);
	}


	/**
	 * Methods for applying rotation with compile-time decision about the rotation type.
	 */
	template<typename R, typename P, unsigned DeformationFlag>
	CPU_AND_GPU auto rotatePoint(const R& rotation, const P& point, Unsigned2Type<DeformationFlag>);

	template<typename R, typename P>
	CPU_AND_GPU auto rotatePoint(const R& rotation, const P& point, Unsigned2Type<PoseType::AFFINE>) {
		return mv3x3(rotation, point);
	}

	template<typename R, typename P>
	CPU_AND_GPU auto rotatePoint(const R& rotation, const P& point, Unsigned2Type<PoseType::SO3wT>) {
		return rotatePointWithSO3(rotation, point);
	}

	template<typename R, typename P>
	CPU_AND_GPU auto rotatePoint(const R& rotation, const P& point, Unsigned2Type<PoseType::QUATERNION>) {
		return rotatePointWithUnitQuaternion(rotation, point);
	}


	/**
	 * Methods for extracting a particular part of the pose (rotation or translation) from the
	 * pose array. They are useful also for making a copy of global data into the local memory.
	 */
	template<typename Pose, unsigned DeformationFlag>
	CPU_AND_GPU auto extractRotation(const Pose& pose, Unsigned2Type<DeformationFlag>);

	template<typename Pose>
	CPU_AND_GPU auto extractRotation(const Pose& pose, Unsigned2Type<PoseType::AFFINE>) {
		return makeTuple(
			pose[I<0>()], pose[I<1>()], pose[I<2>()],
			pose[I<3>()], pose[I<4>()], pose[I<5>()],
			pose[I<6>()], pose[I<7>()], pose[I<8>()]
		);
	}

	template<typename Pose>
	CPU_AND_GPU auto extractRotation(const Pose& pose, Unsigned2Type<PoseType::SO3wT>) {
		return makeTuple(pose[I<0>()], pose[I<1>()], pose[I<2>()]);
	}

	template<typename Pose>
	CPU_AND_GPU auto extractRotation(const Pose& pose, Unsigned2Type<PoseType::QUATERNION>) {
		return makeTuple(pose[I<0>()], pose[I<1>()], pose[I<2>()], pose[I<3>()]);
	}

	template<typename Pose, unsigned DeformationFlag>
	CPU_AND_GPU auto extractTranslation(const Pose& pose, Unsigned2Type<DeformationFlag>);

	template<typename Pose>
	CPU_AND_GPU auto extractTranslation(const Pose& pose, Unsigned2Type<PoseType::AFFINE>) {
		return makeTuple(pose[I<9>()], pose[I<10>()], pose[I<11>()]);
	}

	template<typename Pose>
	CPU_AND_GPU auto extractTranslation(const Pose& pose, Unsigned2Type<PoseType::SO3wT>) {
		return makeTuple(pose[I<3>()], pose[I<4>()], pose[I<5>()]);
	}

	template<typename Pose>
	CPU_AND_GPU auto extractTranslation(const Pose& pose, Unsigned2Type<PoseType::QUATERNION>) {
		return makeTuple(pose[I<4>()], pose[I<5>()], pose[I<6>()]);
	}


	/**
	 * Methods for constructing a complete pose tuple from rotation and translation components.
	 */
	template<typename R, typename T, unsigned DeformationFlag>
	CPU_AND_GPU auto constructPose(const R& rotation, const T& translation, Unsigned2Type<DeformationFlag>);

	template<typename R, typename T>
	CPU_AND_GPU auto constructPose(const R& rotation, const T& translation, Unsigned2Type<PoseType::AFFINE>) {
		return makeTuple(
			rotation[I<0>()], rotation[I<1>()], rotation[I<2>()],
			rotation[I<3>()], rotation[I<4>()], rotation[I<5>()],
			rotation[I<6>()], rotation[I<7>()], rotation[I<8>()],
			translation[I<0>()], translation[I<1>()], translation[I<2>()]
		);
	}

	template<typename R, typename T>
	CPU_AND_GPU auto constructPose(const R& rotation, const T& translation, Unsigned2Type<PoseType::SO3wT>) {
		return makeTuple(
			rotation[I<0>()], rotation[I<1>()], rotation[I<2>()],
			translation[I<0>()], translation[I<1>()], translation[I<2>()]
		);
	}

	template<typename R, typename T>
	CPU_AND_GPU auto constructPose(const R& rotation, const T& translation, Unsigned2Type<PoseType::QUATERNION>) {
		return makeTuple(
			rotation[I<0>()], rotation[I<1>()], rotation[I<2>()], rotation[I<3>()],
			translation[I<0>()], translation[I<1>()], translation[I<2>()]
		);
	}


	/**
	 * Methods for applying rotation with compile-time decision about the rotation type.
	 */
	template<typename RotationVec, unsigned DeformationFlag>
	CPU_AND_GPU auto convertRotationToMatrix(const RotationVec& rotationVec, Unsigned2Type<DeformationFlag>);

	template<typename RotationVec>
	CPU_AND_GPU auto convertRotationToMatrix(const RotationVec& rotationVec, Unsigned2Type<PoseType::AFFINE>) {
		// The rotation vector is already a matrix.
		return rotationVec;
	}

	template<typename RotationVec>
	CPU_AND_GPU auto convertRotationToMatrix(const RotationVec& rotationVec, Unsigned2Type<PoseType::QUATERNION>) {
		return convertUnitQuaternionToMatrix(rotationVec);
	}

	template<typename RotationVec>
	CPU_AND_GPU auto convertRotationToMatrix(const RotationVec& rotationVec, Unsigned2Type<PoseType::SO3wT>) {
		return convertSO3ToMatrix(rotationVec);
	}


	/**
	 * Method for applying pose with compile-time decision about the rotation type.
	 */
	template<typename Pose, typename P, unsigned DeformationFlag>
	CPU_AND_GPU auto transformPoint(const Pose& pose, const P& point, Unsigned2Type<DeformationFlag>) {
		const auto rotation = extractRotation(pose, Unsigned2Type<DeformationFlag>());
		const auto translation = extractTranslation(pose, Unsigned2Type<DeformationFlag>());

		return add3(
			rotatePoint(rotation, point, Unsigned2Type<DeformationFlag>()),
			translation
		);
	}


	/**
	 * Methods for inverting a rotation.
	 */
	template<typename R, unsigned DeformationFlag>
	CPU_AND_GPU auto invertRotation(const R& rotation, Unsigned2Type<DeformationFlag>);

	template<typename R>
	CPU_AND_GPU auto invertRotation(const R& rotation, Unsigned2Type<PoseType::AFFINE>) {
		return invert3x3(rotation);
	}

	template<typename R>
	CPU_AND_GPU auto invertRotation(const R& rotation, Unsigned2Type<PoseType::SO3wT>) {
		return makeTuple(-rotation[I<0>()], -rotation[I<1>()], -rotation[I<2>()]);
	}

	template<typename Q>
	CPU_AND_GPU auto conjugateQuaternion(const Q& q) {
		return makeTuple(q[I<0>()], -q[I<1>()], -q[I<2>()], -q[I<3>()]);
	}

	template<typename R>
	CPU_AND_GPU auto invertRotation(const R& rotation, Unsigned2Type<PoseType::QUATERNION>) {
		return conjugateQuaternion(rotation);
	}


	/**
	 * Methods for inverting a pose.
	 */
	template<typename Pose, unsigned DeformationFlag>
	CPU_AND_GPU auto invertPose(const Pose& pose, Unsigned2Type<DeformationFlag>) {
		using T = typename BaseType<Pose>::type;

		const auto rotation = extractRotation(pose, Unsigned2Type<DeformationFlag>());
		auto rotationInverse = invertRotation(rotation, Unsigned2Type<DeformationFlag>());

		const auto translation = extractTranslation(pose, Unsigned2Type<DeformationFlag>());
		auto translationInverse = scale3(rotatePoint(rotationInverse, translation, Unsigned2Type<DeformationFlag>()), T(-1));

		return constructPose(rotationInverse, translationInverse, Unsigned2Type<DeformationFlag>());
	}

} // namespace matrix_lib 
