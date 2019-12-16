#pragma once
#include "PoseProcessing.h"

namespace matrix_lib {

	/**
	 * Dual quaternions are a 8-parameter representation of rotation and translation.
	 */
	template<typename FloatType>
	struct Quaternion {
		CPU_AND_GPU Quaternion() {}
		CPU_AND_GPU Quaternion(FloatType _w, FloatType _x, FloatType _y, FloatType _z) : q0(_x, _y, _z, _w) {}
		CPU_AND_GPU Quaternion(const Vec4<FloatType>& _q) : q0(_q) {}
		CPU_AND_GPU Quaternion(const Mat3<FloatType>& _rot) {
			FloatType m00 = _rot(0, 0);	FloatType m01 = _rot(0, 1);	FloatType m02 = _rot(0, 2);
			FloatType m10 = _rot(1, 0);	FloatType m11 = _rot(1, 1);	FloatType m12 = _rot(1, 2);
			FloatType m20 = _rot(2, 0);	FloatType m21 = _rot(2, 1);	FloatType m22 = _rot(2, 2);

			FloatType tr = m00 + m11 + m22;

			FloatType qw, qx, qy, qz;
			if (tr > 0) {
				FloatType S = sqrt(tr + (FloatType)1.0) * 2; // S=4*qw 
				qw = (FloatType)0.25 * S;
				qx = (m21 - m12) / S;
				qy = (m02 - m20) / S;
				qz = (m10 - m01) / S;
			}
			else if ((m00 > m11)&(m00 > m22)) {
				FloatType S = sqrt((FloatType)1.0 + m00 - m11 - m22) * (FloatType)2; // S=4*qx 
				qw = (m21 - m12) / S;
				qx = (FloatType)0.25 * S;
				qy = (m01 + m10) / S;
				qz = (m02 + m20) / S;
			}
			else if (m11 > m22) {
				FloatType S = sqrt((FloatType)1.0 + m11 - m00 - m22) * (FloatType)2; // S=4*qy
				qw = (m02 - m20) / S;
				qx = (m01 + m10) / S;
				qy = (FloatType)0.25 * S;
				qz = (m12 + m21) / S;
			}
			else {
				FloatType S = sqrt((FloatType)1.0 + m22 - m00 - m11) * (FloatType)2; // S=4*qz
				qw = (m10 - m01) / S;
				qx = (m02 + m20) / S;
				qy = (m12 + m21) / S;
				qz = (FloatType)0.25 * S;
			}

			q0 = Vec4<FloatType>(qx, qy, qz, qw);
		}

		CPU_AND_GPU FloatType& x() { return q0.x(); }
		CPU_AND_GPU FloatType& y() { return q0.y(); }
		CPU_AND_GPU FloatType& z() { return q0.z(); }
		CPU_AND_GPU FloatType& w() { return q0.w(); }

		CPU_AND_GPU const FloatType& x() const { return q0.x(); }
		CPU_AND_GPU const FloatType& y() const { return q0.y(); }
		CPU_AND_GPU const FloatType& z() const { return q0.z(); }
		CPU_AND_GPU const FloatType& w() const { return q0.w(); }

		CPU_AND_GPU Quaternion<FloatType> conjugate() const { return Quaternion<FloatType>(q0.w(), -q0.x(), -q0.y(), -q0.z()); }
		CPU_AND_GPU FloatType square_norm() const { return q0.w()*q0.w() + q0.x()*q0.x() + q0.y()*q0.y() + q0.z()*q0.z(); }
		CPU_AND_GPU FloatType norm() const { return std::sqrt(square_norm()); }
		CPU_AND_GPU FloatType dot(const Quaternion<FloatType>& _quat) const { return q0.w()*_quat.w() + q0.x()*_quat.x() + q0.y()*_quat.y() + q0.z()*_quat.z(); }
		CPU_AND_GPU void normalize() { q0 = q0.getNormalized(); }
		CPU_AND_GPU Quaternion<FloatType> normalized() const { Quaternion<FloatType> q(*this); q.normalize(); return q; }

		CPU_AND_GPU Mat3<FloatType> matrix() const {
			// Normalize quaternion before converting to so3 matrix.
			Quaternion q(*this);
			q.normalize();

			Mat3<FloatType> rot;
			rot(0, 0) = 1 - 2 * q.y()*q.y() - 2 * q.z()*q.z();
			rot(0, 1) = 2 * q.x()*q.y() - 2 * q.z()*q.w();
			rot(0, 2) = 2 * q.x()*q.z() + 2 * q.y()*q.w();
			rot(1, 0) = 2 * q.x()*q.y() + 2 * q.z()*q.w();
			rot(1, 1) = 1 - 2 * q.x()*q.x() - 2 * q.z()*q.z();
			rot(1, 2) = 2 * q.y()*q.z() - 2 * q.x()*q.w();
			rot(2, 0) = 2 * q.x()*q.z() - 2 * q.y()*q.w();
			rot(2, 1) = 2 * q.y()*q.z() + 2 * q.x()*q.w();
			rot(2, 2) = 1 - 2 * q.x()*q.x() - 2 * q.y()*q.y();
			return rot;
		}

		CPU_AND_GPU Vec3<FloatType> vec() const { return Vec3<FloatType>(q0.x(), q0.y(), q0.z()); }

		Vec4<FloatType> q0;
	};

	template<typename FloatType>
	CPU_AND_GPU Quaternion<FloatType> operator+(const Quaternion<FloatType>& _left, const Quaternion<FloatType>& _right) {
		return{ _left.w() + _right.w(), _left.x() + _right.x(), _left.y() + _right.y(), _left.z() + _right.z() };
	}

	template<typename FloatType>
	CPU_AND_GPU Quaternion<FloatType> operator*(float _scalar, const Quaternion<FloatType>& _quat) {
		return{ _scalar*_quat.w(), _scalar*_quat.x(), _scalar*_quat.y(), _scalar*_quat.z() };
	}

	template<typename FloatType>
	CPU_AND_GPU Quaternion<FloatType> operator*(const Quaternion<FloatType>& _quat, float _scalar) {
		return _scalar * _quat;
	}

	template<typename FloatType>
	CPU_AND_GPU Quaternion<FloatType> operator*(const Quaternion<FloatType>& _q0, const Quaternion<FloatType>& _q1) {
		Quaternion<FloatType> q;
		q.w() = _q0.w()*_q1.w() - _q0.x()*_q1.x() - _q0.y()*_q1.y() - _q0.z()*_q1.z();
		q.x() = _q0.w()*_q1.x() + _q0.x()*_q1.w() + _q0.y()*_q1.z() - _q0.z()*_q1.y();
		q.y() = _q0.w()*_q1.y() - _q0.x()*_q1.z() + _q0.y()*_q1.w() + _q0.z()*_q1.x();
		q.z() = _q0.w()*_q1.z() + _q0.x()*_q1.y() - _q0.y()*_q1.x() + _q0.z()*_q1.w();

		return q;
	}

	template<typename FloatType>
	struct DualNumber {
		CPU_AND_GPU DualNumber() : q0(0), q1(0) {}
		CPU_AND_GPU DualNumber(FloatType _q0, FloatType _q1) : q0(_q0), q1(_q1) {}

		CPU_AND_GPU DualNumber<FloatType> operator+(const DualNumber<FloatType>& _dn) const {
			return{ q0 + _dn.q0, q1 + _dn.q1 };
		}

		CPU_AND_GPU DualNumber<FloatType>& operator+=(const DualNumber<FloatType>& _dn) {
			*this = *this + _dn;
			return *this;
		}

		CPU_AND_GPU DualNumber<FloatType> operator*(const DualNumber<FloatType>& _dn) const {
			return{ q0*_dn.q0, q0*_dn.q1 + q1 * _dn.q0 };
		}

		CPU_AND_GPU DualNumber<FloatType>& operator*=(const DualNumber<FloatType>& _dn) {
			*this = *this * _dn;
			return *this;
		}

		CPU_AND_GPU DualNumber<FloatType> reciprocal() const {
			return{ 1.0f / q0, -q1 / (q0*q0) };
		}

		CPU_AND_GPU DualNumber<FloatType> sqrt() const {
			return{ std::sqrt(q0), q1 / (2 * std::sqrt(q0)) };
		}

		FloatType q0, q1;
	};

	// Forward declaration
	template<typename FloatType>
	struct DualQuaternion;

	template<typename FloatType>
	CPU_AND_GPU DualQuaternion<FloatType> operator*(const DualNumber<FloatType>& _dn, const DualQuaternion<FloatType>& _dq);

	template<typename FloatType>
	struct DualQuaternion {
		CPU_AND_GPU DualQuaternion() {}
		CPU_AND_GPU DualQuaternion(const Quaternion<FloatType>&_q0, const Quaternion<FloatType>&_q1) : q0(_q0), q1(_q1) {}
		CPU_AND_GPU DualQuaternion(const Mat3<FloatType>& r, const Vec3<FloatType>& t) {
			DualQuaternion<FloatType> rot_part(Quaternion<FloatType>(r), Quaternion<FloatType>(0, 0, 0, 0));
			DualQuaternion<FloatType> vec_part(Quaternion<FloatType>(1, 0, 0, 0), Quaternion<FloatType>(0, 0.5f*t.x(), 0.5f*t.y(), 0.5f*t.z()));
			*this = vec_part * rot_part;
		}

		CPU_AND_GPU DualQuaternion<FloatType> operator+(const DualQuaternion<FloatType>& _dq) const {
			Quaternion<FloatType> quat0(q0 + _dq.q0);
			Quaternion<FloatType> quat1(q1 + _dq.q1);
			return{ quat0, quat1 };
		}

		CPU_AND_GPU DualQuaternion<FloatType> operator*(const DualQuaternion<FloatType>& _dq) const {
			Quaternion<FloatType> quat0(q0*_dq.q0);
			Quaternion<FloatType> quat1(q1*_dq.q0 + q0 * _dq.q1);
			return{ quat0, quat1 };
		}

		CPU_AND_GPU DualQuaternion<FloatType>& operator+=(const DualQuaternion<FloatType>& _dq) {
			*this = *this + _dq;
			return *this;
		}

		CPU_AND_GPU DualQuaternion<FloatType>& operator*=(const DualQuaternion<FloatType>& _dq) {
			*this = *this * _dq;
			return *this;
		}

		CPU_AND_GPU DualQuaternion<FloatType> operator*(const DualNumber<FloatType>& _dn) const {
			return _dn * *this;
		}

		CPU_AND_GPU DualQuaternion<FloatType>& operator*=(const DualNumber<FloatType>& _dn) {
			*this = *this * _dn;
			return *this;
		}

		CPU_AND_GPU DualNumber<FloatType> dual_number() const {
			return DualNumber<FloatType>(q0.w(), q1.w());
		}

		CPU_AND_GPU DualQuaternion<FloatType> conjugate() const {
			return { q0.conjugate(), q1.conjugate() };
		}

		CPU_AND_GPU DualNumber<FloatType> squared_norm() const {
			return *this * this->conjugate();
		}

		CPU_AND_GPU DualNumber<FloatType> norm() const {
			float a0 = q0.norm();
			float a1 = q0.dot(q1) / q0.norm();
			return { a0, a1 };
		}

		CPU_AND_GPU DualQuaternion<FloatType> inverse() const {
			return this->conjugate() * this->squared_norm().reciprocal();
		}

		CPU_AND_GPU void normalize() {
			*this = *this * this->norm().reciprocal();
		}

		CPU_AND_GPU DualQuaternion<FloatType> normalized() const {
			return *this * this->norm().reciprocal();
		}

		CPU_AND_GPU Mat4<FloatType> matrix() const {
			DualQuaternion<FloatType> quat_normalized = this->normalized();
			Mat3<FloatType> r = quat_normalized.q0.matrix();
			Quaternion<FloatType> vec_part = 2.0f*quat_normalized.q1*quat_normalized.q0.conjugate();
			Vec3<FloatType> t = vec_part.vec();

			return Mat4<FloatType>(r, t);
		}

		Quaternion<FloatType> q0, q1;
	};

	template<typename FloatType>
	CPU_AND_GPU DualQuaternion<FloatType> operator*(const DualNumber<FloatType>& _dn, const DualQuaternion<FloatType>& _dq) {
		Quaternion<FloatType> quat0 = _dn.q0*_dq.q0;
		Quaternion<FloatType> quat1 = _dn.q0*_dq.q1 + _dn.q1*_dq.q0;
		return { quat0, quat1 };
	}

	using DualQuatd = DualQuaternion<double>;
	using DualQuatf = DualQuaternion<float>;

} // namespace matrix_lib
