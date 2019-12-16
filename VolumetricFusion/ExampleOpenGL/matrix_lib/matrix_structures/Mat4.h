#pragma once
#include "Vec3.h"
#include "Vec4.h"

namespace matrix_lib {

	/**
	 * 4x4 matrix.
	 * The arrangement of the matrix is row-like.
	 * The index of a specific position is:
	 * 0  1   2  3
	 * 4  5   6  7
	 * 8  9  10  11
	 * 12 13 14  15
	 */
	template <class T>
	class Mat4 {
	public:
		/**
		 * Constructors and assignment operators.
		 */
		// An uninitialized matrix.
		CPU_AND_GPU Mat4() {
			setZero();
		}

		// Initialize with values stored in an array.
		CPU_AND_GPU Mat4(const T* values) {
			for (unsigned int i = 0; i < 16; i++) {
				m_matrix[i] = values[i];
			}
		}

		// Initialize with float4 row vectors.
		CPU_AND_GPU Mat4(const float4& xrow, const float4& yrow, const float4& zrow, const float4& wrow) {
			(*this)(0, 0) = xrow.x;
			(*this)(0, 1) = xrow.y;
			(*this)(0, 2) = xrow.z;
			(*this)(0, 3) = xrow.w;
			(*this)(1, 0) = yrow.x;
			(*this)(1, 1) = yrow.y;
			(*this)(1, 2) = yrow.z;
			(*this)(1, 3) = yrow.w;
			(*this)(2, 0) = zrow.x;
			(*this)(2, 1) = zrow.y;
			(*this)(2, 2) = zrow.z;
			(*this)(2, 3) = zrow.w;
			(*this)(3, 0) = wrow.x;
			(*this)(3, 1) = wrow.y;
			(*this)(3, 2) = wrow.z;
			(*this)(3, 3) = wrow.w;
		}

		// Initialize with int4 row vectors.
		CPU_AND_GPU Mat4(const int4& xrow, const int4& yrow, const int4& zrow, const int4& wrow) {
			(*this)(0, 0) = xrow.x;
			(*this)(0, 1) = xrow.y;
			(*this)(0, 2) = xrow.z;
			(*this)(0, 3) = xrow.w;
			(*this)(1, 0) = yrow.x;
			(*this)(1, 1) = yrow.y;
			(*this)(1, 2) = yrow.z;
			(*this)(1, 3) = yrow.w;
			(*this)(2, 0) = zrow.x;
			(*this)(2, 1) = zrow.y;
			(*this)(2, 2) = zrow.z;
			(*this)(2, 3) = zrow.w;
			(*this)(3, 0) = wrow.x;
			(*this)(3, 1) = wrow.y;
			(*this)(3, 2) = wrow.z;
			(*this)(3, 3) = wrow.w;
		}

		// Initializes the matrix row wise (given 3 row vectors); the last row and column are initialized with 0,0,0,1.
		CPU_AND_GPU Mat4(const Vec3<T>& v0, const Vec3<T>& v1, const Vec3<T>& v2) {
			(*this)(0, 0) = v0.x();
			(*this)(0, 1) = v0.y();
			(*this)(0, 2) = v0.z();
			(*this)(0, 3) = T(0);
			(*this)(1, 0) = v1.x();
			(*this)(1, 1) = v1.y();
			(*this)(1, 2) = v1.z();
			(*this)(1, 3) = T(0);
			(*this)(2, 0) = v2.x();
			(*this)(2, 1) = v2.y();
			(*this)(2, 2) = v2.z();
			(*this)(2, 3) = T(0);
			(*this)(3, 0) = T(0);
			(*this)(3, 1) = T(0);
			(*this)(3, 2) = T(0);
			(*this)(3, 3) = T(1);
		}

		// Initializes the matrix row wise (given 4 row Vectors).
		CPU_AND_GPU Mat4(const Vec4<T>& v0, const Vec4<T>& v1, const Vec4<T>& v2, const Vec4<T>& v3) {
			(*this)(0, 0) = v0.x();
			(*this)(0, 1) = v0.y();
			(*this)(0, 2) = v0.z();
			(*this)(0, 3) = v0.w();
			(*this)(1, 0) = v1.x();
			(*this)(1, 1) = v1.y();
			(*this)(1, 2) = v1.z();
			(*this)(1, 3) = v1.w();
			(*this)(2, 0) = v2.x();
			(*this)(2, 1) = v2.y();
			(*this)(2, 2) = v2.z();
			(*this)(2, 3) = v2.w();
			(*this)(3, 0) = v3.x();
			(*this)(3, 1) = v3.y();
			(*this)(3, 2) = v3.z();
			(*this)(3, 3) = v3.w();
		}

		// Initializes the matrix row wise.
		CPU_AND_GPU Mat4(const T& m00, const T& m01, const T& m02, const T& m03,
		          const T& m10, const T& m11, const T& m12, const T& m13,
		          const T& m20, const T& m21, const T& m22, const T& m23,
		          const T& m30, const T& m31, const T& m32, const T& m33) {
			(*this)(0, 0) = m00;
			(*this)(0, 1) = m01;
			(*this)(0, 2) = m02;
			(*this)(0, 3) = m03;
			(*this)(1, 0) = m10;
			(*this)(1, 1) = m11;
			(*this)(1, 2) = m12;
			(*this)(1, 3) = m13;
			(*this)(2, 0) = m20;
			(*this)(2, 1) = m21;
			(*this)(2, 2) = m22;
			(*this)(2, 3) = m23;
			(*this)(3, 0) = m30;
			(*this)(3, 1) = m31;
			(*this)(3, 2) = m32;
			(*this)(3, 3) = m33;
		}

		// Initializes the matrix row wise, from a tuple.
		template<typename TList>
		CPU_AND_GPU Mat4(const Tuple<TList>& t) {
			m_matrix[0] = t[I<0>()];
			m_matrix[1] = t[I<1>()];
			m_matrix[2] = t[I<2>()];
			m_matrix[3] = t[I<3>()];
			m_matrix[4] = t[I<4>()];
			m_matrix[5] = t[I<5>()];
			m_matrix[6] = t[I<6>()];
			m_matrix[7] = t[I<7>()];
			m_matrix[8] = t[I<8>()];
			m_matrix[9] = t[I<9>()];
			m_matrix[10] = t[I<10>()];
			m_matrix[11] = t[I<11>()];
			m_matrix[12] = t[I<12>()];
			m_matrix[13] = t[I<13>()];
			m_matrix[14] = t[I<14>()];
			m_matrix[15] = t[I<15>()];
		}

		// Initializes the matrix from a 3x3 matrix (last row and column are set to 0,0,0,1).
		CPU_AND_GPU Mat4(const Mat3<T>& other) {
			for (unsigned char i = 0; i < 3; i++) {
				for (unsigned char j = 0; j < 3; j++) {
					at(i, j) = other.at(i, j);
				}
				at(i, 3) = 0.0;
			}
			at(3, 0) = at(3, 1) = at(3, 2) = 0.0;
			at(3, 3) = 1.0;
		}

		// Initialize with a matrix from another type.
		template <class U>
		CPU_AND_GPU Mat4(const Mat4<U>& other) {
			for (unsigned int i = 0; i < 16; i++) {
				getData()[i] = T(other.getData()[i]);
			}
		}

		// Initializes a matrix from a 3x3 matrix + translation (last row set to 0,0,0,1).
		CPU_AND_GPU Mat4(const Mat3<T>& other, const Vec3<T>& t) {
			for (unsigned char i = 0; i < 3; i++) {
				for (unsigned char j = 0; j < 3; j++) {
					at(i, j) = other.at(i, j);
				}
				at(i, 3) = t[i];
			}
			at(3, 0) = at(3, 1) = at(3, 2) = 0.0;
			at(3, 3) = 1.0;
		}

		// Overwrites the 3x3 part; all other values remain unchanged.
		CPU_AND_GPU void setMatrix3x3(const Mat3<T>& m) {
			for (unsigned char i = 0; i < 3; i++) {
				for (unsigned char j = 0; j < 3; j++) {
					at(i, j) = m.at(i, j);
				}
			}
		}

		// Overwrites the 3x3 rotation part of the matrix; all other values remain unchanced.
		CPU_AND_GPU void setRotationMatrix(const Mat3<T>& m) {
			setMatrix3x3(m);
		}

		// Overwrites the translation vector; all other values remain unchanged.
		CPU_AND_GPU void setTranslationVector(const Vec3<T>& t) {
			at(0, 3) = t.x();
			at(1, 3) = t.y();
			at(2, 3) = t.z();
		}

		// Overwrites the translation vector; all other values remain unchanged.
		CPU_AND_GPU void setTranslationVector(T t) {
			at(0, 3) = t;
			at(1, 3) = t;
			at(2, 3) = t;
		}

		// Returns the 3x3 matrix part.
		CPU_AND_GPU Mat3<T> getMatrix3x3() const {
			Mat3<T> mat;
			for (unsigned char i = 0; i < 3; i++) {
				for (unsigned char j = 0; j < 3; j++) {
					mat.at(i, j) = at(i, j);
				}
			}
			return mat;
		}

		// Same as getMatrix3x3().
		CPU_AND_GPU Mat3<T> getRotation() const {
			return getMatrix3x3();
		}

		// Returns the translation part of the matrix.
		CPU_AND_GPU Vec3<T> getTranslation() const {
			return Vec3<T>(at(0, 3), at(1, 3), at(2, 3));
		}


		// Overwrite the matrix with an identity-matrix.
		CPU_AND_GPU void setIdentity() {
			setScale(1.0, 1.0, 1.0);
		}

		CPU_AND_GPU static Mat4 identity() {
			Mat4 res;
			res.setIdentity();
			return res;
		}

		// Sets the matrix zero (or a specified value).
		CPU_AND_GPU void setZero(T v = T(0)) {
			m_matrix[0] = m_matrix[1] = m_matrix[2] = m_matrix[3] = v;
			m_matrix[4] = m_matrix[5] = m_matrix[6] = m_matrix[7] = v;
			m_matrix[8] = m_matrix[9] = m_matrix[10] = m_matrix[11] = v;
			m_matrix[12] = m_matrix[13] = m_matrix[14] = m_matrix[15] = v;
		}

		CPU_AND_GPU static Mat4 zero(T v = T(0)) {
			Mat4 res;
			res.setZero(v);
			return res;
		}

		// Overwrite the matrix with a translation-matrix.
		CPU_AND_GPU void setTranslation(T t) {
			m_matrix[0] = T(1);
			m_matrix[1] = T(0);
			m_matrix[2] = T(0);
			m_matrix[3] = t;
			m_matrix[4] = T(0);
			m_matrix[5] = T(1);
			m_matrix[6] = T(0);
			m_matrix[7] = t;
			m_matrix[8] = T(0);
			m_matrix[9] = T(0);
			m_matrix[10] = T(1);
			m_matrix[11] = t;
			m_matrix[12] = T(0);
			m_matrix[13] = T(0);
			m_matrix[14] = T(0);
			m_matrix[15] = T(1);
		}

		CPU_AND_GPU static Mat4 translation(T t) {
			Mat4 res;
			res.setTranslation(t);
			return res;
		}

		// Overwrite the matrix with a translation-matrix.
		CPU_AND_GPU void setTranslation(T x, T y, T z) {
			m_matrix[0] = T(1);
			m_matrix[1] = T(0);
			m_matrix[2] = T(0);
			m_matrix[3] = x;
			m_matrix[4] = T(0);
			m_matrix[5] = T(1);
			m_matrix[6] = T(0);
			m_matrix[7] = y;
			m_matrix[8] = T(0);
			m_matrix[9] = T(0);
			m_matrix[10] = T(1);
			m_matrix[11] = z;
			m_matrix[12] = T(0);
			m_matrix[13] = T(0);
			m_matrix[14] = T(0);
			m_matrix[15] = T(1);
		}

		CPU_AND_GPU static Mat4 translation(T x, T y, T z) {
			Mat4 res;
			res.setTranslation(x, y, z);
			return res;
		}

		// Overwrite the matrix with a translation-matrix
		CPU_AND_GPU void setTranslation(const Vec3<T>& v) {
			m_matrix[0] = T(1);
			m_matrix[1] = T(0);
			m_matrix[2] = T(0);
			m_matrix[3] = v.x();
			m_matrix[4] = T(0);
			m_matrix[5] = T(1);
			m_matrix[6] = T(0);
			m_matrix[7] = v.y();
			m_matrix[8] = T(0);
			m_matrix[9] = T(0);
			m_matrix[10] = T(1);
			m_matrix[11] = v.z();
			m_matrix[12] = T(0);
			m_matrix[13] = T(0);
			m_matrix[14] = T(0);
			m_matrix[15] = T(1);
		}

		CPU_AND_GPU static Mat4 translation(const Vec3<T>& v) {
			Mat4 res;
			res.setTranslation(v);
			return res;
		}

		// Overwrite the matrix with a rotation-matrix around a coordinate-axis (angle is specified in degrees).
		CPU_AND_GPU void setRotationX(T angle) {
			T angleRad = math_proc::degreesToRadians(angle);
			T sinAngle = sin(angleRad);
			T cosAngle = cos(angleRad);

			m_matrix[0] = T(1);
			m_matrix[1] = T(0);
			m_matrix[2] = T(0);
			m_matrix[3] = T(0);
			m_matrix[4] = T(0);
			m_matrix[5] = cosAngle;
			m_matrix[6] = -sinAngle;
			m_matrix[7] = T(0);
			m_matrix[8] = T(0);
			m_matrix[9] = sinAngle;
			m_matrix[10] = cosAngle;
			m_matrix[11] = T(0);
			m_matrix[12] = T(0);
			m_matrix[13] = T(0);
			m_matrix[14] = T(0);
			m_matrix[15] = T(1);
		}

		CPU_AND_GPU static Mat4 rotationX(T angle) {
			Mat4 res;
			res.setRotationX(angle);
			return res;
		}

		// Overwrite the matrix with a rotation-matrix around a coordinate-axis (angle is specified in degrees).
		CPU_AND_GPU void setRotationY(T angle) {
			T angleRad = math_proc::degreesToRadians(angle);
			T sinAngle = sin(angleRad);
			T cosAngle = cos(angleRad);

			m_matrix[0] = cosAngle;
			m_matrix[1] = T(0);
			m_matrix[2] = sinAngle;
			m_matrix[3] = T(0);
			m_matrix[4] = T(0);
			m_matrix[5] = T(1);
			m_matrix[6] = T(0);
			m_matrix[7] = T(0);
			m_matrix[8] = -sinAngle;
			m_matrix[9] = T(0);
			m_matrix[10] = cosAngle;
			m_matrix[11] = T(0);
			m_matrix[12] = T(0);
			m_matrix[13] = T(0);
			m_matrix[14] = T(0);
			m_matrix[15] = T(1);
		}

		CPU_AND_GPU static Mat4 rotationY(T angle) {
			Mat4 res;
			res.setRotationY(angle);
			return res;
		}

		// Overwrite the matrix with a rotation-matrix around a coordinate-axis (angle is specified in degrees).
		CPU_AND_GPU void setRotationZ(T angle) {
			T angleRad = math_proc::degreesToRadians(angle);
			T sinAngle = sin(angleRad);
			T cosAngle = cos(angleRad);

			m_matrix[0] = cosAngle;
			m_matrix[1] = -sinAngle;
			m_matrix[2] = T(0);
			m_matrix[3] = T(0);
			m_matrix[4] = sinAngle;
			m_matrix[5] = cosAngle;
			m_matrix[6] = T(0);
			m_matrix[7] = T(0);
			m_matrix[8] = T(0);
			m_matrix[9] = T(0);
			m_matrix[10] = T(1);
			m_matrix[11] = T(0);
			m_matrix[12] = T(0);
			m_matrix[13] = T(0);
			m_matrix[14] = T(0);
			m_matrix[15] = T(1);
		}

		CPU_AND_GPU static Mat4 rotationZ(T angle) {
			Mat4 res;
			res.setRotationZ(angle);
			return res;
		}

		// Overwrite the matrix with a rotation-matrix around a coordinate-axis (angle is specified in degrees).
		CPU_AND_GPU void setRotation(T yaw, T pitch, T roll) {
			*this = rotationY(yaw) * rotationX(pitch) * rotationZ(roll);
		}

		CPU_AND_GPU static Mat4 rotation(T yaw, T pitch, T roll) {
			Mat4 res;
			res.setRotation(yaw, pitch, roll);
			return res;
		}

		// Overwrite the matrix with a rotation-matrix around a coordinate-axis (angle is specified in degrees).
		CPU_AND_GPU void setRotation(const Vec3<T>& axis, T angle) {
			*this = Mat3<T>::rotation(axis, angle);
		}

		CPU_AND_GPU static Mat4 rotation(const Vec3<T>& axis, T angle) {
			Mat4 res;
			res.setRotation(axis, angle);
			return res;
		}

		// Overwrite the matrix with a rotation-matrix around a coordinate-axis (angle is specified in degrees).
		CPU_AND_GPU void setRotation(const Vec3<T>& axis, T angle, const Vec3<T>& center) {
			*this = translation(-center) * rotation(axis, angle) * translation(center);
		}

		CPU_AND_GPU static Mat4 rotation(const Vec3<T>& axis, T angle, const Vec3<T>& center) {
			Mat4 ret;
			ret.setRotation(axis, angle, center);
			return ret;
		}

		// Generates a pose matrix.
		CPU_AND_GPU static Mat4 pose(T yaw, T pitch, T roll, T x, T y, T z) {
			Mat4 res;
			res.setRotation(yaw, pitch, roll);
			res.at(0, 3) = x;
			res.at(1, 3) = y;
			res.at(2, 3) = z;
			return res;
		}

		// Overwrite the matrix with a scale-matrix.
		CPU_AND_GPU void setScale(T x, T y, T z) {
			m_matrix[0] = x;
			m_matrix[1] = T(0);
			m_matrix[2] = T(0);
			m_matrix[3] = T(0);
			m_matrix[4] = T(0);
			m_matrix[5] = y;
			m_matrix[6] = T(0);
			m_matrix[7] = T(0);
			m_matrix[8] = T(0);
			m_matrix[9] = T(0);
			m_matrix[10] = z;
			m_matrix[11] = T(0);
			m_matrix[12] = T(0);
			m_matrix[13] = T(0);
			m_matrix[14] = T(0);
			m_matrix[15] = T(1);
		}

		CPU_AND_GPU static Mat4 scale(T x, T y, T z) {
			Mat4 res;
			res.setScale(x, y, z);
			return res;
		}

		// Overwrite the matrix with a scale-matrix.
		CPU_AND_GPU void setScale(T s) {
			setScale(s, s, s);
		}

		CPU_AND_GPU static Mat4 scale(T s) {
			Mat4 res;
			res.setScale(s);
			return res;
		}

		// Overwrite the matrix with a scale-matrix.
		CPU_AND_GPU void setScale(const Vec3<T>& v) {
			m_matrix[0] = v.x();
			m_matrix[1] = T(0);
			m_matrix[2] = T(0);
			m_matrix[3] = T(0);
			m_matrix[4] = T(0);
			m_matrix[5] = v.y();
			m_matrix[6] = T(0);
			m_matrix[7] = T(0);
			m_matrix[8] = T(0);
			m_matrix[9] = T(0);
			m_matrix[10] = v.z();
			m_matrix[11] = T(0);
			m_matrix[12] = T(0);
			m_matrix[13] = T(0);
			m_matrix[14] = T(0);
			m_matrix[15] = T(1);
		}

		CPU_AND_GPU static Mat4 scale(const Vec3<T>& v) {
			Mat4 res;
			res.setScale(v);
			return res;
		}

		// Overwrite the matrix with a diagonal matrix.
		CPU_AND_GPU void setDiag(T x, T y, T z, T w) {
			setScale(x, y, z);
			m_matrix[15] = w;
		}

		CPU_AND_GPU static Mat4 diag(T x, T y, T z, T w) {
			Mat4 res;
			res.setDiag(x, y, z, w);
			return res;
		}

		/**
		 * Basic operations.
		 */
		// Equal operator
		CPU_AND_GPU bool operator==(const Mat4<T>& other) const {
			for (unsigned i = 0; i < 16; i++) {
				if (m_matrix[i] != other[i]) return false;
			}
			return true;
		}

		// Not equal operator.
		CPU_AND_GPU bool operator!=(const Mat4<T>& other) const {
			return !(*this == other);
		}

		CPU_AND_GPU T trace() const {
			return (m_matrix[0] + m_matrix[5] + m_matrix[10] + m_matrix[15]);
		}


		// Return the product of the operand with matrix
		CPU_AND_GPU Mat4 operator*(const Mat4& other) const {
			Mat4<T> result;
			//unrolling is slower (surprise?)
			for (unsigned char i = 0; i < 4; i++) {
				for (unsigned char j = 0; j < 4; j++) {
					result.at(i, j) =
						this->at(i, 0) * other.at(0, j) +
						this->at(i, 1) * other.at(1, j) +
						this->at(i, 2) * other.at(2, j) +
						this->at(i, 3) * other.at(3, j);
				}
			}
			return result;
		}

		// Multiply operand with matrix b.
		CPU_AND_GPU Mat4& operator*=(const Mat4& other) {
			Mat4<T> prod = (*this) * other;
			*this = prod;
			return *this;
		}

		// Multiply each element in the matrix with a scalar factor.
		CPU_AND_GPU Mat4 operator*(T r) const {
			Mat4<T> result;
			for (unsigned int i = 0; i < 16; i++) {
				result.m_matrix[i] = m_matrix[i] * r;
			}
			return result;
		}

		// Multiply each element in the matrix with a scalar factor.
		CPU_AND_GPU Mat4& operator*=(T r) {
			for (unsigned int i = 0; i < 16; i++) {
				m_matrix[i] *= r;
			}
			return *this;
		}

		// Divide the matrix by a scalar factor.
		CPU_AND_GPU Mat4 operator/(T r) const {
			Mat4<T> result;
			for (unsigned int i = 0; i < 16; i++) {
				result.m_matrix[i] = m_matrix[i] / r;
			}
			return result;
		}

		// Divide each element in the matrix with a scalar factor.
		CPU_AND_GPU Mat4& operator/=(T r) {
			for (unsigned int i = 0; i < 16; i++) {
				m_matrix[i] /= r;
			}
			return *this;
		}

		// Transform a 4D-Vector with the matrix.
		CPU_AND_GPU Vec4<T> operator*(const Vec4<T>& v) const {
			return Vec4<T>(
				m_matrix[0] * v.x() + m_matrix[1] * v.y() + m_matrix[2] * v.z() + m_matrix[3] * v.w(),
				m_matrix[4] * v.x() + m_matrix[5] * v.y() + m_matrix[6] * v.z() + m_matrix[7] * v.w(),
				m_matrix[8] * v.x() + m_matrix[9] * v.y() + m_matrix[10] * v.z() + m_matrix[11] * v.w(),
				m_matrix[12] * v.x() + m_matrix[13] * v.y() + m_matrix[14] * v.z() + m_matrix[15] * v.w()
			);
		}

		// Transform a 3D-Vector with the matrix (implicit w=1 and implicit subsequent de-homogenization).
		CPU_AND_GPU Vec3<T> operator*(const Vec3<T>& v) const {
			Vec4<T> result(
				m_matrix[0] * v.x() + m_matrix[1] * v.y() + m_matrix[2] * v.z() + m_matrix[3],
				m_matrix[4] * v.x() + m_matrix[5] * v.y() + m_matrix[6] * v.z() + m_matrix[7],
				m_matrix[8] * v.x() + m_matrix[9] * v.y() + m_matrix[10] * v.z() + m_matrix[11],
				m_matrix[12] * v.x() + m_matrix[13] * v.y() + m_matrix[14] * v.z() + m_matrix[15]
			);
			result.dehomogenize();
			return Vec3<T>(result.x(), result.y(), result.z());
		}

		// Return the sum of the operand with matrix b.
		CPU_AND_GPU Mat4 operator+(const Mat4& other) const {
			Mat4<T> result;
			for (unsigned int i = 0; i < 16; i++) {
				result.m_matrix[i] = m_matrix[i] + other.m_matrix[i];
			}
			return result;
		}

		// Add matrix other to the operand.
		CPU_AND_GPU Mat4& operator+=(const Mat4& other) {
			for (unsigned int i = 0; i < 16; i++) {
				m_matrix[i] += other.m_matrix[i];
			}
			return *this;
		}

		// Return the difference of the operand with matrix b.
		CPU_AND_GPU Mat4 operator-(const Mat4& other) const {
			Mat4<T> result;
			for (unsigned int i = 0; i < 16; i++) {
				result.m_matrix[i] = m_matrix[i] - other.m_matrix[i];
			}
			return result;
		}

		// Subtract matrix other from the operand.
		CPU_AND_GPU Mat4 operator-=(const Mat4& other) {
			for (unsigned int i = 0; i < 16; i++) {
				m_matrix[i] -= other.m_matrix[i];
			}
			return *this;
		}

		// Return the determinant of the matrix.
		CPU_AND_GPU T det() const {
			return m_matrix[0] * det3x3(1, 2, 3, 1, 2, 3)
				- m_matrix[4] * det3x3(0, 2, 3, 1, 2, 3)
				+ m_matrix[8] * det3x3(0, 1, 3, 1, 2, 3)
				- m_matrix[12] * det3x3(0, 1, 2, 1, 2, 3);
		}

		// Return the determinant of the 3x3 sub-matrix.
		CPU_AND_GPU T det3x3() const {
			return det3x3(0, 1, 2, 0, 1, 2);
		}

		/**
		 * Indexing operators.
		 */
		// Access element of matrix at the given row and column for constant access.
		CPU_AND_GPU T at(unsigned char row, unsigned char col) const {
			runtime_assert((row<4) && (col<4), "Index out of bounds.");
			return m_matrix[col + row * 4];
		}

		// Access element of matrix at the given row and column.
		CPU_AND_GPU T& at(unsigned char row, unsigned char col) {
			runtime_assert((row<4) && (col<4), "Index out of bounds.");
			return m_matrix[col + row * 4];
		}

		// Access element of matrix at the given row and column for constant access.
		CPU_AND_GPU T operator()(unsigned int row, unsigned int col) const {
			runtime_assert((row<4) && (col<4), "Index out of bounds.");
			return m_matrix[col + row * 4];
		}

		// Access element of matrix at the given row and column.
		CPU_AND_GPU T& operator()(unsigned int row, unsigned int col) {
			runtime_assert((row<4) && (col<4), "Index out of bounds.");
			return m_matrix[col + row * 4];
		}

		// Access i-th element of the matrix for constant access.
		CPU_AND_GPU T operator[](unsigned int i) const {
			runtime_assert(i < 16, "Index out of bounds.");
			return m_matrix[i];
		}

		// Access i-th element of the matrix.
		CPU_AND_GPU T& operator[](unsigned int i) {
			runtime_assert(i < 16, "Index out of bounds.");
			return m_matrix[i];
		}

		// Access i-th element of the matrix for constant access at compile-time.
		template<unsigned i>
		CPU_AND_GPU const T& operator[](I<i>) const {
			static_assert(i < 16, "Index out of bounds.");
			return m_matrix[i];
		}

		// Access i-th element of the matrix at compile-time.
		template<unsigned i>
		CPU_AND_GPU T& operator[](I<i>) {
			static_assert(i < 16, "Index out of bounds.");
			return m_matrix[i];
		}

		/**
		 * Getters/setters.
		 */
		// Get the x column out of the matrix.
		CPU_AND_GPU Vec4<T> xcol() const {
			return Vec4<T>(m_matrix[0], m_matrix[4], m_matrix[8], m_matrix[12]);
		}

		// Get the y column out of the matrix.
		CPU_AND_GPU Vec4<T> ycol() const {
			return Vec4<T>(m_matrix[1], m_matrix[5], m_matrix[9], m_matrix[13]);
		}

		// Get the y column out of the matrix.
		CPU_AND_GPU Vec4<T> zcol() const {
			return Vec4<T>(m_matrix[2], m_matrix[6], m_matrix[10], m_matrix[14]);
		}

		// Get the t column out of the matrix.
		CPU_AND_GPU Vec4<T> tcol() const {
			return Vec4<T>(m_matrix[3], m_matrix[7], m_matrix[11], m_matrix[15]);
		}

		// Get the x row out of the matrix.
		CPU_AND_GPU Vec4<T> xrow() const {
			return Vec4<T>(m_matrix[0], m_matrix[1], m_matrix[2], m_matrix[3]);
		}

		// Get the y row out of the matrix.
		CPU_AND_GPU Vec4<T> yrow() const {
			return Vec4<T>(m_matrix[4], m_matrix[5], m_matrix[6], m_matrix[7]);
		}

		// Get the y row out of the matrix.
		CPU_AND_GPU Vec4<T> zrow() const {
			return Vec4<T>(m_matrix[8], m_matrix[9], m_matrix[10], m_matrix[11]);
		}

		// Get the t row out of the matrix.
		CPU_AND_GPU Vec4<T> trow() const {
			return Vec4<T>(m_matrix[12], m_matrix[13], m_matrix[14], m_matrix[15]);
		}

		CPU_AND_GPU const T* getData() const {
			return &m_matrix[0];
		}

		CPU_AND_GPU T* getData() {
			return &m_matrix[0];
		}

		// Return the inverse matrix; but does not change the current matrix.
		CPU_AND_GPU Mat4<T> getInverse() const {
			T inv[16];

			inv[0] = m_matrix[5] * m_matrix[10] * m_matrix[15] -
				m_matrix[5] * m_matrix[11] * m_matrix[14] -
				m_matrix[9] * m_matrix[6] * m_matrix[15] +
				m_matrix[9] * m_matrix[7] * m_matrix[14] +
				m_matrix[13] * m_matrix[6] * m_matrix[11] -
				m_matrix[13] * m_matrix[7] * m_matrix[10];

			inv[4] = -m_matrix[4] * m_matrix[10] * m_matrix[15] +
				m_matrix[4] * m_matrix[11] * m_matrix[14] +
				m_matrix[8] * m_matrix[6] * m_matrix[15] -
				m_matrix[8] * m_matrix[7] * m_matrix[14] -
				m_matrix[12] * m_matrix[6] * m_matrix[11] +
				m_matrix[12] * m_matrix[7] * m_matrix[10];

			inv[8] = m_matrix[4] * m_matrix[9] * m_matrix[15] -
				m_matrix[4] * m_matrix[11] * m_matrix[13] -
				m_matrix[8] * m_matrix[5] * m_matrix[15] +
				m_matrix[8] * m_matrix[7] * m_matrix[13] +
				m_matrix[12] * m_matrix[5] * m_matrix[11] -
				m_matrix[12] * m_matrix[7] * m_matrix[9];

			inv[12] = -m_matrix[4] * m_matrix[9] * m_matrix[14] +
				m_matrix[4] * m_matrix[10] * m_matrix[13] +
				m_matrix[8] * m_matrix[5] * m_matrix[14] -
				m_matrix[8] * m_matrix[6] * m_matrix[13] -
				m_matrix[12] * m_matrix[5] * m_matrix[10] +
				m_matrix[12] * m_matrix[6] * m_matrix[9];

			inv[1] = -m_matrix[1] * m_matrix[10] * m_matrix[15] +
				m_matrix[1] * m_matrix[11] * m_matrix[14] +
				m_matrix[9] * m_matrix[2] * m_matrix[15] -
				m_matrix[9] * m_matrix[3] * m_matrix[14] -
				m_matrix[13] * m_matrix[2] * m_matrix[11] +
				m_matrix[13] * m_matrix[3] * m_matrix[10];

			inv[5] = m_matrix[0] * m_matrix[10] * m_matrix[15] -
				m_matrix[0] * m_matrix[11] * m_matrix[14] -
				m_matrix[8] * m_matrix[2] * m_matrix[15] +
				m_matrix[8] * m_matrix[3] * m_matrix[14] +
				m_matrix[12] * m_matrix[2] * m_matrix[11] -
				m_matrix[12] * m_matrix[3] * m_matrix[10];

			inv[9] = -m_matrix[0] * m_matrix[9] * m_matrix[15] +
				m_matrix[0] * m_matrix[11] * m_matrix[13] +
				m_matrix[8] * m_matrix[1] * m_matrix[15] -
				m_matrix[8] * m_matrix[3] * m_matrix[13] -
				m_matrix[12] * m_matrix[1] * m_matrix[11] +
				m_matrix[12] * m_matrix[3] * m_matrix[9];

			inv[13] = m_matrix[0] * m_matrix[9] * m_matrix[14] -
				m_matrix[0] * m_matrix[10] * m_matrix[13] -
				m_matrix[8] * m_matrix[1] * m_matrix[14] +
				m_matrix[8] * m_matrix[2] * m_matrix[13] +
				m_matrix[12] * m_matrix[1] * m_matrix[10] -
				m_matrix[12] * m_matrix[2] * m_matrix[9];

			inv[2] = m_matrix[1] * m_matrix[6] * m_matrix[15] -
				m_matrix[1] * m_matrix[7] * m_matrix[14] -
				m_matrix[5] * m_matrix[2] * m_matrix[15] +
				m_matrix[5] * m_matrix[3] * m_matrix[14] +
				m_matrix[13] * m_matrix[2] * m_matrix[7] -
				m_matrix[13] * m_matrix[3] * m_matrix[6];

			inv[6] = -m_matrix[0] * m_matrix[6] * m_matrix[15] +
				m_matrix[0] * m_matrix[7] * m_matrix[14] +
				m_matrix[4] * m_matrix[2] * m_matrix[15] -
				m_matrix[4] * m_matrix[3] * m_matrix[14] -
				m_matrix[12] * m_matrix[2] * m_matrix[7] +
				m_matrix[12] * m_matrix[3] * m_matrix[6];

			inv[10] = m_matrix[0] * m_matrix[5] * m_matrix[15] -
				m_matrix[0] * m_matrix[7] * m_matrix[13] -
				m_matrix[4] * m_matrix[1] * m_matrix[15] +
				m_matrix[4] * m_matrix[3] * m_matrix[13] +
				m_matrix[12] * m_matrix[1] * m_matrix[7] -
				m_matrix[12] * m_matrix[3] * m_matrix[5];

			inv[14] = -m_matrix[0] * m_matrix[5] * m_matrix[14] +
				m_matrix[0] * m_matrix[6] * m_matrix[13] +
				m_matrix[4] * m_matrix[1] * m_matrix[14] -
				m_matrix[4] * m_matrix[2] * m_matrix[13] -
				m_matrix[12] * m_matrix[1] * m_matrix[6] +
				m_matrix[12] * m_matrix[2] * m_matrix[5];

			inv[3] = -m_matrix[1] * m_matrix[6] * m_matrix[11] +
				m_matrix[1] * m_matrix[7] * m_matrix[10] +
				m_matrix[5] * m_matrix[2] * m_matrix[11] -
				m_matrix[5] * m_matrix[3] * m_matrix[10] -
				m_matrix[9] * m_matrix[2] * m_matrix[7] +
				m_matrix[9] * m_matrix[3] * m_matrix[6];

			inv[7] = m_matrix[0] * m_matrix[6] * m_matrix[11] -
				m_matrix[0] * m_matrix[7] * m_matrix[10] -
				m_matrix[4] * m_matrix[2] * m_matrix[11] +
				m_matrix[4] * m_matrix[3] * m_matrix[10] +
				m_matrix[8] * m_matrix[2] * m_matrix[7] -
				m_matrix[8] * m_matrix[3] * m_matrix[6];

			inv[11] = -m_matrix[0] * m_matrix[5] * m_matrix[11] +
				m_matrix[0] * m_matrix[7] * m_matrix[9] +
				m_matrix[4] * m_matrix[1] * m_matrix[11] -
				m_matrix[4] * m_matrix[3] * m_matrix[9] -
				m_matrix[8] * m_matrix[1] * m_matrix[7] +
				m_matrix[8] * m_matrix[3] * m_matrix[5];

			inv[15] = m_matrix[0] * m_matrix[5] * m_matrix[10] -
				m_matrix[0] * m_matrix[6] * m_matrix[9] -
				m_matrix[4] * m_matrix[1] * m_matrix[10] +
				m_matrix[4] * m_matrix[2] * m_matrix[9] +
				m_matrix[8] * m_matrix[1] * m_matrix[6] -
				m_matrix[8] * m_matrix[2] * m_matrix[5];

			T matrixDet = m_matrix[0] * inv[0] + m_matrix[1] * inv[4] + m_matrix[2] * inv[8] + m_matrix[3] * inv[12];

			T matrixDetr = T(1.0) / matrixDet;

			Mat4<T> res;
			for (unsigned int i = 0; i < 16; i++) {
				res[i] = inv[i] * matrixDetr;
			}
			return res;

		}

		// Overwrite the current matrix with its inverse
		CPU_AND_GPU void invert() {
			*this = getInverse();
		}

		// Return the transposed matrix.
		CPU_AND_GPU Mat4 getTranspose() const {
			Mat4<T> result;
			for (unsigned char x = 0; x < 4; x++) {
				result.at(x, 0) = at(0, x);
				result.at(x, 1) = at(1, x);
				result.at(x, 2) = at(2, x);
				result.at(x, 3) = at(3, x);
			}
			return result;
		}

		// Transpose the matrix in place.
		CPU_AND_GPU void transpose() {
			*this = getTranspose();
		}

		// Save matrix to .mat file.
		CPU_AND_GPU void saveMatrixToFile(const std::string& name) const {
			std::ofstream out(name.c_str());
			out << 4 << " " << 4 << "\n";
			for (unsigned int i = 0; i < 4; i++) {
				for (unsigned int j = 0; j < 4; j++) {
					out << (*this)(i, j) << " ";
				}
				out << "\n";
			}
		}

		// Load matrix from .mat file.
		CPU_AND_GPU void loadMatrixFromFile(const std::string& name) {
			std::ifstream in(name.c_str());
			unsigned int height, width;
			in >> height >> width;
			runtime_assert(height == width && height == 4, "Matrix dimensions are not correct.");
			for (unsigned int i = 0; i < 4; i++) {
				for (unsigned int j = 0; j < 4; j++) {
					in >> (*this)(i, j);
				}
			}
		}

		CPU_AND_GPU std::string toString(const std::string& seperator) const {
			std::string result;
			for (int i = 0; i < 4; i++)
				for (int j = 0; j < 4; j++) {
					result += to_string(m_matrix[i + j * 4]);
					if (i != 3 || j != 3)
						result += seperator;
				}
			return result;
		}

	private:
		T m_matrix[16];

		// Calculate determinant of a 3x3 sub-matrix given by the indices of the rows and columns.
		CPU_AND_GPU T det3x3(unsigned int i0 = 0, unsigned int i1 = 1, unsigned int i2 = 2, unsigned int j0 = 0,
		                 unsigned int j1 = 1, unsigned int j2 = 2) const {
			return
				((*this)(i0, j0) * (*this)(i1, j1) * (*this)(i2, j2))
				+ ((*this)(i0, j1) * (*this)(i1, j2) * (*this)(i2, j0))
				+ ((*this)(i0, j2) * (*this)(i1, j0) * (*this)(i2, j1))
				- ((*this)(i2, j0) * (*this)(i1, j1) * (*this)(i0, j2))
				- ((*this)(i2, j1) * (*this)(i1, j2) * (*this)(i0, j0))
				- ((*this)(i2, j2) * (*this)(i1, j0) * (*this)(i0, j1));
		}
	};

	/**
	 * Math operations.
	 */
	template <class T>
	CPU_AND_GPU Mat4<T> operator*(T s, const Mat4<T>& m) {
		return m * s;
	}

	template <class T>
	CPU_AND_GPU Mat4<T> operator/(T s, const Mat4<T>& m) {
		return Mat4<T>(
			s / m(0, 0), s / m(0, 1), s / m(0, 2), s / m(0, 3),
			s / m(1, 0), s / m(1, 1), s / m(1, 2), s / m(1, 3),
			s / m(2, 0), s / m(2, 1), s / m(2, 2), s / m(2, 3),
			s / m(3, 0), s / m(3, 1), s / m(3, 2), s / m(3, 3)
		);
	}

	template <class T>
	CPU_AND_GPU Mat4<T> operator+(T s, const Mat4<T>& m) {
		return m + s;
	}

	template <class T>
	CPU_AND_GPU Mat4<T> operator-(T s, const Mat4<T>& m) {
		return -m + s;
	}

	// Writes to a stream.
	template <class T>
	CPU_AND_GPU std::ostream& operator<<(std::ostream& s, const Mat4<T>& m) {
		return (
			s <<
			m(0, 0) << " " << m(0, 1) << " " << m(0, 2) << " " << m(0, 3) << std::endl <<
			m(1, 0) << " " << m(1, 1) << " " << m(1, 2) << " " << m(1, 3) << std::endl <<
			m(2, 0) << " " << m(2, 1) << " " << m(2, 2) << " " << m(2, 3) << std::endl <<
			m(3, 0) << " " << m(3, 1) << " " << m(3, 2) << " " << m(3, 3) << std::endl
		);
	}

	// Reads from a stream.
	template <class T>
	CPU_AND_GPU std::istream& operator>>(std::istream& s, Mat4<T>& m) {
		return (
			s >>
			m(0, 0) >> m(0, 1) >> m(0, 2) >> m(0, 3) >>
			m(1, 0) >> m(1, 1) >> m(1, 2) >> m(1, 3) >>
			m(2, 0) >> m(2, 1) >> m(2, 2) >> m(2, 3) >>
			m(3, 0) >> m(3, 1) >> m(3, 2) >> m(3, 3));
	}

	typedef Mat4<int> Mat4i;
	typedef Mat4<int> Mat4u;
	typedef Mat4<float> Mat4f;
	typedef Mat4<double> Mat4d;

} // namespace matrix_lib 
