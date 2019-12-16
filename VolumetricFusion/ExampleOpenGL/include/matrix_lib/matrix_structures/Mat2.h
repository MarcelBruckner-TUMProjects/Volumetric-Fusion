#pragma once
#include "Vec2.h"

namespace matrix_lib {

	/**
	 * 2x2 matrix.
	 * The arrangement of the matrix is row-like.
	 * The index of a specific position is:
	 * 0  1
	 * 2  3
	 */
	template <class T>
	class Mat2 {
	public:
		/**
		 * Constructors and assignment operators.
		 */
		// An uninitialized matrix.
		CPU_AND_GPU Mat2() {
			setZero();
		}

		// Initialize with float4 row vectors.
		CPU_AND_GPU Mat2(const float4& xrow, const float4& yrow) {
			m_matrix[0] = xrow.x;
			m_matrix[1] = xrow.y;
			m_matrix[2] = yrow.x;
			m_matrix[3] = yrow.y;
		}

		// Initialize with int4 row vectors.
		CPU_AND_GPU Mat2(const int4& xrow, const int4& yrow) {
			m_matrix[0] = xrow.x;
			m_matrix[1] = xrow.y;
			m_matrix[2] = yrow.x;
			m_matrix[3] = yrow.y;
		}

		// Initialize with values stored in an array.
		CPU_AND_GPU Mat2(const T* values) {
			for (unsigned int i = 0; i < 4; i++) {
				m_matrix[i] = values[i];
			}
		}

		// Initialize from 2 row vectors.
		CPU_AND_GPU Mat2(const Vec2<T>& v0, const Vec2<T>& v1) {
			m_matrix[0] = v0.x();
			m_matrix[1] = v0.y();
			m_matrix[2] = v1.x();
			m_matrix[3] = v1.y();
		}

		// Initializes the matrix row wise.
		CPU_AND_GPU Mat2(const T& m00, const T& m01,
		          const T& m10, const T& m11) {
			m_matrix[0] = m00;
			m_matrix[1] = m01;
			m_matrix[2] = m10;
			m_matrix[3] = m11;
		}

		// Initializes the matrix row wise, from a tuple.
		template<typename TList>
		CPU_AND_GPU Mat2(const Tuple<TList>& t) {
			m_matrix[0] = t[I<0>()];
			m_matrix[1] = t[I<1>()];
			m_matrix[2] = t[I<2>()];
			m_matrix[3] = t[I<3>()];
		}

		// Initialize with a matrix from another type.
		template <class U>
		CPU_AND_GPU Mat2(const Mat2<U>& other) {
			for (unsigned int i = 0; i < 4; i++) {
				m_matrix[i] = T(other.getData()[i]);
			}
		}

		// Overwrite the m_matrix with an identity-matrix.
		CPU_AND_GPU void setIdentity() {
			setScale(1.0, 1.0);
		}

		CPU_AND_GPU static Mat2 identity() {
			Mat2 res;
			res.setIdentity();
			return res;
		}

		// Sets the matrix zero (or a specified value).
		CPU_AND_GPU void setZero(T v = T(0)) {
			m_matrix[0] = m_matrix[1] = v;
			m_matrix[2] = m_matrix[3] = v;
		}

		CPU_AND_GPU static Mat2 zero(T v = T(0)) {
			Mat2 res;
			res.setZero(v);
			return res;
		}

		//! overwrite the matrix with a scale-matrix.
		CPU_AND_GPU void setScale(T x, T y) {
			m_matrix[0] = x;
			m_matrix[1] = T(0);
			m_matrix[2] = T(0);
			m_matrix[3] = y;
		}

		CPU_AND_GPU static Mat2 scale(T x, T y) {
			Mat2 res;
			res.setScale(x, y);
			return res;
		}

		// Overwrite the matrix with a scale-matrix.
		CPU_AND_GPU void setScale(T s) {
			setScale(s, s);
		}

		CPU_AND_GPU static Mat2 scale(T s) {
			Mat2 res;
			res.setScale(s);
			return res;
		}

		// Overwrite the matrix with a scale-matrix
		CPU_AND_GPU void setScale(const Vec2<T>& v) {
			m_matrix[0] = v.x;
			m_matrix[1] = T(0);
			m_matrix[2] = T(0);
			m_matrix[3] = v.y;
		}

		CPU_AND_GPU static Mat2 scale(const Vec2<T>& v) {
			Mat2 res;
			res.setScale(v);
			return res;
		}

		// Overwrite the matrix with a diagonal matrix.
		CPU_AND_GPU void setDiag(T x, T y) {
			setScale(x, y);
		}

		CPU_AND_GPU static Mat2 diag(T x, T y) {
			Mat2 res;
			res.setDiag(x, y);
			return res;
		}

		/**
		 * Basic operations.
		 */
		// Equal operator.
		CPU_AND_GPU bool operator==(const Mat2<T>& other) const {
			for (unsigned i = 0; i < 4; i++) {
				if (m_matrix[i] != other[i]) return false;
			}
			return true;
		}

		// Not equal operator.
		CPU_AND_GPU bool operator!=(const Mat2<T>& other) const {
			return !(*this == other);
		}

		CPU_AND_GPU T trace() const {
			return (m_matrix[0] + m_matrix[3]);
		}

		// Return the product of the operand with matrix.
		CPU_AND_GPU Mat2 operator*(const Mat2& other) const {
			Mat2<T> result;
			//TODO unroll the loop
			for (unsigned char i = 0; i < 2; i++) {
				for (unsigned char j = 0; j < 2; j++) {
					result.at(i, j) =
						this->at(i, 0) * other.at(0, j) +
						this->at(i, 1) * other.at(1, j);
				}
			}
			return result;
		}

		// Multiply operand with m_matrix b.
		CPU_AND_GPU Mat2& operator*=(const Mat2& other) {
			Mat2<T> prod = (*this) * other;
			*this = prod;
			return *this;
		}

		// Multiply each element in the matrix with a scalar factor.
		CPU_AND_GPU Mat2 operator*(T r) const {
			Mat2<T> result;
			for (unsigned int i = 0; i < 4; i++) {
				result.m_matrix[i] = m_matrix[i] * r;
			}
			return result;
		}

		// Multiply each element in the matrix with a scalar factor.
		CPU_AND_GPU Mat2& operator*=(T r) {
			for (unsigned int i = 0; i < 4; i++) {
				m_matrix[i] *= r;
			}
			return *this;
		}

		// Divide the matrix by a scalar factor.
		CPU_AND_GPU Mat2 operator/(T r) const {
			Mat2<T> result;
			for (unsigned int i = 0; i < 4; i++) {
				result.m_matrix[i] = m_matrix[i] / r;
			}
			return result;
		}

		// Divide each element in the matrix with a scalar factor.
		CPU_AND_GPU Mat2& operator/=(T r) {
			for (unsigned int i = 0; i < 4; i++) {
				m_matrix[i] /= r;
			}
			return *this;
		}

		// Transform a 2D-Vector with the matrix.
		CPU_AND_GPU Vec2<T> operator*(const Vec2<T>& v) const {
			return Vec2<T>(
				m_matrix[0] * v[0] + m_matrix[1] * v[1],
				m_matrix[2] * v[0] + m_matrix[3] * v[1]
			);
		}

		// Return the sum of the operand with matrix b.
		CPU_AND_GPU Mat2 operator+(const Mat2& other) const {
			Mat2<T> result;
			for (unsigned int i = 0; i < 4; i++) {
				result.m_matrix[i] = m_matrix[i] + other.m_matrix[i];
			}
			return result;
		}

		// Add matrix other to the operand.
		CPU_AND_GPU Mat2& operator+=(const Mat2& other) {
			for (unsigned int i = 0; i < 4; i++) {
				m_matrix[i] += other.m_matrix[i];
			}
			return *this;
		}

		// Return the difference of the operand with matrix b.
		CPU_AND_GPU Mat2 operator-(const Mat2& other) const {
			Mat2<T> result;
			for (unsigned int i = 0; i < 4; i++) {
				result.m_matrix[i] = m_matrix[i] - other.m_matrix[i];
			}
			return result;
		}

		// Subtract matrix other from the operand.
		CPU_AND_GPU Mat2 operator-=(const Mat2& other) {
			for (unsigned int i = 0; i < 4; i++) {
				m_matrix[i] -= other.m_matrix[i];
			}
			return *this;
		}

		// Return the determinant of the matrix.
		CPU_AND_GPU T det() const {
			return m_matrix[0] * m_matrix[3] - m_matrix[1] * m_matrix[2];
		}

		/**
		 * Indexing operators.
		 */
		// Access element of matrix at the given row and column for constant access.
		CPU_AND_GPU T at(unsigned char row, unsigned char col) const {
			runtime_assert((row<2) && (col<2), "Index out of bounds.");
			return m_matrix[col + row * 2];
		}

		// Access element of matrix at the given row and column.
		CPU_AND_GPU T& at(unsigned char row, unsigned char col) {
			runtime_assert((row<2) && (col<2), "Index out of bounds.");
			return m_matrix[col + row * 2];
		}

		// Access element of matrix at the given row and column for constant access.
		CPU_AND_GPU T operator()(unsigned int row, unsigned int col) const {
			runtime_assert((row<2) && (col<2), "Index out of bounds.");
			return m_matrix[col + row * 2];
		}

		// Access element of matrix at the given row and column.
		CPU_AND_GPU T& operator()(unsigned int row, unsigned int col) {
			runtime_assert((row<2) && (col<2), "Index out of bounds.");
			return m_matrix[col + row * 2];
		}

		// Access i-th element of the matrix for constant access.
		CPU_AND_GPU const T& operator[](unsigned int i) const {
			runtime_assert(i < 4, "Index out of bounds.");
			return m_matrix[i];
		}

		// Access i-th element of the matrix.
		CPU_AND_GPU T& operator[](unsigned int i) {
			runtime_assert(i < 4, "Index out of bounds.");
			return m_matrix[i];
		}

		// Access i-th element of the matrix for constant access at compile-time.
		template<unsigned i>
		CPU_AND_GPU const T& operator[](I<i>) const {
			static_assert(i < 4, "Index out of bounds.");
			return m_matrix[i];
		}

		// Access i-th element of the matrix at compile-time.
		template<unsigned i>
		CPU_AND_GPU T& operator[](I<i>) {
			static_assert(i < 4, "Index out of bounds.");
			return m_matrix[i];
		}

		/**
		 * Getters/setters.
		 */
		// Get the x column out of the matrix.
		CPU_AND_GPU Vec2<T> xcol() const {
			return Vec2<T>(m_matrix[0], m_matrix[2]);
		}

		// Get the y column out of the matrix.
		CPU_AND_GPU Vec2<T> ycol() const {
			return Vec2<T>(m_matrix[1], m_matrix[3]);
		}

		// Get the x row out of the matrix.
		CPU_AND_GPU Vec2<T> xrow() const {
			return Vec2<T>(m_matrix[0], m_matrix[1]);
		}

		// Get the y row out of the matrix.
		CPU_AND_GPU Vec2<T> yrow() const {
			return Vec2<T>(m_matrix[2], m_matrix[3]);
		}

		// Return the inverse matrix; but does not change the current matrix.
		CPU_AND_GPU Mat2 getInverse() const {
			T inv[4];

			inv[0] = m_matrix[3];
			inv[1] = -m_matrix[1];
			inv[2] = -m_matrix[2];
			inv[3] = m_matrix[0];

			T matrixDet = det();
			T matrixDetr = T(1.0) / matrixDet;

			Mat2<T> res;
			for (unsigned int i = 0; i < 4; i++) {
				res.m_matrix[i] = inv[i] * matrixDetr;
			}
			return res;
		}

		CPU_AND_GPU const T* getData() const {
			return &m_matrix[0];
		}

		CPU_AND_GPU T* getData() {
			return &m_matrix[0];
		}

		// Overwrite the current matrix with its inverse.
		CPU_AND_GPU void invert() {
			*this = getInverse();
		}

		// Return the transposed matrix.
		CPU_AND_GPU Mat2 getTranspose() const {
			Mat2<T> result;
			for (unsigned char x = 0; x < 2; x++) {
				result.at(x, 0) = at(0, x);
				result.at(x, 1) = at(1, x);
			}
			return result;
		}

		// Transpose the matrix in place.
		CPU_AND_GPU void transpose() {
			*this = getTranspose();
		}

		CPU_AND_GPU std::string toString(const std::string& separator = ",") const {
			std::string result;
			for (int i = 0; i < 2; i++)
				for (int j = 0; j < 2; j++) {
					result += to_string(m_matrix[i + j * 2]);
					if (i != 1 || j != 1)
						result += separator;
				}
			return result;
		}

	private:
		T m_matrix[4];
	};

	/**
	 * Math operations.
	 */
	template <class T>
	CPU_AND_GPU Mat2<T> operator*(T s, const Mat2<T>& m) {
		return m * s;
	}

	template <class T>
	CPU_AND_GPU Mat2<T> operator/(T s, const Mat2<T>& m) {
		return Mat2<T>(
			s / m(0, 0), s / m(0, 1), 
			s / m(0, 2), s / m(1, 0)
		);
	}

	template <class T>
	CPU_AND_GPU Mat2<T> operator+(T s, const Mat2<T>& m) {
		return m + s;
	}

	template <class T>
	CPU_AND_GPU Mat2<T> operator-(T s, const Mat2<T>& m) {
		return -m + s;
	}

	// Writes to a stream.
	template <class T>
	CPU_AND_GPU std::ostream& operator<<(std::ostream& s, const Mat2<T>& m) {
		return (
			s <<
			m(0, 0) << " " << m(0, 1) << " " << std::endl <<
			m(1, 0) << " " << m(1, 1) << " " << std::endl
		);
	}

	// Reads from a stream.
	template <class T>
	CPU_AND_GPU std::istream& operator>>(std::istream& s, const Mat2<T>& m) {
		return (
			s >>
			m(0, 0) >> m(0, 1) >>
			m(1, 0) >> m(1, 1)
		);
	}

	typedef Mat2<int> Mat2i;
	typedef Mat2<int> Mat2u;
	typedef Mat2<float> Mat2f;
	typedef Mat2<double> Mat2d;

} // namespace matrix_lib 
