#pragma once
#include "Vec3.h"

namespace matrix_lib {

	/**
	 * 4D vector.
	 */
	template <class T>
	class Vec4 {
	public:
		/**
		 * Constructors and assignment operators.
		 */
		CPU_AND_GPU explicit Vec4(T v) {
			m_array[0] = m_array[1] = m_array[2] = m_array[3] = v;
		}

		// Default constructor (we only set values to zero if we are dealing
		// with fundamental types, otherwise we just use default constructor of
		// a compound type).
		CPU_AND_GPU void reset(std::true_type) {
			m_array[0] = m_array[1] = m_array[2] = m_array[3] = T(0);
		}

		CPU_AND_GPU void reset(std::false_type) { }

		CPU_AND_GPU Vec4() {
			reset(std::is_fundamental<T>::type());
		}

		CPU_AND_GPU Vec4(T x, T y, T z, T w) {
			m_array[0] = x;
			m_array[1] = y;
			m_array[2] = z;
			m_array[3] = w;
		}

		// Initialize with tuple.
		CPU_AND_GPU Vec4(const Tuple<typename TL<T, T, T, T>::type>& t) {
			m_array[0] = t[I<0>()];
			m_array[1] = t[I<1>()];
			m_array[2] = t[I<2>()];
			m_array[3] = t[I<3>()];
		}

		// Initialize with float4.
		CPU_AND_GPU Vec4(const float4& v) {
			m_array[0] = v.x;
			m_array[1] = v.y;
			m_array[2] = v.z;
			m_array[3] = v.w;
		}

		// Initialize with int4.
		CPU_AND_GPU Vec4(const int4& v) {
			m_array[0] = v.x;
			m_array[1] = v.y;
			m_array[2] = v.z;
			m_array[3] = v.w;
		}

		// Initialize with raw array.
		CPU_AND_GPU Vec4(const T* other) {
			m_array[0] = other[0];
			m_array[1] = other[1];
			m_array[2] = other[2];
			m_array[3] = other[3];
		}

		// Copy data from other template type.
		template <class U>
		CPU_AND_GPU Vec4(const Vec4<U>& other) {
			m_array[0] = T(other[0]);
			m_array[1] = T(other[1]);
			m_array[2] = T(other[2]);
			m_array[3] = T(other[3]);
		}

		CPU_AND_GPU explicit Vec4(const Vec3<T>& other, T w = T(1)) {
			m_array[0] = other[0];
			m_array[1] = other[1];
			m_array[2] = other[2];
			m_array[3] = w;
		}

		// Copy constructor.
		CPU_AND_GPU Vec4(const Vec4& other) {
			m_array[0] = other.m_array[0];
			m_array[1] = other.m_array[1];
			m_array[2] = other.m_array[2];
			m_array[3] = other.m_array[3];
		}

		// Move constructor.
		CPU_AND_GPU Vec4(Vec4&& other) {
			m_array[0] = std::move(other.m_array[0]);
			m_array[1] = std::move(other.m_array[1]);
			m_array[2] = std::move(other.m_array[2]);
			m_array[3] = std::move(other.m_array[3]);
		}

		// Copy assignment.
		CPU_AND_GPU Vec4<T>& operator=(const Vec4& other) {
			m_array[0] = other.m_array[0];
			m_array[1] = other.m_array[1];
			m_array[2] = other.m_array[2];
			m_array[3] = other.m_array[3];
			return *this;
		}

		// Move assignment.
		CPU_AND_GPU Vec4<T>& operator=(Vec4&& other) {
			m_array[0] = std::move(other.m_array[0]);
			m_array[1] = std::move(other.m_array[1]);
			m_array[2] = std::move(other.m_array[2]);
			m_array[3] = std::move(other.m_array[3]);
			return *this;
		}

		// Destructor.
		~Vec4() = default;

		/**
		 * Basic operations.
		 */
		CPU_AND_GPU Vec4<T> operator-() const {
			return Vec4<T>(-m_array[0], -m_array[1], -m_array[2], -m_array[3]);
		}

		CPU_AND_GPU Vec4<T> operator+(const Vec4& other) const {
			return Vec4<T>(m_array[0] + other.m_array[0], m_array[1] + other.m_array[1],
			               m_array[2] + other.m_array[2], m_array[3] + other.m_array[3]);
		}

		CPU_AND_GPU Vec4<T> operator+(T val) const {
			return Vec4<T>(m_array[0] + val, m_array[1] + val, m_array[2] + val, m_array[3] + val);
		}

		CPU_AND_GPU void operator+=(const Vec4& other) {
			m_array[0] += other.m_array[0];
			m_array[1] += other.m_array[1];
			m_array[2] += other.m_array[2];
			m_array[3] += other.m_array[3];
		}

		CPU_AND_GPU void operator-=(const Vec4& other) {
			m_array[0] -= other.m_array[0];
			m_array[1] -= other.m_array[1];
			m_array[2] -= other.m_array[2];
			m_array[3] -= other.m_array[3];
		}

		CPU_AND_GPU void operator+=(T val) {
			m_array[0] += val;
			m_array[1] += val;
			m_array[2] += val;
			m_array[3] += val;
		}

		CPU_AND_GPU void operator-=(T val) {
			m_array[0] -= val;
			m_array[1] -= val;
			m_array[2] -= val;
			m_array[3] -= val;
		}

		CPU_AND_GPU void operator*=(T val) {
			m_array[0] *= val;
			m_array[1] *= val;
			m_array[2] *= val;
			m_array[3] *= val;
		}

		CPU_AND_GPU void operator/=(T val) {
			// Optimized version for float/double (doesn't work for int) - assumes compiler statically optimizes if.
			if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
				T inv = (T)1 / val;
				m_array[0] *= inv;
				m_array[1] *= inv;
				m_array[2] *= inv;
				m_array[3] *= inv;
			}
			else {
				m_array[0] /= val;
				m_array[1] /= val;
				m_array[2] /= val;
				m_array[3] /= val;
			}
		}

		CPU_AND_GPU Vec4<T> operator*(T val) const {
			return Vec4<T>(m_array[0] * val, m_array[1] * val, m_array[2] * val, m_array[3] * val);
		}

		CPU_AND_GPU Vec4<T> operator/(T val) const {
			// Optimized version for float/double (doesn't work for int) - assumes compiler statically optimizes if.
			if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
				T inv = (T)1 / val;
				return Vec4<T>(m_array[0] * inv, m_array[1] * inv, m_array[2] * inv, m_array[3] * inv);
			}
			else {
				return Vec4<T>(m_array[0] / val, m_array[1] / val, m_array[2] / val, m_array[3] / val);
			}
		}

		// Cross product (of .xyz).
		CPU_AND_GPU Vec4<T> operator^(const Vec4& other) const {
			return Vec4<T>(m_array[1] * other.m_array[2] - m_array[2] * other.m_array[1],
			               m_array[2] * other.m_array[0] - m_array[0] * other.m_array[2],
			               m_array[0] * other.m_array[1] - m_array[1] * other.m_array[0], T(1));
		}

		// Dot product.
		CPU_AND_GPU T operator|(const Vec4& other) const {
			return (m_array[0] * other.m_array[0] + m_array[1] * other.m_array[1] + m_array[2] *
				other.m_array[2] + m_array[3] * other.m_array[3]);
		}

		CPU_AND_GPU Vec4<T> operator-(const Vec4& other) const {
			return Vec4<T>(m_array[0] - other.m_array[0], m_array[1] - other.m_array[1], m_array[2] - other.m_array[2],
			               m_array[3] - other.m_array[3]);
		}

		CPU_AND_GPU Vec4<T> operator-(T val) const {
			return Vec4<T>(m_array[0] - val, m_array[1] - val, m_array[2] - val, m_array[3] - val);
		}

		CPU_AND_GPU bool operator==(const Vec4& other) const {
			if ((m_array[0] == other.m_array[0]) && (m_array[1] == other.m_array[1]) &&
				(m_array[2] == other.m_array[2]) && (m_array[3] == other.m_array[3])) {
				return true;
			}

			return false;
		}

		CPU_AND_GPU bool operator!=(const Vec4& other) const {
			return !(*this == other);
		}

		CPU_AND_GPU bool isValid() const {
			return (m_array[0] == m_array[0] && m_array[1] == m_array[1] && m_array[2] == m_array[2] && m_array[3] == m_array[3]);
		}

		CPU_AND_GPU bool isFinite() const {
			return std::isfinite(m_array[0]) && std::isfinite(m_array[1]) && std::isfinite(m_array[2]) && std::isfinite(m_array[3]);
		}

		CPU_AND_GPU T lengthSq() const {
			return (m_array[0] * m_array[0] + m_array[1] * m_array[1] + m_array[2] * m_array[2] + m_array[3] * m_array[3]);
		}

		CPU_AND_GPU T length() const {
			return sqrt(lengthSq());
		}

		CPU_AND_GPU static inline Vec4<T> normalize(const Vec4<T>& v) {
			return v.getNormalized();
		}

		CPU_AND_GPU static T distSq(const Vec4& v0, const Vec4& v1) {
			return (
				(v0.m_array[0] - v1.m_array[0]) * (v0.m_array[0] - v1.m_array[0]) +
				(v0.m_array[1] - v1.m_array[1]) * (v0.m_array[1] - v1.m_array[1]) +
				(v0.m_array[2] - v1.m_array[2]) * (v0.m_array[2] - v1.m_array[2]) +
				(v0.m_array[3] - v1.m_array[3]) * (v0.m_array[3] - v1.m_array[3])
			);
		}

		CPU_AND_GPU static T dist(const Vec4& v0, const Vec4& v1) {
			return (v0 - v1).length();
		}

		CPU_AND_GPU void normalize() {
			T val = (T)1.0 / length();
			m_array[0] *= val;
			m_array[1] *= val;
			m_array[2] *= val;
			m_array[3] *= val;
		}

		CPU_AND_GPU Vec4<T> getNormalized() const {
			T val = (T)1.0 / length();
			return Vec4<T>(m_array[0] * val, m_array[1] * val, m_array[2] * val,
				m_array[3] * val);
		}

		CPU_AND_GPU void dehomogenize() {
			m_array[0] /= m_array[3];
			m_array[1] /= m_array[3];
			m_array[2] /= m_array[3];
			m_array[3] /= m_array[3];
		}


		CPU_AND_GPU bool isLinearDependent(const Vec4& other) const {
			T factor = x() / other.x();

			if ((std::fabs(x() / factor - other.x()) + std::fabs(y() / factor - other.y()) +
				std::fabs(z() / factor - other.z()) + std::fabs(w() / factor - other.w())) < 0.00001) {
				return true;
			}
			else {
				return false;
			}
		}

		CPU_AND_GPU T* getData() {
			return &m_array[0];
		}

		CPU_AND_GPU const T* getData() const {
			return &m_array[0];
		}

		/**
		 * Indexing operators.
		 */
		CPU_AND_GPU const T& operator[](int i) const {
			runtime_assert(i < 4, "Index out of bounds.");
			return m_array[i];
		}

		CPU_AND_GPU T& operator[](int i) {
			runtime_assert(i < 4, "Index out of bounds.");
			return m_array[i];
		}

		template<unsigned i>
		CPU_AND_GPU T& operator[](I<i>) {
			static_assert(i < 4, "Index out of bounds.");
			return m_array[i];
		}

		template<unsigned i>
		CPU_AND_GPU const T& operator[](I<i>) const {
			static_assert(i < 4, "Index out of bounds.");
			return m_array[i];
		}

		/**
		 * Getters/setters.
		 */
		CPU_AND_GPU const T& x() const { return m_array[0]; }
		CPU_AND_GPU T& x() { return m_array[0]; }
		CPU_AND_GPU const T& y() const { return m_array[1]; }
		CPU_AND_GPU T& y() { return m_array[1]; }
		CPU_AND_GPU const T& z() const { return m_array[2]; }
		CPU_AND_GPU T& z() { return m_array[2]; }
		CPU_AND_GPU const T& w() const { return m_array[3]; }
		CPU_AND_GPU T& w() { return m_array[3]; }

		CPU_AND_GPU std::string toString(char separator = ' ') const {
			return toString(std::string(1, separator));
		}

		CPU_AND_GPU std::string toString(const std::string& separator) const {
			return std::to_string(x()) + separator + std::to_string(y()) + separator + std::to_string(z()) + separator + std::
				to_string(w());
		}

		CPU_AND_GPU void print() const {
			std::cout << "(" << m_array[0] << " / " << m_array[1] << " / " << m_array[2] <<
				" / " << m_array[3] << " ) " << std::endl;
		}

		/**
		 * Conversion to raw memory type.
		 */
		CPU_AND_GPU float4 r() const {
			return make_float4(m_array[0], m_array[1], m_array[2], m_array[3]);
		}

	private:
		T m_array[4];
	};

	/**
	 * Math operators.
	 */
	template <class T>
	CPU_AND_GPU Vec4<T> operator*(T s, const Vec4<T>& v) {
		return v * s;
	}

	template <class T>
	CPU_AND_GPU Vec4<T> operator/(T s, const Vec4<T>& v) {
		return Vec4<T>(s / v.x(), s / v.y(), s / v.z(), s / v.w());
	}

	template <class T>
	CPU_AND_GPU Vec4<T> operator+(T s, const Vec4<T>& v) {
		return v + s;
	}

	template <class T>
	CPU_AND_GPU Vec4<T> operator-(T s, const Vec4<T>& v) {
		return Vec4<T>(s - v.x(), s - v.y(), s - v.z(), s - v.w());
	}

	// Write a Vec4 to a stream.
	template <class T>
	CPU_AND_GPU std::ostream& operator<<(std::ostream& s, const Vec4<T>& v) {
		return (s << v[0] << " " << v[1] << " " << v[2] << " " << v[3]);
	}

	// Read a Vec4 from a stream.
	template <class T>
	CPU_AND_GPU std::istream& operator>>(std::istream& s, Vec4<T>& v) {
		return (s >> v[0] >> v[1] >> v[2] >> v[3]);
	}

	/**
	 * Comparison operators.
	 */
	template<class T> CPU_AND_GPU bool operator==(const Vec4<T>& lhs, const Vec4<T>& rhs) { return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2] && lhs[3] == rhs[3]; }
	template<class T> CPU_AND_GPU bool operator!=(const Vec4<T>& lhs, const Vec4<T>& rhs) { return !operator==(lhs, rhs); }

	typedef Vec4<double> Vec4d;
	typedef Vec4<float> Vec4f;
	typedef Vec4<int> Vec4i;
	typedef Vec4<short> Vec4s;
	typedef Vec4<short> Vec4us;
	typedef Vec4<unsigned int> Vec4ui;
	typedef Vec4<unsigned char> Vec4uc;
	typedef Vec4<unsigned long long> Vec4ul;
	typedef Vec4<long long> Vec4l;

	/**
	 * Math operations.
	 */
	namespace math_proc {
		template<class T>
		CPU_AND_GPU Vec4i round(const Vec4<T>& f) {
			return Vec4i(round(f.x()), round(f.y()), round(f.z()), round(f.w()));
		}
		template<class T>
		CPU_AND_GPU Vec4i ceil(const Vec4<T>& f) {
			return Vec4i(ceil(f.x()), ceil(f.y()), ceil(f.z()), ceil(f.w()));
		}
		template<class T>
		CPU_AND_GPU Vec4i floor(const Vec4<T>& f) {
			return Vec4i(floor(f.x()), floor(f.y()), floor(f.z()), floor(f.w()));
		}
		template<class T>
		CPU_AND_GPU Vec4<T> abs(const Vec4<T>& p) {
			return Vec4<T>(abs(p.x()), abs(p.y()), abs(p.z()), abs(p.w()));
		}
		template<class T>
		CPU_AND_GPU Vec4<T> sqrt(const Vec4<T>& p) {
			return Vec4<T>(sqrt(p.x()), sqrt(p.y()), sqrt(p.z()), sqrt(p.w()));
		}
		template<class T>
		CPU_AND_GPU Vec4<T> max(const Vec4<T>& p, T v) {
			return Vec4<T>(
				max(p.x(), v),
				max(p.y(), v),
				max(p.z(), v),
				max(p.w(), v));
		}
		template<class T>
		CPU_AND_GPU Vec4<T> max(const Vec4<T>& p, const Vec4<T>& v) {
			return Vec4<T>(
				max(p.x(), v.x()),
				max(p.y(), v.y()),
				max(p.z(), v.z()),
				max(p.w(), v.w()));
		}
		template<class T>
		CPU_AND_GPU Vec4<T> min(const Vec4<T>& p, T v) {
			return Vec4<T>(
				min(p.x(), v),
				min(p.y(), v),
				min(p.z(), v),
				min(p.w(), v));
		}
		template<class T>
		CPU_AND_GPU Vec4<T> min(const Vec4<T>& p, const Vec4<T>& v) {
			return Vec4<T>(
				min(p.x(), v.x()),
				min(p.y(), v.y()),
				min(p.z(), v.z()),
				min(p.w(), v.w()));
		}
		template<class T>
		CPU_AND_GPU bool floatEqual(const Vec4<T>& p0, const Vec4<T>& p1, T eps = (T)0.000001) {
			return
				floatEqual(p0.x(), p1.x(), eps) &&
				floatEqual(p0.y(), p1.y(), eps) &&
				floatEqual(p0.z(), p1.z(), eps) &&
				floatEqual(p0.w(), p1.w(), eps);
		}
	} // namespace math_proc

} // namespace matrix_lib
