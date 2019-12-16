#pragma once
#include "SparseVec.h"

namespace solo {
	
	/**
	 * Dual class is the main building block for autodifferentiation. 
	 * By evaluting a function using a dual object, we get the function's value and also all its partial
	 * derivatives.
	 */
	template<typename FloatType, class IdxList>
	class Dual {
	public:
		CPU_AND_GPU Dual() : m_real{ 0 } { }
		CPU_AND_GPU Dual(FloatType real, const SparseVec<FloatType, IdxList>& imag) : m_real{ real }, m_imag{ imag } {	}

		/**
		 * Special method for initialization. The real component is set to the given
		 * value and the imaginary components are all set to 1.0.
		 */
		CPU_AND_GPU void init(FloatType real) {
			m_real = real;
			m_imag = FloatType(1.0);
		}

		/**
		 * Sets the real value to 'real' and all imaginary components to 'imag'.
		 */
		CPU_AND_GPU void set(FloatType real, FloatType imag) {
			m_real = real;
			m_imag = imag;
		}

		/**
		 * Assignment operator with floating-point values. The real component is set
		 * to the given value, while the imaginary components are zeroed out. That means
		 * that no gradient will get propagated, if we assign something to the dual
		 * value.
		 */
		CPU_AND_GPU Dual& operator=(FloatType real) {
			m_real = real;
			m_imag = FloatType(0.0);
			return *this;
		}

		/**
		 * Assignment operator for storing the residual value.
		 */
		template<typename FT, class OtherIdxList>
		friend class Dual;

		template<class OtherIdxList>
		CPU_AND_GPU Dual& operator=(const Dual<FloatType, OtherIdxList>& other) {
			m_real = other.m_real;
			m_imag = other.m_imag;
			return *this;
		}

		template<class OtherIdxList>
		CPU_AND_GPU Dual& operator=(Dual<FloatType, OtherIdxList>&& other) {
			m_real = std::move(other.m_real);
			m_imag = std::move(other.m_imag);
			return *this;
		}

		/**
		 * Getters/setters.
		 */
		CPU_AND_GPU const FloatType& r() const { return m_real; }
		CPU_AND_GPU FloatType& r() { return m_real; }

		CPU_AND_GPU const SparseVec<FloatType, IdxList>& i() const { return m_imag; }
		CPU_AND_GPU SparseVec<FloatType, IdxList>& i() { return m_imag; }

		/**
		 * Unary operations.
		 */
		CPU_AND_GPU Dual<FloatType, IdxList> operator+() const {
			return *this;
		}

		CPU_AND_GPU Dual<FloatType, IdxList> operator-() const {
			return Dual<FloatType, IdxList>{ -m_real, -m_imag };
		}

	private:
		FloatType                     m_real;
		SparseVec<FloatType, IdxList> m_imag;
	};


	/**
	 * Helper method to extract a real component from a dual number or real value
	 * input.
	 */
	template<typename FloatType>
	CPU_AND_GPU FloatType real(const FloatType& realInput) {
		return realInput;
	}

	template<typename FloatType, typename IdxList>
	CPU_AND_GPU FloatType real(const Dual<FloatType, IdxList>& dualInput) {
		return dualInput.r();
	}


	/**
	 * We import the basic operations on FloatType numbers from standard library.
	 */
	// TODO: Add different operations, when CUDA is enabled.
	using std::exp;
	using std::log;
	using std::pow;
	using std::sqrt;
	using std::sin;
	using std::cos;
	using std::tan;
	using std::asin;
	using std::acos;
	using std::atan;
	using std::atan2;
	using std::sinh;
	using std::cosh;
	using std::tanh;
	using std::abs;


	/**
	 * Implementation of common operations on Dual objects. 
	 * Here some of the commonly used C++ operations from http://en.cppreference.com/w/cpp/numeric/math
	 * are implemented. If some other operations on Dual objects are needed, they can be implemented
	 * separately.
	 */
	
	//####################################################################################################
	// Basic operations: +, -, *, /.

	template <typename FloatType, class IdxList1, class IdxList2, typename ResultIndexList = typename JointIndexList<IdxList1, IdxList2>::type>
	CPU_AND_GPU auto operator+(const Dual<FloatType, IdxList1>& d1, const Dual<FloatType, IdxList2>& d2) -> Dual<FloatType, ResultIndexList> {
		return Dual<FloatType, ResultIndexList>{
			d1.r() + d2.r(),
			d1.i() + d2.i()
		};
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU auto operator+(const Dual<FloatType, IdxList>& d, FloatType f) -> Dual<FloatType, IdxList> {
		return Dual<FloatType, IdxList>{
			d.r() + f,
			d.i()
		};
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU auto operator+(FloatType f, const Dual<FloatType, IdxList>& d) -> Dual<FloatType, IdxList> {
		return Dual<FloatType, IdxList>{
			d.r() + f,
			d.i()
		};
	}

	template <typename FloatType, class IdxList1, class IdxList2, typename ResultIndexList = typename JointIndexList<IdxList1, IdxList2>::type>
	CPU_AND_GPU auto operator-(const Dual<FloatType, IdxList1>& d1, const Dual<FloatType, IdxList2>& d2) -> Dual<FloatType, ResultIndexList> {
		return Dual<FloatType, ResultIndexList>{
			d1.r() - d2.r(),
			d1.i() - d2.i()
		};
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU auto operator-(const Dual<FloatType, IdxList>& d, FloatType f) -> Dual<FloatType, IdxList> {
		return Dual<FloatType, IdxList>{
			d.r() - f,
			d.i()
		};
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU auto operator-(FloatType f, const Dual<FloatType, IdxList>& d) -> Dual<FloatType, IdxList> {
		return Dual<FloatType, IdxList>{
			f - d.r(),
			-d.i()
		};
	}

	template <typename FloatType, class IdxList1, class IdxList2, typename ResultIndexList = typename JointIndexList<IdxList1, IdxList2>::type>
	CPU_AND_GPU auto operator*(const Dual<FloatType, IdxList1>& d1, const Dual<FloatType, IdxList2>& d2) -> Dual<FloatType, ResultIndexList> {
		return Dual<FloatType, ResultIndexList>{
			d1.r() * d2.r(),
			d1.r() * d2.i() + d2.r() * d1.i()
		};
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU auto operator*(const Dual<FloatType, IdxList>& d, FloatType f) -> Dual<FloatType, IdxList> {
		return Dual<FloatType, IdxList>{
			f * d.r(),
			f * d.i()
		};
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU auto operator*(FloatType f, const Dual<FloatType, IdxList>& d) -> Dual<FloatType, IdxList> {
		return Dual<FloatType, IdxList>{
			f * d.r(),
			f * d.i()
		};
	}

	template <typename FloatType, class IdxList1, class IdxList2, typename ResultIndexList = typename JointIndexList<IdxList1, IdxList2>::type>
	CPU_AND_GPU auto operator/(const Dual<FloatType, IdxList1>& d1, const Dual<FloatType, IdxList2>& d2) -> Dual<FloatType, ResultIndexList> {
		runtime_assert((d2.r() * d2.r()) > FLT_EPSILON, "Division by zero at Dual division.");

		// (r1 + i1 E) / (r2 + i2 E) = (r1 + i1 E) (r2 - i2 E) / (r2*r2) = (r1 r2 + (i1 * r2 - r1 * i2) E) / (r2 * r2)
		const FloatType r2Inverse = FloatType(1) / d2.r();
		return Dual<FloatType, ResultIndexList>{
			d1.r() * r2Inverse,
			r2Inverse * d1.i() - (d1.r() * r2Inverse * r2Inverse) * d2.i()
		};
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU auto operator/(const Dual<FloatType, IdxList>& d, FloatType f) -> Dual<FloatType, IdxList> {
		runtime_assert(f > FLT_EPSILON, "Division by zero at Dual division.");

		const FloatType fInverse = FloatType(1) / f;
		return Dual<FloatType, IdxList>{
			fInverse * d.r(),
			fInverse * d.i()
		};
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU auto operator/(FloatType f, const Dual<FloatType, IdxList>& d) -> Dual<FloatType, IdxList> {
		runtime_assert((d.r() * d.r()) > FLT_EPSILON, "Division by zero at Dual division.");

		const FloatType rInverse = FloatType(1) / d.r();
		return Dual<FloatType, IdxList>{
			f * rInverse,
			(-f * rInverse * rInverse) * d.i()
		};
	}


	//####################################################################################################
	// Comparison operations.

	template <typename FloatType, class IdxList1, class IdxList2>
	CPU_AND_GPU bool operator==(const Dual<FloatType, IdxList1>& lhs, const Dual<FloatType, IdxList2>& rhs) { return lhs.r() == rhs.r(); }
	
	template <typename FloatType, class IdxList1, class IdxList2>
	CPU_AND_GPU bool operator!=(const Dual<FloatType, IdxList1>& lhs, const Dual<FloatType, IdxList2>& rhs){ return !operator==(lhs, rhs); }
	
	template <typename FloatType, class IdxList1, class IdxList2>
	CPU_AND_GPU bool operator< (const Dual<FloatType, IdxList1>& lhs, const Dual<FloatType, IdxList2>& rhs){ return lhs.r() < rhs.r(); }
	
	template <typename FloatType, class IdxList1, class IdxList2>
	CPU_AND_GPU bool operator> (const Dual<FloatType, IdxList1>& lhs, const Dual<FloatType, IdxList2>& rhs){ return  operator< (rhs, lhs); }
	
	template <typename FloatType, class IdxList1, class IdxList2>
	CPU_AND_GPU bool operator<=(const Dual<FloatType, IdxList1>& lhs, const Dual<FloatType, IdxList2>& rhs){ return !operator> (lhs, rhs); }
	
	template <typename FloatType, class IdxList1, class IdxList2>
	CPU_AND_GPU bool operator>=(const Dual<FloatType, IdxList1>& lhs, const Dual<FloatType, IdxList2>& rhs){ return !operator< (lhs, rhs); }

	template <typename FloatType, class IdxList>
	CPU_AND_GPU bool operator==(const Dual<FloatType, IdxList>& lhs, FloatType rhs) { return lhs.r() == rhs; }

	template <typename FloatType, class IdxList>
	CPU_AND_GPU bool operator!=(const Dual<FloatType, IdxList>& lhs, FloatType rhs) { return !operator==(lhs, rhs); }

	template <typename FloatType, class IdxList>
	CPU_AND_GPU bool operator< (const Dual<FloatType, IdxList>& lhs, FloatType rhs) { return lhs.r() < rhs; }

	template <typename FloatType, class IdxList>
	CPU_AND_GPU bool operator> (const Dual<FloatType, IdxList>& lhs, FloatType rhs) { return  operator< (rhs, lhs); }

	template <typename FloatType, class IdxList>
	CPU_AND_GPU bool operator<=(const Dual<FloatType, IdxList>& lhs, FloatType rhs) { return !operator> (lhs, rhs); }

	template <typename FloatType, class IdxList>
	CPU_AND_GPU bool operator>=(const Dual<FloatType, IdxList>& lhs, FloatType rhs) { return !operator< (lhs, rhs); }

	template <typename FloatType, class IdxList>
	CPU_AND_GPU bool operator==(FloatType lhs, const Dual<FloatType, IdxList>& rhs) { return lhs == rhs.r(); }

	template <typename FloatType, class IdxList>
	CPU_AND_GPU bool operator!=(FloatType lhs, const Dual<FloatType, IdxList>& rhs) { return !operator==(lhs, rhs); }

	template <typename FloatType, class IdxList>
	CPU_AND_GPU bool operator< (FloatType lhs, const Dual<FloatType, IdxList>& rhs) { return lhs < rhs.r(); }

	template <typename FloatType, class IdxList>
	CPU_AND_GPU bool operator> (FloatType lhs, const Dual<FloatType, IdxList>& rhs) { return  operator< (rhs, lhs); }

	template <typename FloatType, class IdxList>
	CPU_AND_GPU bool operator<=(FloatType lhs, const Dual<FloatType, IdxList>& rhs) { return !operator> (lhs, rhs); }

	template <typename FloatType, class IdxList>
	CPU_AND_GPU bool operator>=(FloatType lhs, const Dual<FloatType, IdxList>& rhs) { return !operator< (lhs, rhs); }


	//####################################################################################################
	// Exponential functions.

	template <typename FloatType, class IdxList>
	CPU_AND_GPU Dual<FloatType, IdxList> exp(const Dual<FloatType, IdxList>& d) {
		const FloatType rExp = exp(d.r());
		return Dual<FloatType, IdxList>{
			rExp,
			rExp * d.i()
		};
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU Dual<FloatType, IdxList> log(const Dual<FloatType, IdxList>& d) {
		return Dual<FloatType, IdxList>{
			log(d.r()),
			(FloatType(1) / d.r()) * d.i()
		};
	}


	//####################################################################################################
	// Power functions.

	template <typename FloatType, class IdxList>
	CPU_AND_GPU Dual<FloatType, IdxList> pow(const Dual<FloatType, IdxList>& d, FloatType f) {
		return Dual<FloatType, IdxList>{
			pow(d.r(), f),
			(f * pow(d.r(), f - FloatType(1))) * d.i()
		};
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU Dual<FloatType, IdxList> sqrt(const Dual<FloatType, IdxList>& d) {
		return Dual<FloatType, IdxList>{
			sqrt(d.r()),
			(FloatType(1) / (sqrt(d.r() * FloatType(2)))) * d.i()
		};
	}


	//####################################################################################################
	// Trigonometric functions.

	template <typename FloatType, class IdxList>
	CPU_AND_GPU Dual<FloatType, IdxList> sin(const Dual<FloatType, IdxList>& d) {
		return Dual<FloatType, IdxList>{
			sin(d.r()),
			cos(d.r()) * d.i()
		};
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU Dual<FloatType, IdxList> cos(const Dual<FloatType, IdxList>& d) {
		return Dual<FloatType, IdxList>{
			cos(d.r()),
			(-sin(d.r())) * d.i()
		};
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU Dual<FloatType, IdxList> tan(const Dual<FloatType, IdxList>& d) {
		const FloatType rTan = tan(d.r());
		return Dual<FloatType, IdxList>{
			rTan,
			(FloatType(1) + rTan * rTan) * d.i()
		};
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU Dual<FloatType, IdxList> asin(const Dual<FloatType, IdxList>& d) {
		return Dual<FloatType, IdxList>{
			asin(d.r()),
			(FloatType(1) / sqrt(FloatType(1) - d.r() * d.r())) * d.i()
		};
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU Dual<FloatType, IdxList> acos(const Dual<FloatType, IdxList>& d) {
		return Dual<FloatType, IdxList>{
			acos(d.r()),
			(-FloatType(1) / sqrt(FloatType(1) - d.r() * d.r())) * d.i()
		};
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU Dual<FloatType, IdxList> atan(const Dual<FloatType, IdxList>& d) {
		return Dual<FloatType, IdxList>{
			atan(d.r()),
			(FloatType(1) / (FloatType(1) + d.r() * d.r())) * d.i()
		};
	}

	template <typename FloatType, class IdxList1, class IdxList2, typename ResultIndexList = typename JointIndexList<IdxList1, IdxList2>::type>
	CPU_AND_GPU auto atan2(const Dual<FloatType, IdxList1>& d2, const Dual<FloatType, IdxList2>& d1) -> Dual<FloatType, ResultIndexList> {
		return Dual<FloatType, ResultIndexList>{
			atan2(d2.r(), d1.r()),
			(FloatType(1) / (d1.r() * d1.r() + d2.r() * d2.r())) * (d1.r() * d2.i() - d2.r() * d1.i())
		};
	}


	//####################################################################################################
	// Hyperbolic functions.

	template <typename FloatType, class IdxList>
	CPU_AND_GPU Dual<FloatType, IdxList> sinh(const Dual<FloatType, IdxList>& d) {
		return Dual<FloatType, IdxList>{
			sinh(d.r()),
			cosh(d.r()) * d.i()
		};
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU Dual<FloatType, IdxList> cosh(const Dual<FloatType, IdxList>& d) {
		return Dual<FloatType, IdxList>{
			cosh(d.r()),
			sinh(d.r()) * d.i()
		};
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU Dual<FloatType, IdxList> tanh(const Dual<FloatType, IdxList>& d) {
		const FloatType rTanh = tanh(d.r());
		return Dual<FloatType, IdxList>{
			rTanh,
			(FloatType(1) - rTanh * rTanh) * d.i()
		};
	}


	//####################################################################################################
	// Absolute value.
	
	template <typename FloatType, class IdxList>
	CPU_AND_GPU Dual<FloatType, IdxList> abs(const Dual<FloatType, IdxList>& d) {
		return d >= FloatType(0) ? d : -d;
	}

} // namespace solo