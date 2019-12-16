#pragma once
#include <common_utils/RuntimeAssertion.h>
#include <common_utils/meta_structures/BasicOperations.h>

#include "IndexList.h"

namespace solo {

	/**
	 * Sparse vector class stores current gradients for a sparse set of parameters.
	 */
	template <typename FloatType, class IdxList>
	class SparseVec {
	public:
		/**
		 * Default constructor.
		 * All present parameter gradients are always initialized with 1.0.
		 */
		CPU_AND_GPU SparseVec() {
			#pragma unroll
			for (int i = 0; i < IndexListLength<IdxList>::value; ++i) {
				m_data[i] = FloatType(1.0);
			}
		}

		/**
		 * Constructor with FloatType value.
		 * All present parameter gradients are always initialized with given value.
		 */
		CPU_AND_GPU SparseVec(FloatType val) {
			#pragma unroll
			for (int i = 0; i < IndexListLength<IdxList>::value; ++i) {
				m_data[i] = val;
			}
		}

		/**
		 * Assignment operator for storing the residual value.
		 * All non-present parameter gradients are always assigned 0.0.
		 */
		template<typename FT, class OtherIdxList>
		friend class SparseVec;

		template<class OtherIdxList>
		CPU_AND_GPU SparseVec& operator=(const SparseVec<FloatType, OtherIdxList>& other) {
			static_for<IndexListLength<IdxList>::value, AssignElement>(m_data, other);
			return *this;
		}

		template<class OtherIdxList>
		CPU_AND_GPU SparseVec& operator=(SparseVec<FloatType, OtherIdxList>&& other) {
			static_for<IndexListLength<IdxList>::value, AssignElement>(m_data, other);
			return *this;
		}

		/**
		 * Assignment operator for setting all components to the same FloatType value.
		 */
		CPU_AND_GPU SparseVec& operator=(FloatType val) {
			#pragma unroll
			for (int i = 0; i < IndexListLength<IdxList>::value; ++i) {
				m_data[i] = val;
			}
			return *this;
		}

		/**
		 * Unary operators.
		 */
		CPU_AND_GPU SparseVec<FloatType, IdxList> operator+() const {
			return *this;
		}

		CPU_AND_GPU SparseVec<FloatType, IdxList> operator-() const {
			SparseVec<FloatType, IdxList> result;
			#pragma unroll
			for (int i = 0; i < IndexListLength<IdxList>::value; ++i) {
				result.m_data[i] = -m_data[i];
			}
			return result;
		}

		/**
		 * Indexing operation (using indices from index array).
		 */
		template<unsigned i>
		CPU_AND_GPU FloatType& operator[](I<i>) {
			static_assert(LocalPosOfIndex<i, IdxList>::value >= 0, "Index not in the index list.");
			return m_data[LocalPosOfIndex<i, IdxList>::value];
		}

		template<unsigned i>
		CPU_AND_GPU const FloatType& operator[](I<i>) const {
			static_assert(LocalPosOfIndex<i, IdxList>::value >= 0, "Index not in the index list.");
			return m_data[LocalPosOfIndex<i, IdxList>::value];
		}

		/**
		 * Indexing operation with default value (using indices from index array).
		 */
		template<unsigned i>
		CPU_AND_GPU FloatType elementAt(I<i>, FloatType defaultValue) const {
			return queryElementAtIndex(I<i>(), defaultValue, Bool2Type<IndexExists<i, IdxList>::value>());
		}

		/**
		 * Indexing operation (using local indices [0, size - 1]).
		 */
		CPU_AND_GPU FloatType& operator[](unsigned i) {
			runtime_assert(i < IndexListLength<IdxList>::value, "Index out of range.");
			return m_data[i];
		}

		CPU_AND_GPU const FloatType& operator[](unsigned i) const {
			runtime_assert(i < IndexListLength<IdxList>::value, "Index out of range.");
			return m_data[i];
		}

		/**
		 * Getters.
		 */
		CPU_AND_GPU unsigned getSize() const {
			return IndexListLength<IdxList>::value;
		}

	private:
		FloatType m_data[IndexListLength<IdxList>::value];

		/**
		 * Helper methods for querying an element with a default value.
		 */
		template<unsigned i>
		CPU_AND_GPU FloatType queryElementAtIndex(I<i>, FloatType defaultValue, Bool2Type<true>) const {
			static_assert(LocalPosOfIndex<i, IdxList>::value >= 0, "That should not happen.");
			return m_data[LocalPosOfIndex<i, IdxList>::value];
		}

		template<unsigned i>
		CPU_AND_GPU FloatType queryElementAtIndex(I<i>, FloatType defaultValue, Bool2Type<false>) const {
			return defaultValue;
		}

		/**
		 * Helper method to assign all elements from input SparseVec to this SparseVec.
		 */
		struct AssignElement {
			template<int idx, typename OtherIdxList> 
			CPU_AND_GPU static void f(FloatType* data, const SparseVec<FloatType, OtherIdxList>& other) {
				// If no element is present, the gradient is evaluated to 0.0.
				data[idx] = other.elementAt(I<IndexAt<idx, IdxList>::value>(), FloatType(0));
			}
		};
	};


	/**
	 * Executes the addition of the elements at indices of the resulting vector, summing together two given sparse vectors.
	 * If any of the vectors doesn't have an element, the default value 0.0 is used. 
	 */
	template<typename SparseVecResult, typename SparseVec1, typename SparseVec2>
	CPU_AND_GPU void computeSparseVecAdditionHelper(SparseVecResult& res, const SparseVec1& v1, const SparseVec2& v2, Type2Type<NullType>) { }

	template<typename SparseVecResult, typename SparseVec1, typename SparseVec2, unsigned i, typename OtherIndices>
	CPU_AND_GPU void computeSparseVecAdditionHelper(SparseVecResult& res, const SparseVec1& v1, const SparseVec2& v2, Type2Type<IndexList<i, OtherIndices>>) {
		// We sum the current element at index i.
		res[I<i>()] = v1.elementAt(I<i>(), 0.0) + v2.elementAt(I<i>(), 0.0);

		// We recursively sum all the elements at other components.
		computeSparseVecAdditionHelper(res, v1, v2, Type2Type<OtherIndices>());
	}

	template<typename FloatType, typename ResultIndexList, typename SparseVec1, typename SparseVec2>
	CPU_AND_GPU void computeSparseVecAddition(SparseVec<FloatType, ResultIndexList>& res, const SparseVec1& v1, const SparseVec2& v2) {
		computeSparseVecAdditionHelper(res, v1, v2, Type2Type<ResultIndexList>());
	}

	/**
	 * Executes the addition to the elements at indices of the resulting vector, adding scaled elements of the input vector.
	 * The scaling factor alpha should also be provided. If the input vector doesn't have an element at the given index,
	 * nothing is added to the resulting vector.
	 */
	template<typename FloatType, typename ResultIndexList, typename SpVec>
	CPU_AND_GPU void computeSparseVecScaledAdditionHelper(SparseVec<FloatType, ResultIndexList>& res, const SpVec& v, FloatType alpha, Type2Type<NullType>) { }

	template<typename FloatType, typename ResultIndexList, typename SpVec, unsigned i, typename OtherIndices>
	CPU_AND_GPU void computeSparseVecScaledAdditionHelper(SparseVec<FloatType, ResultIndexList>& res, const SpVec& v, FloatType alpha, Type2Type<IndexList<i, OtherIndices>>) {
		// We add the alpha * x the current element at index i.
		res[I<i>()] = res[I<i>()] + alpha * v.elementAt(I<i>(), 0.0);

		// We recursively execute scaled additions all the elements at other components.
		computeSparseVecScaledAdditionHelper(res, v, alpha, Type2Type<OtherIndices>());
	}

	template<typename FloatType, typename ResultIndexList, typename SpVec>
	CPU_AND_GPU void computeSparseVecScaledAddition(SparseVec<FloatType, ResultIndexList>& res, const SpVec& v, FloatType alpha) {
		computeSparseVecScaledAdditionHelper(res, v, alpha, Type2Type<ResultIndexList>());
	}


	/**
	 * Executes the subtraction of the elements at indices of the resulting vector, subtracting two given sparse vectors.
	 * If any of the vectors doesn't have an element, the default value 0.0 is used.
	 */
	template<typename SparseVecResult, typename SparseVec1, typename SparseVec2>
	CPU_AND_GPU void computeSparseVecSubtractionHelper(SparseVecResult& res, const SparseVec1& v1, const SparseVec2& v2, Type2Type<NullType>) { }

	template<typename SparseVecResult, typename SparseVec1, typename SparseVec2, unsigned i, typename OtherIndices>
	CPU_AND_GPU void computeSparseVecSubtractionHelper(SparseVecResult& res, const SparseVec1& v1, const SparseVec2& v2, Type2Type<IndexList<i, OtherIndices>>) {
		// We subtract the current element at index i.
		res[I<i>()] = v1.elementAt(I<i>(), 0.0) - v2.elementAt(I<i>(), 0.0);

		// We recursively subtract all the elements at other components.
		computeSparseVecSubtractionHelper(res, v1, v2, Type2Type<OtherIndices>());
	}

	template<typename FloatType, typename ResultIndexList, typename SparseVec1, typename SparseVec2>
	CPU_AND_GPU void computeSparseVecSubtraction(SparseVec<FloatType, ResultIndexList>& res, const SparseVec1& v1, const SparseVec2& v2) {
		computeSparseVecSubtractionHelper(res, v1, v2, Type2Type<ResultIndexList>());
	}


	/**
	 * Operation definitions.
	 */
	template <typename FloatType, class IdxList1, class IdxList2, typename ResultIndexList = typename JointIndexList<IdxList1, IdxList2>::type>
	CPU_AND_GPU auto operator+(const SparseVec<FloatType, IdxList1>& v1, const SparseVec<FloatType, IdxList2>& v2) -> SparseVec<FloatType, ResultIndexList> {
		SparseVec<FloatType, ResultIndexList> result;
		computeSparseVecAddition(result, v1, v2);
		return result;
	}

	template <typename FloatType, class IdxList1, class IdxList2, typename ResultIndexList = typename JointIndexList<IdxList1, IdxList2>::type>
	CPU_AND_GPU auto operator-(const SparseVec<FloatType, IdxList1>& v1, const SparseVec<FloatType, IdxList2>& v2) -> SparseVec<FloatType, ResultIndexList> {
		SparseVec<FloatType, ResultIndexList> result;
		computeSparseVecSubtraction(result, v1, v2);
		return result;
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU auto operator+(const SparseVec<FloatType, IdxList>& v, FloatType val) -> SparseVec<FloatType, IdxList> {
		SparseVec<FloatType, IdxList> result;
		#pragma unroll
		for (int i = 0; i < IndexListLength<IdxList>::value; ++i) {
			result[i] = v[i] + val;
		}
		return result;
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU auto operator+(FloatType val, const SparseVec<FloatType, IdxList>& v) -> SparseVec<FloatType, IdxList> {
		return v + val;
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU auto operator-(const SparseVec<FloatType, IdxList>& v, FloatType val) -> SparseVec<FloatType, IdxList> {
		SparseVec<FloatType, IdxList> result;
		#pragma unroll
		for (int i = 0; i < IndexListLength<IdxList>::value; ++i) {
			result[i] = v[i] - val;
		}
		return result;
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU auto operator-(FloatType val, const SparseVec<FloatType, IdxList>& v) -> SparseVec<FloatType, IdxList> {
		SparseVec<FloatType, IdxList> result;
		#pragma unroll
		for (int i = 0; i < IndexListLength<IdxList>::value; ++i) {
			result[i] = val - v[i];
		}
		return result;
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU auto operator*(const SparseVec<FloatType, IdxList>& v, FloatType val) -> SparseVec<FloatType, IdxList> {
		SparseVec<FloatType, IdxList> result;
		#pragma unroll
		for (int i = 0; i < IndexListLength<IdxList>::value; ++i) {
			result[i] = v[i] * val;
		}
		return result;
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU auto operator*(FloatType val, const SparseVec<FloatType, IdxList>& v) -> SparseVec<FloatType, IdxList> {
		return v * val;
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU auto operator/(const SparseVec<FloatType, IdxList>& v, FloatType val) -> SparseVec<FloatType, IdxList> {
		SparseVec<FloatType, IdxList> result;
		#pragma unroll
		for (int i = 0; i < IndexListLength<IdxList>::value; ++i) {
			result[i] = v[i] / val;
		}
		return result;
	}

	template <typename FloatType, class IdxList>
	CPU_AND_GPU auto operator/(FloatType val, const SparseVec<FloatType, IdxList>& v) -> SparseVec<FloatType, IdxList> {
		SparseVec<FloatType, IdxList> result;
		#pragma unroll
		for (int i = 0; i < IndexListLength<IdxList>::value; ++i) {
			result[i] = val / v[i];
		}
		return result;
	}


	// TODO: Think about expression templates to reduce the memory allocations (avoid creating object instances
	// after each operation).
	
	//template <typename E>
	//class SparseVecExpression {
	//public:
	//	/* Indexing operator, using local positions. */
	//	double operator[](unsigned i) const { return static_cast<E const&>(*this)[i]; }

	//	/* Overloading operators to access the derived class. */
	//	operator E&() { return static_cast<E&>(*this); }
	//	operator const E&() const { return static_cast<const E&>(*this); }
	//};

	//template <typename FloatType, class IdxList>
	//class SparseVec : public SparseVecExpression<SparseVec<FloatType, IdxList>> {
	//public:
	//	/* Interface implementation. */
	//	FloatType operator[](unsigned i) const { return m_data[i]; }

	//	/* A SparseVec can be constructed from any SparseVecExpression, forcing its evaluation. */
	//	template <typename E>
	//	SparseVec(const SparseVecExpression<E>& expression) {
	//		for (size_t i = 0; i != Length<IdxList>::value; ++i) {
	//			m_data[i] = expression[i];
	//		}
	//	}

	//private:
	//	FloatType m_data[Length<IdxList>::value];
	//};

	//template <typename Operation, typename E1, typename E2>
	//class Expression {
	//public:
	//	Expression(const E1& u, const E2& v) : m_u{ u }, m_v{ v } { }

	//	double operator[](unsigned i) const {
	//		return Operation::apply(m_u[i], m_v[i]);
	//	}

	//private:
	//	using Op = Operation;
	//	const E1& m_u;
	//	const E2& m_v;
	//};

} // namespace solo
