#pragma once
#include "DenseMatrix.h"
#include "DenseMatrixWrapper.h"
#include "DenseMatrixInterface.h"

namespace solo {

	template<typename FloatType, typename MemoryStorageType>
	class ParamVector {
	public:
		ParamVector() = delete;

		/**
		 * Constructor from raw array.
		 * The raw array should include all parameters in one contiguous chunk of memory.
		 * The parameter positions in this vector are used when computing the index matrices.
		 * You can use DenseMatrix class to generate the vector, or any other vector class.
		 */
		ParamVector(FloatType* paramArray, unsigned size) : 
			m_paramArray{ paramArray }, 
			m_size{ size }  
		{ }

		/**
		 * Returns pointer to the underlying array.
		 */
		FloatType* getData() { return m_paramArray; }
		const FloatType* getData() const { return m_paramArray; }

		/**
		 * Returns the size of the underlying array.
		 */
		unsigned getSize() const { return m_size; }

	private:
		FloatType* m_paramArray{ nullptr };
		unsigned m_size{ 0 };
	};

} // namespace solo