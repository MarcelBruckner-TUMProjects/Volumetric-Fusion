#pragma once
#include "DenseMatrix.h"
#include "DenseMatrixWrapper.h"
#include "DenseMatrixInterface.h"

namespace solo {

	template<typename MemoryStorageType>
	class IndexMatrix {
	public:
		/**
		 * Constructor from raw array.
		 * The raw array should be stored as column-major. The number of rows should equal
		 * the number of residuals in the constraint. The number of columns should equal the
		 * total dimension of all parameters, present in the constraint.
		 * You can use DenseMatrix class to generate the index matrix, or any other column-major
		 * matrix class.
		 */
		IndexMatrix(int* indexColumnMajorArray) : m_indexArray{ indexColumnMajorArray } {}

		/**
		 * Returns pointer to the underlying array.
		 */
		int* getData() { return m_indexArray; }
		const int* getData() const { return m_indexArray; }

	private:
		int* m_indexArray{ nullptr };
	};

} // namespace solo