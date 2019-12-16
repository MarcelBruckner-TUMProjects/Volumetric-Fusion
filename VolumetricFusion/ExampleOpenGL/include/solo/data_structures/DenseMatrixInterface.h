#pragma once
#include <common_utils/RuntimeAssertion.h>
#include <common_utils/meta_structures/BasicTypes.h>
#include <common_utils/memory_managment/MemoryContainer.h>

#include "solo/utils/Common.h"
#include "DenseMatrix.h"
#include "DenseMatrixWrapper.h"

namespace solo {

	/**
	 * Dense matrix interface is an interface to dense matrix, which only stores a pointer to the memory.
	 * Therefore it can be use on the GPU as well (if the pointer is pointing to the GPU memory). It includes
	 * some useful operations on dense matrices.
	 */
	template<typename FloatType, typename MemoryStorageType>
	class DenseMatrixInterface {
	public:
		/**
		 * Constructor from dense matrix container.
		 */
		DenseMatrixInterface(DenseMatrix<FloatType>& denseMatrix) :
			m_rows{ denseMatrix.rows() },
			m_cols{ denseMatrix.cols() },
			m_data{ denseMatrix.getContainer().getData(Type2Type<MemoryStorageType>()) }
		{ }

		/**
		 * Constructor from dense matrix wrapper.
		 */
		DenseMatrixInterface(DenseMatrixWrapper<FloatType>& denseMatrix) :
			m_rows{ denseMatrix.rows() },
			m_cols{ denseMatrix.cols() },
			m_data{ denseMatrix.getWrapper().getData(Type2Type<MemoryStorageType>()) }
		{ }

		/**
		 * Indexing operators.
		 * Column index can be given as a compile-time or a run-time variable.
		 * We need to specify what kind of memory we want to access. The default type is CPU/host memory.
		 */
		template<unsigned col>
		CPU_AND_GPU FloatType& operator()(unsigned row, I<col>) {
			return (*this)(row, col);
		}

		template<unsigned col>
		CPU_AND_GPU const FloatType& operator()(unsigned row, I<col>) const {
			return (*this)(row, col);
		}

		CPU_AND_GPU FloatType& operator()(unsigned row, unsigned col) {
			runtime_assert(col < cols(), "Column id out of range.");
			runtime_assert(row < m_rows, "Row id out of range.");
			return m_data[col * m_rows + row];
		}

		CPU_AND_GPU const FloatType& operator()(unsigned row, unsigned col) const {
			runtime_assert(col < cols(), "Column id out of range.");
			runtime_assert(row < m_rows, "Row id out of range.");
			return m_data[col * m_rows + row];
		}

		/**
		 * Getters.
		 */
		CPU_AND_GPU unsigned rows() const {
			return m_rows;
		}

		CPU_AND_GPU unsigned cols() const {
			return m_cols;
		}

		CPU_AND_GPU unsigned getSize() const {
			return m_rows * m_cols;
		}

	private:
		const unsigned m_rows{ 0 };
		const unsigned m_cols{ 0 };
		FloatType*     m_data{ nullptr };
	};

} // namespace solo