#pragma once
#include <common_utils/meta_structures/BasicTypes.h>
#include <common_utils/memory_managment/MemoryContainer.h>

#include "solo/utils/Common.h"

using namespace common_utils;

namespace solo {
	
	/**
	 * Dense matrix class stores matrices of any dimension in dense format.
	 * It stores elements of type FloatType column-wise in memory.
	 */
	template<typename FloatType>
	class DenseMatrix {
	public:
		/**
		 * Default constructor.
		 */
		DenseMatrix() = default;

		/**
		 * Initialization with raw array.
		 */
		DenseMatrix(const FloatType* data, unsigned rows, unsigned cols) :
			m_rows{ rows },
			m_container{ data, rows * cols }
		{ }

		/**
		 * Indexing operators.
		 * Only read operations are supported. For writing to the matrix, use DenseMatrixInterface.
		 * We need to specify what kind of memory we want to access. The default type is CPU/host memory.
		 */
		template<unsigned col, typename MemoryType>
		CPU_AND_GPU const FloatType& operator()(unsigned row, I<col>, Type2Type<MemoryType>) const {
			return (*this)(row, col, Type2Type<MemoryType>());
		}

		template<unsigned col>
		CPU_AND_GPU const FloatType& operator()(unsigned row, I<col>) const {
			return (*this)(row, col, Type2Type<MemoryTypeCPU>());
		}

		template<typename MemoryType>
		CPU_AND_GPU const FloatType& operator()(unsigned row, unsigned col, Type2Type<MemoryType>) const {
			runtime_assert(col < cols(), "Column id out of range.");
			runtime_assert(row < m_rows, "Row id out of range.");
			return m_container.getElement(col * m_rows + row, Type2Type<MemoryType>());
		}

		CPU_AND_GPU const FloatType& operator()(unsigned row, unsigned col) const {
			return (*this)(row, col, Type2Type<MemoryTypeCPU>());
		}

		/**
		 * Memory allocation of the given size.
		 * You can pick the type of memory you want to allocate, default type is CPU/host memory.
		 */
		template<typename MemoryType>
		void allocate(unsigned rows, unsigned cols, Type2Type<MemoryType>) {
			m_container.allocate(rows * cols, Type2Type<MemoryType>());
			m_rows = rows;
		}

		void allocate(unsigned rows, unsigned cols, bool bAllocateHost = true, bool bAllocateDevice = false) {
			m_container.allocate(rows * cols, bAllocateHost, bAllocateDevice);
			m_rows = rows;
		}

		/**
		 * Clears the allocated memory.
		 */
		void clear() {
			m_container.clear();
			m_rows = 0;
		}

		/**
		 * Getters.
		 */
		CPU_AND_GPU unsigned rows() const {
			return m_rows;
		}

		CPU_AND_GPU unsigned cols() const {
			return m_rows > 0 ? static_cast<unsigned>(m_container.getSize() / m_rows) : 0;
		}

		CPU_AND_GPU size_t getSize() const {
			return m_container.getSize();
		}

		CPU_AND_GPU size_t getByteSize() const {
			return m_container.getByteSize();
		}

		template<typename MemoryType>
		CPU_AND_GPU FloatType* getData(Type2Type<MemoryType>) {
			return m_container.getData(Type2Type<MemoryType>());
		}

		CPU_AND_GPU MemoryContainer<FloatType>& getContainer() {
			return m_container;
		}

	private:
		unsigned                   m_rows{ 0 };
		MemoryContainer<FloatType> m_container;
	};

} // namespace solo