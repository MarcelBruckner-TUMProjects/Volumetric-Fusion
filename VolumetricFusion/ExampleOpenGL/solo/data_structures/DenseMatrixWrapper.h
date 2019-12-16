#pragma once
#include <common_utils/meta_structures/BasicTypes.h>
#include <common_utils/memory_managment/MemoryWrapper.h>

#include "solo/utils/Common.h"

using namespace common_utils;

namespace solo {
	
	/**
	 * Dense matrix class stores matrices of any dimension in dense format.
	 * It stores elements of type FloatType column-wise in memory.
	 */
	template<typename FloatType>
	class DenseMatrixWrapper {
	public:
		/**
		 * Default constructor.
		 */
		DenseMatrixWrapper() = default;

		/**
		 * Initialization with raw array.
		 */
		template<typename MemoryStorageType>
		DenseMatrixWrapper(FloatType* data, unsigned rows, unsigned cols, Type2Type<MemoryStorageType>) :
			m_rows{ rows },
			m_wrapper{ data, rows * cols, Type2Type<MemoryStorageType>() }
		{ }

		/**
		 * Wraps around raw array.
		 */
		template<typename MemoryStorageType>
		void wrapMemory(FloatType* data, unsigned rows, unsigned cols, Type2Type<MemoryStorageType>) {
			m_rows = rows;
			m_wrapper.wrapMemory(data, rows * cols, Type2Type<MemoryStorageType>());
		}

		/**
		 * Getters.
		 */
		CPU_AND_GPU unsigned rows() const {
			return m_rows;
		}

		CPU_AND_GPU unsigned cols() const {
			return m_rows > 0 ? m_wrapper.getSize() / m_rows : 0;
		}

		CPU_AND_GPU size_t getSize() const {
			return m_wrapper.getSize();
		}

		CPU_AND_GPU size_t getByteSize() const {
			return m_wrapper.getByteSize();
		}

		template<typename MemoryType>
		CPU_AND_GPU FloatType* getData(Type2Type<MemoryType>) {
			return m_wrapper.getData(Type2Type<MemoryType>());
		}

		CPU_AND_GPU MemoryWrapper<FloatType>& getWrapper() {
			return m_wrapper;
		}

	private:
		unsigned                 m_rows{ 0 };
		MemoryWrapper<FloatType> m_wrapper;
	};

} // namespace solo