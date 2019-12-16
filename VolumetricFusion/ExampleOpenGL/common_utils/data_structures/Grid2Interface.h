#pragma once
#include "Grid2.h"

namespace common_utils {

	/**
	 * Grid2 interface is an interface to Grid2 data structure, which only stores a pointer to the memory.
	 * Therefore it can be use on the GPU as well (if the pointer is pointing to the GPU memory). It includes
	 * some useful operations on 2-dimensional grids.
	 */
	template<typename T, typename MemoryStorageType>
	class Grid2Interface {
	public:
		/**
		 * Constructor.
		 */
		Grid2Interface(Grid2<T>& grid2) :
			m_dimX{ grid2.getDimX() }, 
			m_dimY{ grid2.getDimY() },
			m_data{ grid2.getData(Type2Type<MemoryStorageType>()) }
		{ }

		/**
		 * Element indexing operators.
		 */
		CPU_AND_GPU const T& operator()(unsigned x, unsigned y) const {
			const unsigned idx = dimToIdx(x, y);
			runtime_assert(idx < m_dimX * m_dimY, "Index out of bounds.");
			return m_data[idx];
		}
		CPU_AND_GPU T& operator()(unsigned x, unsigned y) {
			const unsigned idx = dimToIdx(x, y);
			runtime_assert(idx < m_dimX * m_dimY, "Index out of bounds.");
			return m_data[idx];
		}

		/**
		 * Sets the given value to all elements of the grid.
		 */
		CPU_AND_GPU void setValue(const T& value) {
			unsigned size = m_dimX * m_dimY;
			for (unsigned i = 0; i < size; ++i)
				m_data[i] = value;
		}

		/**
		 * Getters for dimensions.
		 */
		CPU_AND_GPU unsigned getDimX() const { return m_dimX; }
		CPU_AND_GPU unsigned getDimY() const { return m_dimY; }

		CPU_AND_GPU unsigned getSize() const { return  m_dimX * m_dimY; }

		/**
		 * Getters for raw data.
		 */
		CPU_AND_GPU T* getData() { return m_data; }
		CPU_AND_GPU const T* getData() const { return m_data; }

	private:
		unsigned m_dimX{ 0 };
		unsigned m_dimY{ 0 };
		T*       m_data{ nullptr };

		/**
		 * Converts from 2D-index to 1D-index.
		 */
		CPU_AND_GPU unsigned dimToIdx(unsigned x, unsigned y) const {
			return y * m_dimX + x;
		}
	};

} // namespace common_utils