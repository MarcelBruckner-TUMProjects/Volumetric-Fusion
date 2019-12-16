#pragma once
#include "Grid3.h"

namespace common_utils {

	/**
	 * Grid3 interface is an interface to Grid3 data structure, which only stores a pointer to the memory.
	 * Therefore it can be use on the GPU as well (if the pointer is pointing to the GPU memory). It includes
	 * some useful operations on 3-dimensional grids.
	 */
	template<typename T, typename MemoryStorageType>
	class Grid3Interface {
	public:
		/**
		 * Constructor.
		 */
		Grid3Interface(Grid3<T>& grid3) :
			m_dimX{ grid3.getDimX() }, 
			m_dimY{ grid3.getDimY() }, 
			m_dimZ{ grid3.getDimZ() },
			m_data{ grid3.getData(Type2Type<MemoryStorageType>()) }
		{ }

		/**
		 * Element indexing operators.
		 */
		CPU_AND_GPU const T& operator()(unsigned x, unsigned y, unsigned z) const {
			const unsigned idx = dimToIdx(x, y, z);
			runtime_assert(idx < m_dimX * m_dimY * m_dimZ, "Index out of bounds.");
			return m_data[idx];
		}
		CPU_AND_GPU T& operator()(unsigned x, unsigned y, unsigned z) {
			const unsigned idx = dimToIdx(x, y, z);
			runtime_assert(idx < m_dimX * m_dimY * m_dimZ, "Index out of bounds.");
			return m_data[idx];
		}

		/**
		 * Sets the given value to all elements of the grid.
		 */
		CPU_AND_GPU void setValue(const T& value) {
			unsigned size = m_dimX * m_dimY * m_dimZ;
			for (unsigned i = 0; i < size; ++i)
				m_data[i] = value;
		}

		/**
		 * Getters for dimensions.
		 */
		CPU_AND_GPU unsigned getDimX() const { return m_dimX; }
		CPU_AND_GPU unsigned getDimY() const { return m_dimY; }
		CPU_AND_GPU unsigned getDimZ() const { return m_dimZ; }

		CPU_AND_GPU unsigned getSize() const { return  m_dimX * m_dimY * m_dimZ; }

		/**
		 * Getters for raw data.
		 */
		CPU_AND_GPU T* getData() { return m_data; }
		CPU_AND_GPU const T* getData() const { return m_data; }

	private:
		unsigned m_dimX{ 0 };
		unsigned m_dimY{ 0 };
		unsigned m_dimZ{ 0 };
		T*       m_data{ nullptr };

		/**
		 * Converts from 3D-index to 1D-index.
		 */
		CPU_AND_GPU unsigned dimToIdx(unsigned x, unsigned y, unsigned z) const {
			return z + y * m_dimZ + x * m_dimZ * m_dimY;
		}
	};

	template<typename MemoryStorageType> using Grid3Interfacef = Grid3Interface<float, MemoryStorageType>;
	template<typename MemoryStorageType> using Grid3Interfaced = Grid3Interface<double, MemoryStorageType>;
	template<typename MemoryStorageType> using Grid3Interfacei = Grid3Interface<int, MemoryStorageType>;
	template<typename MemoryStorageType> using Grid3Interfaceui = Grid3Interface<unsigned, MemoryStorageType>;

} // namespace common_utils