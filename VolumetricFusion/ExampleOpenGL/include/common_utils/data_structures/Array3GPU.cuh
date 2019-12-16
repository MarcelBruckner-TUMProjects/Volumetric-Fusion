#pragma once
#include "common_utils/Common.h"
#include "common_utils/RuntimeAssertion.h"

namespace common_utils {
	
	/**
	 * 3D array, using x -> y -> z storage.
	 * Only supports GPU/device memory. For CPU/host storage, use Array3 class.
	 * For both CPU/host and GPU/device storage, use Grid3 class. 
	 */
	template<typename T>
	class Array3GPU {
	public:
		/**
		 * Constructors.
		 */
		Array3GPU() = default;
		Array3GPU(unsigned dimX, unsigned dimY, unsigned dimZ) : m_dimX{ dimX }, m_dimY{ dimY }, m_dimZ{ dimZ } { 
			resize(dimX, dimY, dimZ);
		}

		/**
		 * Clears the grid memory and resets its dimension to (0, 0, 0).
		 */
		void clear() {
			freeDevice();
			m_dimX = 0;
			m_dimY = 0;
			m_dimZ = 0;
		}

		/**
		 * Resizes the grid dimensions.
		 */
		void resize(unsigned dimX, unsigned dimY, unsigned dimZ) {
			if (m_dimX != dimX || m_dimY != dimY || m_dimZ != dimZ) {
				m_dimX = dimX;
				m_dimY = dimY;
				m_dimZ = dimZ;
				freeDevice();
				allocateDevice(m_dimX * m_dimY * m_dimZ);
			}
		}

		/**
		 * Element indexing operators.
		 */
		CPU_AND_GPU const T& operator()(unsigned x, unsigned y, unsigned z) const {
			const unsigned idx = dimToIdx(x, y, z);
			runtime_assert(idx < m_data.size(), "Index out of bounds.");
			return m_data[idx];
		}
		CPU_AND_GPU T& operator()(unsigned x, unsigned y, unsigned z) {
			const unsigned idx = dimToIdx(x, y, z);
			runtime_assert(idx < m_data.size(), "Index out of bounds.");
			return m_data[idx];
		}

		/**
		 * Getters for dimensions.
		 */
		CPU_AND_GPU unsigned getDimX() const { return m_dimX; }
		CPU_AND_GPU unsigned getDimY() const { return m_dimY; }
		CPU_AND_GPU unsigned getDimZ() const { return m_dimZ; }

		CPU_AND_GPU unsigned getSize() const { return m_dimX * m_dimY * m_dimZ; }

		CPU_AND_GPU bool isEmpty() const { return getSize() == 0; }

		/**
		 * Returns the pointer to the device data.
		 */
		CPU_AND_GPU T* getData() {
			return m_data;
		}

		CPU_AND_GPU const T* getData() const {
			return m_data;
		}

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

		/**
		 * Helper methods to allocate/free device memory.
		 */
		void allocateDevice(unsigned size) {
			runtime_assert(m_data == nullptr, "We can only allocate device memory if previous memory was freed.");
			CUDA_SAFE_CALL(cudaMalloc(&m_data, size * sizeof(T)));
		}

		void freeDevice() {
			if (m_data) {
				CUDA_SAFE_CALL(cudaFree(m_data));
				m_data = nullptr;
			}
		}
	};

	using Array3GPUf = Array3GPU<float>;
	using Array3GPUd = Array3GPU<double>;
	using Array3GPUi = Array3GPU<int>;
	using Array3GPUui = Array3GPU<unsigned>;

} // namespace common_utils