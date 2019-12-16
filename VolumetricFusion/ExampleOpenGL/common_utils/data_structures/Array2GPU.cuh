#pragma once
#include "common_utils/Common.h"
#include "common_utils/RuntimeAssertion.h"

namespace common_utils {

	/**
	 * 2D array, using row-wise storage.
	 * Only supports GPU/device memory. For CPU/host storage, use Array2 class.
	 * For both CPU/host and GPU/device storage, use Grid2 class. 
	 */
	template<typename T>
	class Array2GPU {
	public:
		/**
		 * Constructors.
		 */
		Array2GPU() = default;
		Array2GPU(unsigned dimX, unsigned dimY) : m_dimX{ dimX }, m_dimY{ dimY } {
			resize(m_dimX, m_dimY);
		}

		/**
		 * Clears the array memory and resets its dimension to (0, 0).
		 */
		void clear() {
			freeDevice();
			m_dimX = 0;
			m_dimY = 0;
		}

		/**
		 * Resizes the array dimensions.
		 */
		void resize(unsigned dimX, unsigned dimY) {
			if (m_dimX != dimX || m_dimY != dimY) {
				m_dimX = dimX;
				m_dimY = dimY;
				freeDevice();
				allocateDevice(m_dimX * m_dimY);
			}
		}

		/**
		 * Element indexing operators.
		 */
		CPU_AND_GPU const T& operator()(unsigned x, unsigned y) const {
			const unsigned idx = dimToIdx(x, y);
			runtime_assert(idx < m_data.size(), "Index out of bounds.");
			return m_data[idx];
		}
		CPU_AND_GPU T& operator()(unsigned x, unsigned y) {
			const unsigned idx = dimToIdx(x, y);
			runtime_assert(idx < m_data.size(), "Index out of bounds.");
			return m_data[idx];
		}

		/**
		 * Getters for dimensions.
		 */
		CPU_AND_GPU unsigned getDimX() const { return m_dimX; }
		CPU_AND_GPU unsigned getDimY() const { return m_dimY; }

		CPU_AND_GPU unsigned getSize() const { return m_dimX * m_dimY; }

		/**
		 * Returns the pointer to the host data.
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
		T*       m_data{ nullptr };

		/**
		 * Converts from 2D-index to 1D-index.
		 */
		CPU_AND_GPU unsigned dimToIdx(unsigned x, unsigned y) const {
			return y * m_dimX + x;
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

	using Array2GPUf = Array2GPU<float>;
	using Array2GPUd = Array2GPU<double>;
	using Array2GPUi = Array2GPU<int>;
	using Array2GPUui = Array2GPU<unsigned>;

} // namespace common_utils