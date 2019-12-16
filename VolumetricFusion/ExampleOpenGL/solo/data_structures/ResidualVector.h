#pragma once
#include <common_utils/meta_structures/BasicTypes.h>

#include "DenseMatrix.h"

namespace solo {
	
	/**
	 * Residual vector stores residual evaluations.
	 * Memory storage is column-wise, with the following dimensions: 
	 * (nResiduals * residualDim) x 1
	 * (n * d) x 1
	 * 
	 * 	   |    | 
	 *  r0 |    | 
	 * 	   |    | 
	 * 	   |    | 
	 *  r1 |    | 
	 * 	   |    | 
	 *  .  |    | 
	 *  .  |    | 
	 *  .  |    | 
	 * 	   |    | 
	 *  rd |    | 
	 * 	   |    | 
	 */
	template<typename FloatType>
	class ResidualVector {
	public:
		ResidualVector() = default;
		ResidualVector(const FloatType* data, unsigned nResiduals, unsigned residualDim) :
			m_nResiduals{ nResiduals },
			m_matrix{ data, nResiduals * residualDim, 1 }
		{ }

		/**
		 * Memory allocation of the given size.
		 * You can pick the type of memory you want to allocate, default type is CPU/host memory.
		 */
		template<typename MemoryType>
		void allocate(unsigned nResiduals, unsigned residualDim, Type2Type<MemoryType>) {
			m_nResiduals = nResiduals;
			m_matrix.allocate(nResiduals * residualDim, 1, Type2Type<MemoryType>());
		}

		void allocate(unsigned nResiduals, unsigned residualDim) {
			allocate(nResiduals, residualDim, Type2Type<MemoryTypeCPU>());
		}

		/**
		 * Getters.
		 */
		CPU_AND_GPU unsigned getNumResiduals() const {
			return m_nResiduals;
		}

		CPU_AND_GPU unsigned getResidualDim() const {
			return m_nResiduals > 0 ? m_matrix.rows() / m_nResiduals : 0;
		}

		CPU_AND_GPU size_t getSize() const {
			return m_matrix.getSize();
		}

		CPU_AND_GPU size_t getByteSize() const {
			return m_matrix.getByteSize();
		}

		CPU_AND_GPU bool isEmpty() const {
			return m_matrix.getSize() == 0;
		}

		CPU_AND_GPU DenseMatrix<FloatType>& mat() {
			return m_matrix;
		}

		CPU_AND_GPU const DenseMatrix<FloatType>& mat() const {
			return m_matrix;
		}

		template<typename MemoryType>
		CPU_AND_GPU FloatType* getData(Type2Type<MemoryType>) {
			return m_matrix.getData(Type2Type<MemoryType>());
		}

		CPU_AND_GPU MemoryContainer<FloatType>& getContainer() {
			return m_matrix.getContainer();
		}

	private:
		unsigned m_nResiduals{ 0 };
		DenseMatrix<FloatType> m_matrix;
	};

} // namespace solo