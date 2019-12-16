#pragma once
#include <common_utils/meta_structures/BasicTypes.h>

#include "DenseMatrix.h"

namespace solo {

	/**
	 * Jacobian matrix stores gradient evaluations (for each residual).
	 * Memory storage is column-wise, with the following dimensions: 
	 * (residualDim * nResiduals) x paramDim
	 * (d * n) x m
	 * 
	 *      p0   p1   ...   pm 
	 *	  |    |    |     |    | 
	 * r0 |    |    |     |    |
	 *	  |    |    |     |    |
	 *	  |    |    |     |    | 
	 * r1 |    |    |     |    |
	 *	  |    |    |     |    |	  
	 * .  |    |    |     |    | 
	 * .  |    |    |     |    |
	 * .  |    |    |     |    |	 
	 *	  |    |    |     |    | 
	 * rd |    |    |     |    |
	 *	  |    |    |     |    |
	 */
	template<typename FloatType>
	class JacobianMatrix {
	public:
		JacobianMatrix() = default;
		JacobianMatrix(const FloatType* data, unsigned nResiduals, unsigned residualDim, unsigned paramDim) : 
			m_nResiduals{ nResiduals },
			m_matrix{ data, nResiduals * residualDim, paramDim } 
		{ }

		/**
		 * Memory allocation of the given size.
		 * You can pick the type of memory you want to allocate, default type is CPU/host memory.
		 */
		template<typename MemoryType>
		void allocate(unsigned nResiduals, unsigned residualDim, unsigned paramDim, Type2Type<MemoryType>) {
			m_nResiduals = nResiduals;
			m_matrix.allocate(nResiduals * residualDim, paramDim, Type2Type<MemoryType>());
		}

		void allocate(unsigned nResiduals, unsigned residualDim, unsigned paramDim) {
			allocate(nResiduals, residualDim, paramDim, Type2Type<MemoryTypeCPU>());
		}

		/**
		 * Getters.
		 */
		CPU_AND_GPU unsigned getNumResiduals() const {
			return m_nResiduals;
		}

		CPU_AND_GPU unsigned getParamDim() const {
			return m_matrix.cols();
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