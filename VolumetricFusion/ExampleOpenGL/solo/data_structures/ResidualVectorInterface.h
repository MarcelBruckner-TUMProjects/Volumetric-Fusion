#pragma once
#include <common_utils/meta_structures/BasicTypes.h>

#include "DenseMatrixInterface.h"
#include "ResidualVector.h"

namespace solo {

	/**
	 * Residual vector interface is an interface to residual vector, which only stores a pointer to the memory.
	 * Therefore it can be use on the GPU as well (if the pointer is pointing to the GPU memory). It includes
	 * some useful operations.
	 */
	template<typename FloatType, typename MemoryStorageType>
	class ResidualVectorInterface {
	public:
		/**
		 * Constructor.
		 */
		ResidualVectorInterface(ResidualVector<FloatType>& residualVector) :
			m_nResiduals{ residualVector.getNumResiduals() },
			m_matrix{ residualVector.mat() }
		{ }
		
		/**
		 * Indexing operators (compile-time).
		 */
		template<unsigned residualId>
		CPU_AND_GPU FloatType& operator()(unsigned i, I<residualId>) {
			return (*this)(i, residualId);
		}

		template<unsigned residualId>
		CPU_AND_GPU const FloatType& operator()(unsigned i, I<residualId>) const {
			return (*this)(i, residualId);
		}

		/**
		 * Indexing operators (run-time).
		 */
		CPU_AND_GPU FloatType& operator()(unsigned i, unsigned residualId) {
			return m_matrix(i + m_nResiduals * residualId, 0);
		}

		CPU_AND_GPU const FloatType& operator()(unsigned i, unsigned residualId) const {
			return m_matrix(i + m_nResiduals * residualId, 0);
		}

		/**
		 * Getters.
		 */
		CPU_AND_GPU unsigned getNumResiduals() const {
			return m_nResiduals;
		}

		CPU_AND_GPU unsigned getResidualDim() const {
			return m_matrix.rows() / m_nResiduals;
		}

		CPU_AND_GPU unsigned getSize() const {
			return m_matrix.getSize();
		}

		CPU_AND_GPU bool isEmpty() const {
			return m_matrix.getSize() == 0;
		}

	private:
		const unsigned m_nResiduals{ 0 };
		DenseMatrixInterface<FloatType, MemoryStorageType> m_matrix;
	};

} // namespace solo