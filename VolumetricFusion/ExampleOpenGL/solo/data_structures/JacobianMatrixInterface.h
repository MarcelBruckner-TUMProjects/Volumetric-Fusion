#pragma once
#include <common_utils/meta_structures/BasicTypes.h>

#include "DenseMatrixInterface.h"
#include "ResidualVector.h"
#include "JacobianMatrix.h"

namespace solo {

	/**
	 * Jacobian matrix interface is an interface to Jacobian matrix, which only stores a pointer to the memory.
	 * Therefore it can be use on the GPU as well (if the pointer is pointing to the GPU memory). It includes
	 * some useful operations.
	 */
	template<typename FloatType, typename MemoryStorageType>
	class JacobianMatrixInterface {
	public:
		/**
		 * Constructor.
		 */
		JacobianMatrixInterface(JacobianMatrix<FloatType>& jacobianMatrix) :
			m_nResiduals{ jacobianMatrix.getNumResiduals() },
			m_matrix{ jacobianMatrix.mat() }
		{ }

		/**
		 * Indexing operators (compile-time).
		 */
		template<unsigned residualId, unsigned paramId>
		CPU_AND_GPU FloatType& operator()(unsigned i, I<residualId>, I<paramId>) {
			return (*this)(i, residualId, paramId);
		}

		template<unsigned residualId, unsigned paramId>
		CPU_AND_GPU const FloatType& operator()(unsigned i, I<residualId>, I<paramId>) const {
			return (*this)(i, residualId, paramId);
		}

		/**
		 * Indexing operators (run-time).
		 */
		CPU_AND_GPU FloatType& operator()(unsigned i, unsigned residualId, unsigned paramId) {
			return m_matrix(i + m_nResiduals * residualId, paramId);
		}

		CPU_AND_GPU const FloatType& operator()(unsigned i, unsigned residualId, unsigned paramId) const {
			return m_matrix(i + m_nResiduals * residualId, paramId);
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