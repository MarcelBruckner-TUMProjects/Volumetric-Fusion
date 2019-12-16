#pragma once
#include <vector>

#ifdef ENABLE_SERIALIZATION
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>
#endif

#include "common_utils/RuntimeAssertion.h"

namespace common_utils {

	/**
	 * 2D array, using row-wise storage.
	 * Only supports CPU/host memory. For GPU/device storage, use Array2GPU class.
	 * For both CPU/host and GPU/device storage, use Grid2 class. 
	 */
	template<typename T>
	class Array2 {
	public:
		/**
		 * Constructors.
		 */
		Array2() = default;
		Array2(unsigned dimX, unsigned dimY) : m_dimX{ dimX }, m_dimY{ dimY }, m_data(dimX * dimY) { }
		Array2(unsigned dimX, unsigned dimY, const T& value) : m_dimX{ dimX }, m_dimY{ dimY }, m_data(dimX * dimY, value) { }

		/**
		 * Clears the array memory and resets its dimension to (0, 0).
		 */
		void clear() {
			m_data.clear();
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
				m_data.clear();
				m_data.resize(m_dimX * m_dimY);
			}
		}

		/**
		 * Element indexing operators.
		 */
		const T& operator()(unsigned x, unsigned y) const {
			const unsigned idx = dimToIdx(x, y);
			runtime_assert(idx < m_data.size(), "Index out of bounds.");
			return m_data[idx];
		}
		T& operator()(unsigned x, unsigned y) {
			const unsigned idx = dimToIdx(x, y);
			runtime_assert(idx < m_data.size(), "Index out of bounds.");
			return m_data[idx];
		}

		/**
		 * Getters for dimensions.
		 */
		unsigned getDimX() const { return m_dimX; }
		unsigned getDimY() const { return m_dimY; }

		unsigned getSize() const { return m_data.size(); }

		/**
		 * Sets the given value to all elements of the array.
		 */
		void setValue(const T& value) {
			for (unsigned i = 0; i < m_data.size(); ++i)
				m_data[i] = value;
		}

		/**
		 * Returns the pointer to the host data.
		 */
		T* getData() {
			return m_data.data();
		}

		const T* getData() const {
			return m_data.data();
		}

	private:
		unsigned  m_dimX{ 0 };
		unsigned  m_dimY{ 0 };
		std::vector<T> m_data;

#		ifdef ENABLE_SERIALIZATION
		/**
		 * Method for class serialization (any new members need to be added).
		 */
		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive& ar, const unsigned int version) {
			ar & m_dimX;
			ar & m_dim&;
			ar & m_data;
		}
#		endif

		/**
		 * Converts from 2D-index to 1D-index.
		 */
		unsigned dimToIdx(unsigned x, unsigned y) const {
			return y * m_dimX + x;
		}
	};

	using Array2f = Array2<float>;
	using Array2d = Array2<double>;
	using Array2i = Array2<int>;
	using Array2ui = Array2<unsigned>;

} // namespace common_utils