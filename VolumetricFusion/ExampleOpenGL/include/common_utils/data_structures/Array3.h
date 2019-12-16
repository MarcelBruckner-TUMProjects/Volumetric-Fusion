#pragma once
#include <vector>

#ifdef ENABLE_SERIALIZATION
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>
#endif

#include "common_utils/RuntimeAssertion.h"

namespace common_utils {
	
	/**
	 * 3D array, using x -> y -> z storage.
	 * Only supports CPU/host memory. For GPU/device storage, use Array3GPU class.
	 * For both CPU/host and GPU/device storage, use Grid3 class. 
	 */
	template<typename T>
	class Array3 {
	public:
		/**
		 * Constructors.
		 */
		Array3() = default;
		Array3(unsigned dimX, unsigned dimY, unsigned dimZ) : m_dimX{ dimX }, m_dimY{ dimY }, m_dimZ{ dimZ }, m_data(dimX * dimY * dimZ) { }
		Array3(unsigned dimX, unsigned dimY, unsigned dimZ, const T& value) : m_dimX{ dimX }, m_dimY{ dimY }, m_dimZ{ dimZ }, m_data(dimX * dimY * dimZ, value) { }

		/**
		 * Clears the grid memory and resets its dimension to (0, 0, 0).
		 */
		void clear() {
			m_data.clear();
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
				m_data.clear();
				m_data.resize(m_dimX * m_dimY * m_dimZ);
			}
		}

		/**
		 * Element indexing operators.
		 */
		const T& operator()(unsigned x, unsigned y, unsigned z) const {
			const unsigned idx = dimToIdx(x, y, z);
			runtime_assert(idx < m_data.size(), "Index out of bounds.");
			return m_data[idx];
		}
		T& operator()(unsigned x, unsigned y, unsigned z) {
			const unsigned idx = dimToIdx(x, y, z);
			runtime_assert(idx < m_data.size(), "Index out of bounds.");
			return m_data[idx];
		}

		/**
		 * Getters for dimensions.
		 */
		unsigned getDimX() const { return m_dimX; }
		unsigned getDimY() const { return m_dimY; }
		unsigned getDimZ() const { return m_dimZ; }

		unsigned getSize() const { return m_data.size(); }

		bool isEmpty() const { return m_data.size() == 0; }
		
		/**
		 * Sets the given value to all elements of the grid.
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
		unsigned  m_dimZ{ 0 };
		std::vector<T> m_data;

#		ifdef ENABLE_SERIALIZATION
		/**
		 * Method for class serialization (any new members need to be added).
		 */
		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive& ar, const unsigned int version) {
			ar & m_dimX;
			ar & m_dimY;
			ar & m_dimZ;
			ar & m_data;
		}
#		endif	

		/**
		 * Converts from 3D-index to 1D-index.
		 */
		unsigned dimToIdx(unsigned x, unsigned y, unsigned z) const {
			return z + y * m_dimZ + x * m_dimZ * m_dimY;
		}
	};

	using Array3f = Array3<float>;
	using Array3d = Array3<double>;
	using Array3i = Array3<int>;
	using Array3ui = Array3<unsigned>;

} // namespace common_utils