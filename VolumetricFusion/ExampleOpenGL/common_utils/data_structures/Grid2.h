#pragma once
#ifdef ENABLE_SERIALIZATION
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>
#endif

#include "common_utils/memory_managment/MemoryContainer.h"
#include "common_utils/meta_structures/BasicTypes.h"

namespace common_utils {

	/**
	 * 2D grid, using row-wise storage.
	 * It is only a storage container, storing both CPU and GPU memory.
	 * In order to execute any operations on the memory, you can use the Grid2Interface.
	 */
	template<typename T>
	class Grid2 {
	public:
		/**
		 * Constructors.
		 */
		Grid2() = default;
		Grid2(unsigned dimX, unsigned dimY, bool bAllocateHost = true, bool bAllocateDevice = false) : m_dimX{ dimX }, m_dimY{ dimY } {
			allocate(dimX, dimY); 
		}

		/**
		 * Clears the grid memory and resets its dimension to (0, 0, 0).
		 */
		void clear() {
			m_container.clear();
			m_dimX = 0;
			m_dimY = 0;
		}

		/**
		 * Allocates the grid dimensions.
		 */
		void allocate(unsigned dimX, unsigned dimY, bool bAllocateHost = true, bool bAllocateDevice = false) {
			m_container.allocate(dimX * dimY, bAllocateHost, bAllocateDevice);
			m_dimX = dimX;
			m_dimY = dimY;
		}

		/**
		 * Getters for dimensions.
		 */
		unsigned getDimX() const { return m_dimX; }
		unsigned getDimY() const { return m_dimY; }

		unsigned getSize() const { return m_container.getSize(); }

		/**
		 * Getters for memory operations.
		 */
		template<typename MemoryType>
		T* getData(Type2Type<MemoryType>) {
			return m_container.getData(Type2Type<MemoryType>());
		}

		template<typename MemoryType>
		const T* getData(Type2Type<MemoryType>) const {
			return m_container.getData(Type2Type<MemoryType>());
		}

		MemoryContainer<T>& getContainer() {
			return m_container;
		}

		const MemoryContainer<T>& getContainer() const {
			return m_container;
		}

	private:
		unsigned           m_dimX{ 0 };
		unsigned           m_dimY{ 0 };
		MemoryContainer<T> m_container;

#		ifdef ENABLE_SERIALIZATION
		/**
		 * Method for class serialization (any new members need to be added).
		 * We serialize what is currently stored in the CPU/host memory. The user needs to
		 * take care that any changes in the GPU/device memory are saved to the CPU/host
		 * memory to serialize the latest memory version.
		 */
		friend class boost::serialization::access;
		template<class Archive>
		void save(Archive& ar, const unsigned int version) const {
			if (m_container.isAllocatedHost()) {
				ar << m_dimX;
				ar << m_dimY;

				unsigned size = m_container.getSize();
				for (int i = 0; i < size; i++) {
					ar << m_container.getData(Type2Type<MemoryTypeCPU>())[i];
				}
			}
			else {
				ar << 0;
				ar << 0;
			}
		}
		template<class Archive>
		void load(Archive& ar, const unsigned int version) {
			ar >> m_dimX;
			ar >> m_dimY;
			allocate(m_dimX, m_dimY, true, false);

			unsigned size = m_container.getSize();
			for (int i = 0; i < size; i++) {
				ar >> m_container.getData(Type2Type<MemoryTypeCPU>())[i];
			}

			m_container.setUpdated(true, false);
		}
		BOOST_SERIALIZATION_SPLIT_MEMBER();
#		endif
	};

	using Grid2f = Grid2<float>;
	using Grid2d = Grid2<double>;
	using Grid2i = Grid2<int>;
	using Grid2ui = Grid2<unsigned>;

} // namespace common_utils