#pragma once
#ifdef ENABLE_SERIALIZATION
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>
#endif

#include "common_utils/memory_managment/MemoryContainer.h"
#include "common_utils/meta_structures/BasicTypes.h"

namespace common_utils {

	/**
	 * 3D grid, using row-wise storage.
	 * It is only a storage container, storing both CPU and GPU memory.
	 * In order to execute any operations on the memory, you can use the Grid3Interface.
	 */
	template<typename T>
	class Grid3 {
	public:
		/**
		 * Constructors.
		 */
		Grid3() = default;
		Grid3(unsigned dimX, unsigned dimY, unsigned dimZ, bool bAllocateHost = true, bool bAllocateDevice = false) : 
			m_dimX{ dimX }, m_dimY{ dimY }, m_dimZ{ dimZ }
		{ 
			allocate(dimX, dimY, dimZ, bAllocateHost, bAllocateDevice);
		}

		/**
		 * Clears the grid memory and resets its dimension to (0, 0, 0).
		 */
		void clear() {
			m_container.clear();
			m_dimX = 0;
			m_dimY = 0;
			m_dimZ = 0;
		}

		/**
		 * Allocates the grid dimensions.
		 */
		void allocate(unsigned dimX, unsigned dimY, unsigned dimZ, bool bAllocateHost = true, bool bAllocateDevice = false) {
			m_container.allocate(dimX * dimY * dimZ, bAllocateHost, bAllocateDevice);
			m_dimX = dimX;
			m_dimY = dimY;
			m_dimZ = dimZ;
		}

		/**
		 * Getters for dimensions.
		 */
		unsigned getDimX() const { return m_dimX; }
		unsigned getDimY() const { return m_dimY; }
		unsigned getDimZ() const { return m_dimZ; }

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
		unsigned           m_dimZ{ 0 };
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
				ar << m_dimZ;

				unsigned size = m_container.getSize();
				for (int i = 0; i < size; i++) {
					ar << m_container.getData(Type2Type<MemoryTypeCPU>())[i];
				}
			}
			else {
				ar << 0;
				ar << 0;
				ar << 0;
			}
		}
		template<class Archive>
		void load(Archive& ar, const unsigned int version) {
			ar >> m_dimX;
			ar >> m_dimY;
			ar >> m_dimZ;
			allocate(m_dimX, m_dimY, m_dimZ, true, false);

			unsigned size = m_container.getSize();
			for (int i = 0; i < size; i++) {
				ar >> m_container.getData(Type2Type<MemoryTypeCPU>())[i];
			}

			m_container.setUpdated(true, false);
		}
		BOOST_SERIALIZATION_SPLIT_MEMBER();
#		endif
	};

	using Grid3f = Grid3<float>;
	using Grid3d = Grid3<double>;
	using Grid3i = Grid3<int>;
	using Grid3ui = Grid3<unsigned>;

} // namespace common_utils