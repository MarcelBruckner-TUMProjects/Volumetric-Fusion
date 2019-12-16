#pragma once
#include "TypeList.h"
#include "Tuple.h"
#include "common_utils/memory_managment/MemoryContainer.h"
#include "BasicOperations.h"

namespace common_utils {

	/**
	 * The Structure of Arrays class.
	 * The implementation of specializations it given below.
	 */
	template<bool isInterface, typename TList>
	class SoA;


	/**
	 * Converts a class that derives from some SoA<isInterface, TList> to SoA<isInterface, TList>.
	 */
	template<bool isInterface, typename TList>
	Type2Type<SoA<isInterface, TList>> getSoAType(const volatile SoA<isInterface, TList>*);

	template<typename DerivedType>
	struct SoAType {
		using type = typename ExtractType<decltype(getSoAType(std::declval<DerivedType*>()))>::type;
	};

	template<typename SoAType>
	struct ExtractSoATypeList;

	template<bool isInterface, typename TList>
	struct ExtractSoATypeList<SoA<isInterface, TList>> {
		using type = TList;
	};

	template<typename SoAType>
	struct ExtractSoAIsInterface;

	template<bool isInterface, typename TList>
	struct ExtractSoAIsInterface<SoA<isInterface, TList>> {
		using type = Bool2Type<isInterface>;
	};


	/**
	 * Checks whether the base of the given class is SoA, or not.
	 */
	template <bool isInterface, typename TList>
	Bool2Type<true> isBaseSoA(const volatile SoA<isInterface, TList>*);
	Bool2Type<false> isBaseSoA(const volatile void*);

	template <typename DerivedType>
	struct IsDerivedFromSoA {
		using type = decltype(isBaseSoA(std::declval<DerivedType*>()));
	};


	/**
	 * Helper struct to add pointers to classes in a list.
	 * We add pointers only for base types, the non-base types are SoA types, therefore they
	 * contain raw pointers, assigned recursively.
	 */
	template<typename Derived, typename BoolType>
	struct CreatePointerTypeListHelper;

	template<typename Derived>
	struct CreatePointerTypeListHelper<Derived, Bool2Type<false>> {
		using type = Derived*;
	};

	template<typename Derived>
	struct CreatePointerTypeListHelper<Derived, Bool2Type<true>> {
		using type = Derived;
	};

	template<typename TList>
	struct CreatePointerTypeList;

	template<>
	struct CreatePointerTypeList<NullType> {
		using type = NullType;
	};

	template<typename T1, typename T2>
	struct CreatePointerTypeList<TypeList<T1, T2>> {
		using type = TypeList<
			typename CreatePointerTypeListHelper<T1, typename IsDerivedFromSoA<T1>::type>::type,
			typename CreatePointerTypeList<T2>::type
		>;
	};


	/**
	 * Helper struct to create a complete pointer list of a list of base classes and also SoA 
	 * classes (which have to be traversed recursively, in order to get a sequence of all base
	 * pointers.
	 */
	template<typename TList>
	struct CreateBasePointerTypeList;

	template<typename Derived, typename BoolType>
	struct CreateBasePointerTypeListHelper;

	template<typename Derived>
	struct CreateBasePointerTypeListHelper<Derived, Bool2Type<false>> {
		using type = TypeList<Derived*, NullType>;
	};

	template<typename Derived>
	struct CreateBasePointerTypeListHelper<Derived, Bool2Type<true>> {
		using type = typename CreateBasePointerTypeList<
			typename ExtractSoATypeList<typename SoAType<Derived>::type>::type
		>::type;
	};

	template<>
	struct CreateBasePointerTypeList<NullType> {
		using type = NullType;
	};

	template<typename T1, typename T2>
	struct CreateBasePointerTypeList<TypeList<T1, T2>> {
		using type = typename MergedTypeList<
			typename CreateBasePointerTypeListHelper<
				T1, typename IsDerivedFromSoA<T1>::type
			>::type,
			typename CreateBasePointerTypeList<T2>::type
		>::type;
	};


	/**
	 * Helper struct to create MemoryContainer type from a base type list.
	 * We create memory containers only for base types. The SoA types will recursively create
	 * memory containers.
	 */
	template<typename Derived, typename BoolType>
	struct CreateMemoryContainerTypeListHelper;

	template<typename Derived>
	struct CreateMemoryContainerTypeListHelper<Derived, Bool2Type<false>> {
		using type = MemoryContainer<Derived>;
	};

	template<typename Derived>
	struct CreateMemoryContainerTypeListHelper<Derived, Bool2Type<true>> {
		using type = Derived;
	};

	template<typename TList>
	struct CreateMemoryContainerTypeList;

	template<>
	struct CreateMemoryContainerTypeList<NullType> {
		using type = NullType;
	};

	template<typename T1, typename T2>
	struct CreateMemoryContainerTypeList<TypeList<T1, T2>> {
		using type = TypeList<
			typename CreateMemoryContainerTypeListHelper<T1, typename IsDerivedFromSoA<T1>::type>::type,
			typename CreateMemoryContainerTypeList<T2>::type
		>;
	};


	/**
	 * Helper methods to compute the size (i.e. the number of base pointers, taking into
	 * account all children) of a Structure of Arrays (SoA) and its elements.
	 */
	template<typename Element>
	struct ElementSize;

	template<typename SoATList>
	struct ElementListSize;

	template<typename Derived, typename BoolType>
	struct ElementSizeHelper;

	template<typename Derived>
	struct ElementSizeHelper<Derived, Bool2Type<false>> {
		enum {
			value = 1
		};
	};

	template<typename Derived>
	struct ElementSizeHelper<Derived, Bool2Type<true>> {
		enum {
			value = ElementListSize<
				typename ExtractSoATypeList<typename SoAType<Derived>::type>::type
			>::value
		};
	};

	template<typename T1, typename T2>
	struct ElementListSize<TypeList<T1, T2>> {
		enum {
			value = ElementSize<T1>::value + ElementListSize<T2>::value
		};
	};

	template<>
	struct ElementListSize<NullType> {
		enum {
			value = 0
		};
	};

	template<typename Element>
	struct ElementSize {
		enum {
			value = ElementSizeHelper<Element, typename IsDerivedFromSoA<Element>::type>::value
		};
	};

	/**
	 * Returns the size (the total number of pointers) of the SoA object. 
	 */
	template<typename DerivedFromSoA>
	struct SoASize {
		enum {
			value = ElementSize<typename SoAType<DerivedFromSoA>::type>::value
		};
	};


	/**
	 * Returns the pointer list of the SoA, as a tuple of base pointers.
	 */
	template<typename SoA>
	struct SoAPointerListHelper;

	template<bool isInterface, typename TList>
	struct SoAPointerListHelper<SoA<isInterface, TList>> {
		using type = typename SoA<isInterface, TList>::BasePointerTupleType;
	};

	template<typename DerivedFromSoA>
	struct SoAPointerList {
		using type = typename SoAPointerListHelper<typename SoAType<DerivedFromSoA>::type>::type;
	};


	/**
	 * Interface type of SoA.
	 */
	template<typename TList>
	class SoA<true, TList> {
	public:
		using PointerTupleType = Tuple<typename CreatePointerTypeList<TList>::type>;
		using BasePointerTupleType = Tuple<typename CreateBasePointerTypeList<TList>::type>;

		/**
		 * Wraps around the data, stored in the pointers of the given tuple.
		 * It copies the pointers from the tuple to the (hierarchial) structure of arrays.
		 * If the offset is given, then the relevant pointers start at a particular index.
		 */
		template<typename GlobalPointers, unsigned Offset>
		CPU_AND_GPU void wrap(const Tuple<GlobalPointers>& globalPointerTuple, Unsigned2Type<Offset>) {
			initializePointers(globalPointerTuple, m_pointerTuple, Unsigned2Type<Offset>(), Unsigned2Type<0>());
		}

		template<typename GlobalPointers>
		CPU_AND_GPU void wrap(const Tuple<GlobalPointers>& globalPointerTuple) {
			initializePointers(globalPointerTuple, m_pointerTuple, Unsigned2Type<0>(), Unsigned2Type<0>());
		}

	protected:
		CPU_AND_GPU const PointerTupleType& t() const { return m_pointerTuple; }
		CPU_AND_GPU PointerTupleType& t() { return m_pointerTuple; }

	private:
		PointerTupleType m_pointerTuple;

		/**
		 * Helper methods for wrapping around a tuple of base pointers.
		 */
		template<typename SubSoA, typename GlobalPointers, unsigned GlobalOffset, int LocalIdx>
		CPU_AND_GPU void assignLocalPointer(
			SubSoA& subArray,
			const Tuple<GlobalPointers>& globalPointerTuple,
			Unsigned2Type<GlobalOffset>,
			Unsigned2Type<LocalIdx>,
			Bool2Type<true>
		) {
			// We recursively initialize all pointers in the sub array.
			subArray.wrap(globalPointerTuple, Unsigned2Type<GlobalOffset>());
		}

		template<typename BasePointer, typename GlobalPointers, unsigned GlobalOffset, int LocalIdx>
		CPU_AND_GPU void assignLocalPointer(
			BasePointer& basePointer,
			const Tuple<GlobalPointers>& globalPointerTuple,
			Unsigned2Type<GlobalOffset>,
			Unsigned2Type<LocalIdx>,
			Bool2Type<false>
		) {
			basePointer = globalPointerTuple[I<GlobalOffset>()];
		}

		template<typename GlobalPointers, typename LocalPointers, unsigned GlobalOffset, int LocalIdx>
		CPU_AND_GPU void initializePointers(
			const Tuple<GlobalPointers>& globalPointerTuple,
			Tuple<LocalPointers>& localPointerTuple,
			Unsigned2Type<GlobalOffset>,
			Unsigned2Type<LocalIdx>
		) {
			// We assign a local pointer to the current local index, and continue for the rest. 
			// The global offset needs to be increment with the size of the element at the current
			// index.
			assignLocalPointer(
				localPointerTuple[I<LocalIdx>()], globalPointerTuple, 
				Unsigned2Type<GlobalOffset>(), Unsigned2Type<LocalIdx>(),
				typename IsDerivedFromSoA<typename TypeAt<LocalIdx, TList>::type>::type()
			);
			initializePointers(
				globalPointerTuple, 
				localPointerTuple, 
				Unsigned2Type<GlobalOffset + ElementSize<typename TypeAt<LocalIdx, TList>::type>::value>(),
				Unsigned2Type<LocalIdx + 1>()
			);
		}

		template<typename GlobalPointers, typename LocalPointers, unsigned GlobalOffset>
		CPU_AND_GPU void initializePointers(
			const Tuple<GlobalPointers>& globalPointerTuple,
			Tuple<LocalPointers>& localPointerTuple,
			Unsigned2Type<GlobalOffset>,
			Unsigned2Type<TypeListLength<TList>::value>
		) {
			// We are done when the local index reaches the tuple size.
		}
	};


	/**
	 * Storage type of SoA.
	 */
	template<typename TList>
	class SoA<false, TList> {
	public:
		using PointerTupleType = Tuple<typename CreatePointerTypeList<TList>::type>;
		using BasePointerTupleType = Tuple<typename CreateBasePointerTypeList<TList>::type>;
		using ContainerTupleType = Tuple<typename CreateMemoryContainerTypeList<TList>::type>;

		/**
		 * Releases the memory, associated with the SoA.
		 */
		void clear() {
			allocate(0, true, true);
		}

		/**
		 * Allocated the given size of host (CPU) or device (GPU) memory for every element type 
		 * in the SoA (recursive allocation is also triggered).
		 * By default, the only the host memory is allocated.
		 */
		void allocate(unsigned size, bool bAllocateHost = true, bool bAllocateDevice = false) {
			if (bAllocateHost) 
				static_for<TupleSize<ContainerTupleType>::value, AllocateElement>(m_containerTuple, size, Type2Type<MemoryTypeCPU>());
			
			#ifdef COMPILE_CUDA
			if (bAllocateDevice)
				static_for<TupleSize<ContainerTupleType>::value, AllocateElement>(m_containerTuple, size, Type2Type<MemoryTypeCUDA>());
			#else
			runtime_assert(!bAllocateDevice, "Device memory of SoA can only be allocated if compiled with flag COMPILE_CUDA.");
			#endif
		}

		/**
		 * Helper method for recursive allocation.
		 * Should NOT be called directly.
		 */
		template<typename MemoryStorageType>
		void allocate(unsigned size, Type2Type<MemoryStorageType>) {
			static_for<TupleSize<ContainerTupleType>::value, AllocateElement>(m_containerTuple, size, Type2Type<MemoryStorageType>());
		}

		/**
		 * Copies the data from one memory type to another (host = CPU, device = CUDA).
		 * Important: Can be called only from CUDA code.
		 */
		void copyHostToDevice() {
			static_for<TupleSize<ContainerTupleType>::value, CopyHostToDeviceElement>(m_containerTuple);	
		}

		void copyDeviceToHost() {
			static_for<TupleSize<ContainerTupleType>::value, CopyDeviceToHostElement>(m_containerTuple);
		}

		/**
		 * We can mark the current status of the memory with update flags. If a certain kind of memory
		 * (CPU or CUDA) is updated, it means it contains the most recent version of the data.
		 */
		void setUpdated(bool bUpdatedHost = false, bool bUpdatedDevice = false) {
			m_bUpdatedHost = bUpdatedHost;
			m_bUpdatedDevice = bUpdatedDevice;
		}

		bool isUpdatedHost() const { return m_bUpdatedHost; }
		bool isUpdatedDevice() const { return m_bUpdatedDevice; }

		/**
		 * Methods to automatically update the host/device memory, if required. They also update the
		 * flags after the update.
		 */
		void updateHostIfNeeded() {
			if (!isUpdatedHost()) {
#				ifdef COMPILE_CUDA
				copyDeviceToHost();
#				endif
				setUpdated(true, true);
			}
		}

		void updateDeviceIfNeeded() {
			if (!isUpdatedDevice()) {
#				ifdef COMPILE_CUDA
				copyHostToDevice();
#				endif
				setUpdated(true, true);
			}
		}

		/**
		 * Returns the size of the data (it is assumed all SoA parts have the same size).
		 */
		CPU_AND_GPU size_t getSize() const {
			return getSizeOfFirstElement(m_containerTuple, Int2Type<TupleSize<ContainerTupleType>::value>());
		}

		/**
		 * Returns the byte size of the data.
		 */
		CPU_AND_GPU size_t getByteSize() const {
			size_t totalSize{ 0 };
			static_for<TupleSize<ContainerTupleType>::value, GetByteSizeElement>(m_containerTuple, totalSize);
			return totalSize;
		}

		/**
		 * Returns a tuple of base pointers for every element type in the SoA (also the pointers
		 * of types in sub SoA structures). Additional flag decides whether the pointers should
		 * point to host or device memory (by default host memory is used).
		 */
		BasePointerTupleType getPointerList() {
			return getPointerList(Type2Type<MemoryTypeCPU>());
		}

		template<typename MemoryStorageType>
		BasePointerTupleType getPointerList(Type2Type<MemoryStorageType>) {
			BasePointerTupleType pointerListTuple;
			storePointers(pointerListTuple, m_containerTuple, Unsigned2Type<0>(), Unsigned2Type<0>(), Type2Type<MemoryStorageType>());
			return pointerListTuple;
		}

		/**
		 * Computes the pointer list.  
		 * Should NOT be called, it's here just for implementation reasons.
		 */
		template<typename GlobalPointers, unsigned Offset, typename MemoryStorageType>
		void computePointerList(Tuple<GlobalPointers>& globalPointerTuple, Unsigned2Type<Offset>, Type2Type<MemoryStorageType>) {
			storePointers(globalPointerTuple, m_containerTuple, Unsigned2Type<Offset>(), Unsigned2Type<0>(), Type2Type<MemoryStorageType>());
		}

	protected:
		const PointerTupleType& t() const { 
			throw std::runtime_error("The member variables should NOT be queried on storage SoA!");
		}
		PointerTupleType& t() {
			throw std::runtime_error("The member variables should NOT be queried on storage SoA!");
		}

	private:
		ContainerTupleType m_containerTuple;

		bool m_bUpdatedHost{ false };
		bool m_bUpdatedDevice{ false };

		/**
		 * Helper method for allocating a particular element of SoA.
		 */
		struct AllocateElement {
			template<int idx, typename ContainerTuple, typename MemoryStorageType> 
			static void f(ContainerTuple& containerTuple, unsigned size, Type2Type<MemoryStorageType>) {
				containerTuple[I<idx>()].allocate(size, Type2Type<MemoryStorageType>());
			}
		};

		/**
		 * Helper methods for copying memory of a particular element of SoAk, from host to device or vice-versa.
		 */
		struct CopyHostToDeviceElement {
			template<int idx, typename ContainerTuple>
			static void f(ContainerTuple& containerTuple) {
				containerTuple[I<idx>()].copyHostToDevice();
			}
		};

		struct CopyDeviceToHostElement {
			template<int idx, typename ContainerTuple>
			static void f(ContainerTuple& containerTuple) {
				containerTuple[I<idx>()].copyDeviceToHost();
			}
		};

		/**
		 * Computes the size of the first element of SoA.
		 */
		template<typename ContainerTuple>
		size_t getSizeOfFirstElement(ContainerTuple& containerTuple, Int2Type<0>) const {
			return 0;
		}

		template<typename ContainerTuple, int SizeOfSoA>
		size_t getSizeOfFirstElement(ContainerTuple& containerTuple, Int2Type<SizeOfSoA>) const {
			return containerTuple[I<0>()].getSize();
		}

		/**
		 * Computes the byte size of the allocated memory.
		 */
		struct GetByteSizeElement {
			template<int idx, typename ContainerTuple>
			static void f(ContainerTuple& containerTuple, size_t& totalSize) {
				totalSize += containerTuple[I<idx>()].getByteSize();
			}
		};

		/**
		 * Helper methods for construction of a tuple of base pointers.
		 */
		template<typename SubSoA, typename GlobalPointers, unsigned GlobalOffset, int LocalIdx, typename MemoryStorageType>
		void assignGlobalPointer(
			SubSoA& subSoA,
			Tuple<GlobalPointers>& globalPointerTuple,
			Unsigned2Type<GlobalOffset>,
			Unsigned2Type<LocalIdx>,
			Type2Type<MemoryStorageType>,
			Bool2Type<true>
		) {
			// We recursively assign all pointers in the sub array.
			subSoA.computePointerList(globalPointerTuple, Unsigned2Type<GlobalOffset>(), Type2Type<MemoryStorageType>());
		}

		template<typename BaseContainer, typename GlobalPointers, unsigned GlobalOffset, int LocalIdx, typename MemoryStorageType>
		void assignGlobalPointer(
			BaseContainer& basePointer,
			Tuple<GlobalPointers>& globalPointerTuple,
			Unsigned2Type<GlobalOffset>,
			Unsigned2Type<LocalIdx>,
			Type2Type<MemoryStorageType>,
			Bool2Type<false>
		) {
			globalPointerTuple[I<GlobalOffset>()] = basePointer.getData(Type2Type<MemoryStorageType>());
		}

		template<typename GlobalPointers, typename LocalContainers, unsigned GlobalOffset, int LocalIdx, typename MemoryStorageType>
		void storePointers(
			Tuple<GlobalPointers>& globalPointerTuple,
			Tuple<LocalContainers>& localContainerTuple,
			Unsigned2Type<GlobalOffset>,
			Unsigned2Type<LocalIdx>,
			Type2Type<MemoryStorageType>
		) {
			// We assign a global pointer to the pointer at current local index, and continue for the 
			// rest. The global offset needs to be increment with the size of the element at the current
			// index.
			assignGlobalPointer(
				localContainerTuple[I<LocalIdx>()], globalPointerTuple,
				Unsigned2Type<GlobalOffset>(), Unsigned2Type<LocalIdx>(), Type2Type<MemoryStorageType>(),
				typename IsDerivedFromSoA<typename TypeAt<LocalIdx, TList>::type>::type()
			);
			storePointers(
				globalPointerTuple,
				localContainerTuple,
				Unsigned2Type<GlobalOffset + ElementSize<typename TypeAt<LocalIdx, TList>::type>::value>(),
				Unsigned2Type<LocalIdx + 1>(),
				Type2Type<MemoryStorageType>()
			);
		}

		template<typename GlobalPointers, typename LocalContainers, unsigned GlobalOffset, typename MemoryStorageType>
		void storePointers(
			Tuple<GlobalPointers>& globalPointerTuple,
			Tuple<LocalContainers>& localContainerTuple,
			Unsigned2Type<GlobalOffset>,
			Unsigned2Type<TypeListLength<TList>::value>,
			Type2Type<MemoryStorageType>
		) {
			// We are done when the local index reaches the type list size.
		}
	};

} // namespace common_utils