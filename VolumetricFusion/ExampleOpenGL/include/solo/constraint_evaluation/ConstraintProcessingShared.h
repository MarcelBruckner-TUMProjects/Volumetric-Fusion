#pragma once
#include <common_utils/meta_structures/BasicTypes.h>
#include <common_utils/meta_structures/TypeList.h>
#include <common_utils/meta_structures/Tuple.h>
#include <common_utils/memory_managment/MemoryWrapper.h>

using namespace common_utils;

namespace solo {
	namespace constraint_proc {
		
		/**
		 * Returns a type list of type pointers from a type list of types.
		 */
		template<typename TList>
		struct DataPointerTypeList;

		template<>
		struct DataPointerTypeList<NullType> {
			using type = NullType;
		};

		template<typename T1, typename T2>
		struct DataPointerTypeList<TypeList<MemoryWrapper<T1>, T2>> {
			using type = TypeList<T1*, typename DataPointerTypeList<T2>::type>;
		};


		/**
		 * Creates tuple of data pointers from a tuple of memory wrappers.
		 */
		template<typename TupleOfTypes>
		struct TupleOfDataPointers;

		template<typename TList>
		struct TupleOfDataPointers<Tuple<TList>> {
			using type = Tuple<typename DataPointerTypeList<TList>::type>;
		};


		/**
		 * Updates the device memory of the tuple, containing memory wrappers. Doesn't execute any update
		 * if not necessary (the memory wrapper flag indicates whether update is necessary).
		 * Important: Can be called only from CUDA code.
		 */
		struct UpdateWrapperMemory {
			template<int idx, typename ContainerTuple> 
			static void f(ContainerTuple& containerTuple, Type2Type<MemoryTypeCPU>) {
				if (!containerTuple[I<idx>()].isUpdatedHost() && !containerTuple[I<idx>()].isWrappingCustomMemory()) {
					// We don't update custom memory pointers.
#					ifdef COMPILE_CUDA
					containerTuple[I<idx>()].copyDeviceToHost();
					containerTuple[I<idx>()].setUpdated(true, true);
#					endif				
				}
			}

			template<int idx, typename ContainerTuple>
			static void f(ContainerTuple& containerTuple, Type2Type<MemoryTypeCUDA>) {
				if (!containerTuple[I<idx>()].isUpdatedDevice() && !containerTuple[I<idx>()].isWrappingCustomMemory()) {
					// We don't update custom memory pointers.
#					ifdef COMPILE_CUDA
					containerTuple[I<idx>()].copyHostToDevice();
					containerTuple[I<idx>()].setUpdated(true, true);
#					endif
				}
			}
		};


		/**
		 * Copies the pointers from a tuple of memory wrappers to the tuple of raw pointers.
		 * The memory type, i.e. whether the pointers should point to host or device memory, is specified
		 * as a template argument.
		 */
		struct CopyPointers {
			template<int idx, typename ContainerTuple, typename PointerTuple, typename MemoryType> 
			static void f(ContainerTuple& containerTuple, PointerTuple& pointerTuple, Type2Type<MemoryType>) {
				if (containerTuple[I<idx>()].isWrappingCustomMemory())
					pointerTuple[I<idx>()] = containerTuple[I<idx>()].getData(Type2Type<MemoryTypeCustom>());
				else	
					pointerTuple[I<idx>()] = containerTuple[I<idx>()].getData(Type2Type<MemoryType>());
			}
		};


		/**
		 * Helper method for correct local memory query.
		 */
		struct IncrementLocalPointers {
			template<int idx, typename LocalData> 
			CPU_AND_GPU static void f(int increment, LocalData& localData) {
				localData[I<idx>()] = localData[I<idx>()] + increment;
			}
		};
		
	} // namespace constraint_proc
} // namespace solo