#pragma once
#include <common_utils/memory_managment/MemoryWrapper.h>

#include "solo/constraint_evaluation/ParameterParser.h"

namespace solo {

	/**
	 * Extracts a non-const type from a pointer type.
	 * Returns EmptyType if a pointer type wasn't given.
	 */
	template<typename PointerType>
	struct ExtractTypeFromPointer {
		using type = EmptyType;
	};

	template<typename T>
	struct ExtractTypeFromPointer<T*> {
		using type = T;
	};

	template<typename T>
	struct ExtractTypeFromPointer<const T*> {
		using type = T;
	};


	/**
	 * Returns a type list of types from a type list of type pointers or tuples of pointers.
	 */
	template<typename TList>
	struct BaseTypeList;

	template<>
	struct BaseTypeList<NullType> {
		using type = NullType;
	};

	template<typename T1, typename T2>
	struct BaseTypeList<TypeList<T1, T2>> {
		using type = TypeList<typename ExtractTypeFromPointer<T1>::type, typename BaseTypeList<T2>::type>;
	};

	template<typename PointerTList, typename Other>
	struct BaseTypeList<TypeList<Tuple<PointerTList>, Other>> {
		using type = typename MergedTypeList<
			typename BaseTypeList<PointerTList>::type,
			typename BaseTypeList<Other>::type
		>::type;
	};


	/**
	 * Interface class for input data.
	 */
	template<typename ...DataPointerTypes>
	class DataHolder {
	public:
		using DataTypeList = typename BaseTypeList<typename TL<DataPointerTypes...>::type>::type;
		using ContainerTypeList = typename WrappedTypeList<DataTypeList, MemoryWrapper>::type;
		using Size = TypeListLength<DataTypeList>;

		Tuple<ContainerTypeList>& getDataTuple() { return m_dataTuple; }

		/**
		 * Returns the byte size of the data.
		 */
		CPU_AND_GPU size_t getByteSize() const {
			size_t totalSize{ 0 };
			static_for<Size::value, GetByteSizeElement>(m_dataTuple, totalSize);
			return totalSize;
		}

	private:
		Tuple<ContainerTypeList> m_dataTuple;

		/**
		 * Helper method for computing a byte size of the allocated memory.
		 */
		struct GetByteSizeElement {
			template<int idx, typename ContainerTuple>
			static void f(ContainerTuple& containerTuple, size_t& totalSize) {
				totalSize += containerTuple[I<idx>()].getByteSize();
			}
		};
	};

	template<typename ...DataPointerTypes>
	using LocalData = DataHolder<DataPointerTypes...>;

	template<typename ...DataPointerTypes>
	using GlobalData = DataHolder<DataPointerTypes...>;

	template<typename ...DataPointerTypes>
	using TextureData = DataHolder<DataPointerTypes...>;

} // namespace solo