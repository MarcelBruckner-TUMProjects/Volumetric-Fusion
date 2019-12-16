#pragma once
#include "TypeList.h"
#include "BasicOperations.h"

namespace common_utils {

	/**
	 * Generates an array with different types, specified with a TypeList.
	 */
	template <class TList, template <class> class PlaceHolder>
	class ArrayGenerator;

	template <template <class> class PlaceHolder>
	class ArrayGenerator<NullType, PlaceHolder> { };

	template <class AtomicType, template <class> class PlaceHolder>
	class ArrayGenerator : public PlaceHolder<AtomicType> {
	public:
		using ElementType = PlaceHolder<AtomicType>;
	};

	template <class T1, class T2, template <class> class PlaceHolder>
	class ArrayGenerator<TypeList<T1, T2>, PlaceHolder> : 
		public ArrayGenerator<T1, PlaceHolder>,
		public ArrayGenerator<T2, PlaceHolder>
	{
	public:
		using Element = ArrayGenerator<T1, PlaceHolder>;
		using Next = ArrayGenerator<T2, PlaceHolder>;
	};

	
	/**
	 * Returns the array element at the given position.
	 * If the array's length is <= pos, the compile-time error is thrown.
	 */
	template<class ArraySubType, typename ReturnType>
	CPU_AND_GPU inline ReturnType& ElementAtHelper(ArraySubType& arrayObj, Type2Type<ReturnType>, Unsigned2Type<0>) {
		typename ArraySubType::Element& element = arrayObj;
		return element;
	}

	template<unsigned i, class ArraySubType, typename ReturnType>
	CPU_AND_GPU inline ReturnType& ElementAtHelper(ArraySubType& arrayObj, Type2Type<ReturnType>, Unsigned2Type<i>) {
		typename ArraySubType::Next& subArrayObj = arrayObj;
		return ElementAtHelper(subArrayObj, Type2Type<ReturnType>(), Unsigned2Type<i - 1>());
	}

	template<unsigned i, class T1, class T2, template <class> class PlaceHolder>
	CPU_AND_GPU auto ElementAt(ArrayGenerator<TypeList<T1, T2>, PlaceHolder>& arrayObj) -> decltype(ElementAtHelper(arrayObj, Type2Type<PlaceHolder<typename TypeAt<i, TypeList<T1, T2>>::type>>(), Unsigned2Type<i>())) {
		using ReturnType = PlaceHolder<typename TypeAt<i, TypeList<T1, T2>>::type>;
		return ElementAtHelper(arrayObj, Type2Type<ReturnType>(), Unsigned2Type<i>());
	}


	/**
	 * Place holder class for the object of a specific type.
	 */
	template<typename T>
	struct Holder {
		T m_value;
	};


	/**
	 * IndexedType is a wrapper around a type that adds an index to the type's signature.
	 * It is useful when you want to differentiate two objects of the same type.
	 */
	template<unsigned i, typename T>
	struct IndexedType {
		T m_t;
	};

	
	/**
	 * IndexedTypeList takes a typelist as an input and returns its indexed value, i.e. each type
	 * in the typelist is being assigned a unique indexed type (first type has index 0, second 1,
	 * etc.).
	 */
	template<typename TList, unsigned i>
	struct IndexedTypeListHelper;

	template<unsigned i>
	struct IndexedTypeListHelper<NullType, i> {
		using type = NullType;
	};

	template<typename T1, typename T2, unsigned i>
	struct IndexedTypeListHelper<TypeList<T1, T2>, i> {
		using type = TypeList<IndexedType<i, T1>, typename IndexedTypeListHelper<T2, i + 1>::type>;
	};

	template<typename TList>
	struct IndexedTypeList {
		using type = typename IndexedTypeListHelper<TList, 0>::type;
	};


	/**
	 * UniqueTuple class is an interface to the underlying array of different types.
	 * All types in the typelist need to be unique, some compilers will fail if two types repeat.
	 */
	template<class TList>
	class UniqueTuple {
	private:
		ArrayGenerator<TList, Holder> m_array;

	public:
		/**
		 * Returns the element at the given index. 
		 */
		template<unsigned i>
		CPU_AND_GPU auto at(I<i>) -> typename TypeAt<i, TList>::type& {
			return ElementAt<i>(m_array).m_value;
		}

		template<unsigned i>
		CPU_AND_GPU auto at(I<i>) const -> const typename TypeAt<i, TList>::type&{
			return ElementAt<i>(const_cast<ArrayGenerator<TList, Holder>&>(m_array)).m_value;
		}

		template<unsigned i>
		CPU_AND_GPU auto operator[](I<i>) -> typename TypeAt<i, TList>::type& {
			return ElementAt<i>(m_array).m_value;
		}

		template<unsigned i>
		CPU_AND_GPU auto operator[](I<i>) const -> const typename TypeAt<i, TList>::type&{
			return ElementAt<i>(const_cast<ArrayGenerator<TList, Holder>&>(m_array)).m_value;
		}

		/**
		 * Returns the number of elements of the array.
		 */
		CPU_AND_GPU unsigned size() const { return TypeListLength<TList>::value; }
	};


	/**
	 * Tuple class is an interface to the underlying array of different types.
	 * Types can repeat themselves, i.e. don't have to be unique, since ambiguity is resolved using
	 * intermediate compile-time indexing.
	 */
	template<class TList>
	class Tuple {
	private:
		using IndexedTList = typename IndexedTypeList<TList>::type;
		ArrayGenerator<IndexedTList, Holder> m_array;

	public:
		/**
		 * Initializes the tuple with given elements. All elements should be of correct type.
		 * The index tells the offset of the start of the tuple.
		 */	
		template<unsigned Idx, typename Element, typename ...OtherElements>
		CPU_AND_GPU void initialize(Unsigned2Type<Idx>, Element&& element, OtherElements&&... others) {
			at(I<Idx>()) = element;
			initialize(Unsigned2Type<Idx + 1>(), std::forward<OtherElements>(others)...);
		}

		template<unsigned Idx>
		CPU_AND_GPU void initialize(Unsigned2Type<Idx>) { }

		/**
		 * Returns the element at the given index.
		 */
		template<unsigned i>
		CPU_AND_GPU auto at(I<i>) -> typename TypeAt<i, TList>::type& {
			return ElementAt<i>(m_array).m_value.m_t;
		}

		template<unsigned i>
		CPU_AND_GPU auto at(I<i>) const -> const typename TypeAt<i, TList>::type&{
			return ElementAt<i>(const_cast<ArrayGenerator<IndexedTList, Holder>&>(m_array)).m_value.m_t;
		}

		template<unsigned i>
		CPU_AND_GPU auto operator[](I<i>) -> typename TypeAt<i, TList>::type& {
			return ElementAt<i>(m_array).m_value.m_t;
		}

		template<unsigned i>
		CPU_AND_GPU auto operator[](I<i>) const -> const typename TypeAt<i, TList>::type&{
			return ElementAt<i>(const_cast<ArrayGenerator<IndexedTList, Holder>&>(m_array)).m_value.m_t;
		}

		/**
		 * Returns the number of elements of the array.
		 */
		CPU_AND_GPU unsigned size() const { return TypeListLength<TList>::value; }
	};


	/**
	 * Empty tuple is a placeholder for a tuple, where no assignment operations should be done and the 
	 * tuple size should be zero. The assignment operations with FloatType type are supported.
	 */
	template<typename FloatType>
	class EmptyTuple {
	public:
		template<unsigned i>
		CPU_AND_GPU FloatType& at(I<i>) { return m_placeholder; }

		template<unsigned i>
		CPU_AND_GPU FloatType& operator[](I<i>) { return m_placeholder; }

		CPU_AND_GPU unsigned size() const { return 0; }

	private:
		FloatType m_placeholder;
	};


	/**
	 * Double empty tuple is an empty tuple that supports double hierachy indexing.
	 */
	template<typename FloatType>
	class DoubleEmptyTuple {
	public:
		template<unsigned i>
		CPU_AND_GPU EmptyTuple<FloatType>& at(I<i>) { return m_emptyTuple; }

		template<unsigned i>
		CPU_AND_GPU EmptyTuple<FloatType>& operator[](I<i>) { return m_emptyTuple; }

		CPU_AND_GPU unsigned size() const { return 0; }

	private:
		EmptyTuple<FloatType> m_emptyTuple;
	};


	/**
	 * Returns the tuple size at compile-time.
	 */
	template<typename T>
	struct TupleSize;

	template<typename TList>
	struct TupleSize<Tuple<TList>> {
		enum {
			value = TypeListLength<TList>::value
		};
	};


	/**
	 * Method for tuple initialization, given any number of elements (with arbitrary types).
	 */
	template<typename ...ElementTypes, typename ElementTList = typename RemoveConstAndRef<typename TL<ElementTypes...>::type>::type>
	CPU_AND_GPU Tuple<ElementTList> makeTuple(ElementTypes&&... elements) {
		Tuple<ElementTList> tuple;
		tuple.initialize(Unsigned2Type<0>(), std::forward<ElementTypes>(elements)...);
		return tuple;
	}

} // namespace common_utils