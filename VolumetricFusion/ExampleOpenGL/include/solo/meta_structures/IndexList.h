#pragma once
#include <common_utils/meta_structures/BasicTypes.h>

using namespace common_utils;

namespace solo {

	/**
	 * Base struct for representing a compile-time list of indices.
	 */
	template<unsigned i, class T>
	struct IndexList {
		using Idx = Unsigned2Type<i>;
		using Next = T;
	};


	/**
	 * Compile-time shortcut to generate index lists.
	 */
	#define INDEXLIST_1(I1) IndexList<I1, NullType>
	#define INDEXLIST_2(I1, I2) IndexList<I1, INDEXLIST_1(I2)>
	#define INDEXLIST_3(I1, I2, I3) IndexList<I1, INDEXLIST_2(I2, I3)>
	#define INDEXLIST_4(I1, I2, I3, I4) IndexList<I1, INDEXLIST_3(I2, I3, I4)>
	#define INDEXLIST_5(I1, I2, I3, I4, I5) IndexList<I1, INDEXLIST_4(I2, I3, I4, I5)>
	#define INDEXLIST_6(I1, I2, I3, I4, I5, I6) IndexList<I1, INDEXLIST_5(I2, I3, I4, I5, I6)>
	#define INDEXLIST_7(I1, I2, I3, I4, I5, I6, I7) IndexList<I1, INDEXLIST_6(I2, I3, I4, I5, I6, I7)>
	#define INDEXLIST_8(I1, I2, I3, I4, I5, I6, I7, I8) IndexList<I1, INDEXLIST_7(I2, I3, I4, I5, I6, I7, I8)>
	#define INDEXLIST_9(I1, I2, I3, I4, I5, I6, I7, I8, I9) IndexList<I1, INDEXLIST_8(I2, I3, I4, I5, I6, I7, I8, I9)>
	#define INDEXLIST_10(I1, I2, I3, I4, I5, I6, I7, I8, I9, I10) IndexList<I1, INDEXLIST_9(I2, I3, I4, I5, I6, I7, I8, I9, I10)>


	/**
	 * Helper method for creating an index list of any length.
	 */
	template<unsigned ...Types>
	struct IL;

	template<>
	struct IL<> {
		using type = NullType;
	};

	template<unsigned T, unsigned ...OtherIndices>
	struct IL<T, OtherIndices...> {
		using type = IndexList<T, typename IL<OtherIndices...>::type>;
	};


	/**
	 * Creates the increasing index list [StartIdx, ... EndLimit - 1].
	 */
	template<unsigned StartIdx, unsigned EndLimit>
	struct IndexListFromRange {
		using type = IndexList<StartIdx, typename IndexListFromRange<StartIdx + 1, EndLimit>::type>;
	};

	template<unsigned EndLimit>
	struct IndexListFromRange<EndLimit, EndLimit> {
		using type = NullType;
	};


	/**
	 * Creates the increasing index list [0, ... N - 1].
	 */
	template<unsigned N>
	struct IndexListFromCount {
		using type = typename IndexListFromRange<0, N>::type;
	};


	/**
	 * Computes the length of the index list, i.e. the number of indices.
	 */
	template<class IdxList>
	struct IndexListLength;

	template<>
	struct IndexListLength<NullType> {
		enum{ value = 0 };
	};

	template<unsigned H, class T>
	struct IndexListLength<IndexList<H, T>> {
		enum{ value = 1 + IndexListLength<T>::value };
	};


	/**
	 * Computes the joint length of the two index lists, by counting common indices only once.
	 */
	template<class IdxList1, class IdxList2>
	struct IndexListJointLength;

	template<>
	struct IndexListJointLength<NullType, NullType> {
		enum{ value = 0 };
	};

	template<unsigned H, class T>
	struct IndexListJointLength<IndexList<H, T>, NullType> {
		enum{ value = 1 + IndexListJointLength<T, NullType>::value };
	};

	template<unsigned H, class T>
	struct IndexListJointLength<NullType, IndexList<H, T>> {
		enum{ value = 1 + IndexListJointLength<NullType, T>::value };
	};

	template<unsigned H, class T1, class T2>
	struct IndexListJointLength<IndexList<H, T1>, IndexList<H, T2>> {
		enum { value = 1 + IndexListJointLength<T1, T2>::value };
	};

	template<unsigned H1, class T1, unsigned H2, class T2>
	struct IndexListJointLength<IndexList<H1, T1>, IndexList<H2, T2>> {
		enum { value = 1 + ((H1 < H2) ? IndexListJointLength<T1, IndexList<H2, T2>>::value : IndexListJointLength<IndexList<H1, T1>, T2>::value) };
	};


	/**
	 * Computes a joint index list of two given index lists IdxList1 and IdxList2.
	 * If the input index lists are sorted, the resulting joint index list is also sorted. It includes
	 * only one copy of common indices (no duplicated indices).
	 */
	template<class IdxList1, class IdxList2>
	struct JointIndexList;

	template<>
	struct JointIndexList<NullType, NullType> {
		using type = NullType;
	};

	template<unsigned H, class T>
	struct JointIndexList<IndexList<H, T>, NullType> {
		using type = IndexList<H, T>;
	};

	template<unsigned H, class T>
	struct JointIndexList<NullType, IndexList<H, T>> {
		using type = IndexList<H, T>;
	};

	template<unsigned H, class T1, class T2>
	struct JointIndexList<IndexList<H, T1>, IndexList<H, T2>> {
		using type = IndexList<H, typename JointIndexList<T1, T2>::type>;
	};

	template<unsigned H1, class T1, unsigned H2, class T2>
	struct JointIndexList<IndexList<H1, T1>, IndexList<H2, T2>> {
		using type = typename ConditionedType<
			H1 < H2,
			IndexList<H1, typename JointIndexList<T1, IndexList<H2, T2>>::type>,
			IndexList<H2, typename JointIndexList<IndexList<H1, T1>, T2>::type>
		>::type;
	};


	/**
	 * Computes the local position of the given index (i.e. index's index in the index list).
	 * If the index is not found in the index list, -1 is returned.
	 */
	template<unsigned index, int localPos, class IdxList>
	struct LocalPosOfIndexHelper;

	template<unsigned index, int localPos>
	struct LocalPosOfIndexHelper<index, localPos, NullType> {
		enum{ value = -1 };
	};

	template<unsigned index, int localPos, unsigned H, class T>
	struct LocalPosOfIndexHelper<index, localPos, IndexList<H, T>> {
		enum{ value = (index == H ? localPos : LocalPosOfIndexHelper<index, localPos + 1, T>::value) };
	};

	template<unsigned index, class IdxList>
	struct LocalPosOfIndex {
		enum{ value = LocalPosOfIndexHelper<index, 0, IdxList>::value };
	};


	/**
	 * Returns 1 if the index exists in the index list, otherwise 0.
	 */
	template<unsigned index, class IdxList>
	struct IndexExists;

	template<unsigned index>
	struct IndexExists<index, NullType> {
		enum{ value = 0 };
	};

	template<unsigned index, unsigned H, class T>
	struct IndexExists<index,  IndexList<H, T>> {
		enum{ value = (index == H ? 1 : IndexExists<index, T>::value) };
	};


	/**
	 * Returns the index at the given position.
	 * If the index list's length is <= index, -1 is returned.
	 */
	template<unsigned pos, class IdxList>
	struct IndexAt;

	template<unsigned H, class T>
	struct IndexAt<0, IndexList<H, T>>  {
		enum{ value = H };
	};

	template<unsigned pos>
	struct IndexAt<pos, NullType>  {
		enum{ value = -1 };
	};

	template<unsigned pos, unsigned H, class T>
	struct IndexAt<pos, IndexList<H, T>>  {
		enum{ value = IndexAt<pos - 1, T>::value };
	};

} // namespace solo