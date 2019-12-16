#pragma once
#include "common_utils/Common.h"

namespace common_utils {
	
	/**
	 * Null type class (is not supposed to be initialized).
	 */
	class NullType {};

	/**
	 * Empty type class (can be initialized to represent an empty object).
	 */
	class EmptyType {};

	/**
	 * If condition is true, its result is type T1, otherwise its result is type T2.
	 */
	template<bool condition, typename T1, typename T2>
	struct ConditionedType;

	template<typename T1, typename T2>
	struct ConditionedType<true, T1, T2> {
		using type = T1;
	};

	template<typename T1, typename T2>
	struct ConditionedType<false, T1, T2> {
		using type = T2;
	};

	/**
	 * Converts the compile-time int and unsigned number to type.
	 */
	template<int i>
	struct Int2Type {};

	template<unsigned u>
	struct Unsigned2Type {};

	template<bool b>
	struct Bool2Type {};

	template<unsigned u>
	struct I {};

	/**
	 * Converts a type to a type class, to enable partial template specialization for functions.
	 */
	template <typename T>
	struct Type2Type {
		using type = T;
	};

	/**
	 * Raw array wrapper that allows empty raw arrays.
	 */
	template<typename Type, unsigned Size>
	struct RawArray {
	public:
		CPU_AND_GPU Type& operator[](unsigned i) { return m_data[i]; }
		CPU_AND_GPU const Type& operator[](unsigned i) const { return m_data[i]; }

	private:
		Type m_data[Size];
	};

	template<typename Type>
	struct RawArray<Type, 0> { };

} // namespace common_utils