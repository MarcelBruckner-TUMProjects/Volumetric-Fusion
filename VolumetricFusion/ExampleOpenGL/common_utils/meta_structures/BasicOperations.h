#pragma once
#include "common_utils/Common.h"
#include "BasicTypes.h"
#include "TypeList.h"

namespace common_utils {

	/**
	 * Extracts the type T from the Type2Type<T> class.
	 */
	template <typename T>
	struct ExtractType;

	template <typename T>
	struct ExtractType<Type2Type<T>> {
		using type = T;
	};


	/**
	 * Compile Time Loops with C++11 - Creating a Generalized static_for Implementation
	 * http://www.codeproject.com/Articles/857354/Compile-Time-Loops-with-Cplusplus-Creating-a-Gener
	 * 
	 * Created by Michael Gazonda on 2014-12-28.
	 * Copyright (c) 2014 Zed Mind Incorporated - http://zedmind.com. All rights reserved.
	 * LICENSED UNDER CPOL 1.02 - http://www.codeproject.com/info/cpol10.aspx
	 * 
	 * This is an implementation that uses enums instead of constexpr for use with Visual Studio 2013.
	 * 
	 * Two versions:
	 * static_for<count, methodStruct>();				=> index range is [0, count - 1]
	 * static_for<startIdx, endIdx, methodStruct>();	=> index range is [startIdx, endIdx]
	 */
	template <size_t for_start, size_t for_end, typename functor, size_t sequence_width, typename... functor_types>
	struct static_for_impl;

	template <size_t count, typename functor, size_t sequence_width = 70, typename... functor_types>
	CPU_AND_GPU inline void static_for(functor_types&&... functor_args) {
		static_for_impl<0, count - 1, functor, sequence_width, functor_types...>::loop(
			std::forward<functor_types>(functor_args)...);
	}

	template <size_t start, size_t end, typename functor, size_t sequence_width = 70, typename... functor_types>
	CPU_AND_GPU inline void static_for(functor_types&&... functor_args) {
		static_for_impl<start, end, functor, sequence_width, functor_types...>::loop(
			std::forward<functor_types>(functor_args)...);
	}

	template <size_t for_start, size_t for_end, typename functor, size_t sequence_width, typename... functor_types>
	struct static_for_impl {
		CPU_AND_GPU static inline void loop(functor_types&&... functor_args) {
			using sequence = point<for_start, for_end>;
			next<sequence>(std::integral_constant<bool, sequence::is_end_point_>(),
				std::forward<functor_types>(functor_args)...);
		}

	private:
		template <size_t pt_start, size_t pt_end>
		struct point {
			// Method declarations.
			template <size_t index>
			struct child_start;

			template <size_t index>
			struct child_end;

			template <size_t max>
			struct points_in_sequence;

			// Member definitions.
			enum { start_ = pt_start };

			enum { end_ = pt_end };

			enum { count_ = end_ - start_ + 1 };

			enum { is_end_point_ = count_ <= sequence_width };

			enum {
				sequence_count_ = points_in_sequence<sequence_width>::v > sequence_width
				? sequence_width
				: points_in_sequence<sequence_width>::v
			};

			// Method definitions.
			template <size_t index>
			struct child_start {
				enum { v = index == 0 ? pt_start : child_end<index == 0 ? sequence_count_ : index - 1>::v + 1 };
			};

			template <size_t index>
			struct child_end {
				enum {
					v = (index >= sequence_count_ - 1)
						    ? pt_end
						    : pt_start + points_in_sequence<sequence_count_>::v * (index + 1) - (index < count_ ? 1 : 0)
				};
			};

			template <size_t max>
			struct points_in_sequence {
				enum { v = count_ / max + ((count_ % max) > 0 ? 1 : 0) };
			};

			template <size_t index>
			using child_point = point<child_start<index>::v, child_end<index>::v>;
		};

		template <size_t flat_start, size_t flat_end, class flat_functor>
		struct flat_for {
			CPU_AND_GPU static inline void flat_loop(functor_types&&... functor_args) {
				flat_next(std::integral_constant<size_t, flat_start>(), std::forward<functor_types>(functor_args)...);
			}

		private:
			CPU_AND_GPU static inline void flat_next(std::integral_constant<size_t, flat_end + 1>, functor_types&&...) { }

			template <size_t index>
			CPU_AND_GPU static inline void flat_next(std::integral_constant<size_t, index>, functor_types&&... functor_args) {
				flat_functor::template f<index>(std::forward<functor_types>(functor_args)...);
				flat_next(std::integral_constant<size_t, index + 1>(), std::forward<functor_types>(functor_args)...);
			}

		};

		template <typename sequence>
		struct flat_sequence {
			template <size_t index>
			CPU_AND_GPU static inline void f(functor_types&&... functor_args) {
				using pt = typename sequence::template child_point<index>;
				next<pt>(std::integral_constant<bool, pt::is_end_point_>(), std::forward<functor_types>(functor_args)...);
			}
		};

		template <typename sequence>
		CPU_AND_GPU static inline void next(std::true_type, functor_types&&... functor_args) {
			flat_for<sequence::start_, sequence::end_, functor>::flat_loop(std::forward<functor_types>(functor_args)...);
		}

		template <typename sequence>
		CPU_AND_GPU static inline void next(std::false_type, functor_types&&... functor_args) {
			flat_for<0, sequence::sequence_count_ - 1, flat_sequence<sequence>>::flat_loop(
				std::forward<functor_types>(functor_args)...);
		}

	};

	
	/**
	 * Returns true if two types are the same, otherwise false.
	 */
	template<typename T1, typename T2>
	struct EqualTypes {
		enum {
			value = 0
		};
	};

	template<typename T>
	struct EqualTypes<T, T> {
		enum {
			value = 1
		};
	};

	
	/**
	 * Returns true if Base type is a base type of Derived type, otherwise false.
	 */
	template <typename Base> Bool2Type<true> isBaseOfTestFunc(const volatile Base*);
	template <typename Base> Bool2Type<false> isBaseOfTestFunc(const volatile void*);
	
	template <typename Base, typename Derived>
	struct IsBaseOf {
		using type = decltype(isBaseOfTestFunc<Base>(std::declval<Derived*>()));
	};


	/**
	 * Removes a const and/or reference qualifier from the type(list).
	 */
	template<typename Type>
	struct RemoveConstAndRef {
		using type = Type;
	};

	template<typename Type>
	struct RemoveConstAndRef<Type&> {
		using type = typename RemoveConstAndRef<Type>::type;
	};

	template<typename Type>
	struct RemoveConstAndRef<const Type> {
		using type = typename RemoveConstAndRef<Type>::type;
	};

	template<typename T1, typename T2>
	struct RemoveConstAndRef<TypeList<T1, T2>> {
		using type = TypeList<
			typename RemoveConstAndRef<T1>::type, 
			typename RemoveConstAndRef<T2>::type
		>;
	};

	
	/**
	 * Extracts the value from unsigned/int types.
	 */
	template<typename T>
	struct ExtractValue;

	template<unsigned i>
	struct ExtractValue<Unsigned2Type<i>> {
		enum {
			value = i
		};
	};

	template<int i>
	struct ExtractValue<Int2Type<i>> {
		enum {
			value = i
		};
	};


	/**
	 * Returns the maximum of two values at compile-time.
	 */
	template<int a, int b, bool AIsGreaterThanB>
	struct MaxValueHelper;

	template<int a, int b>
	struct MaxValueHelper<a, b, true> {
		enum {
			value = a
		};
	};

	template<int a, int b>
	struct MaxValueHelper<a, b, false> {
		enum {
			value = b
		};
	};

	template<int a, int b>
	struct MaxValue {
		enum {
			value = MaxValueHelper<a, b, (a > b)>::value
		};
	};


	/**
	 * Returns the minimum of two values at compile-time.
	 */
	template<int a, int b, bool AIsGreaterThanB>
	struct MinValueHelper;

	template<int a, int b>
	struct MinValueHelper<a, b, true> {
		enum {
			value = b
		};
	};

	template<int a, int b>
	struct MinValueHelper<a, b, false> {
		enum {
			value = a
		};
	};

	template<int a, int b>
	struct MinValue {
		enum {
			value = MinValueHelper<a, b, (a > b)>::value
		};
	};

} // namespace common_utils
