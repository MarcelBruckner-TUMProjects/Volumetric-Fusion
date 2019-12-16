#pragma once
#include <common_utils/meta_structures/BasicTypes.h>

#include "solo/constraint_evaluation/Constraint.h"

namespace solo {
	namespace solver_proc {
		
		/**
		 * Returns the FloatType type of the first constraint in the list of constraints.
		 * If no constraint is given, NullType is returned.
		 */
		template<typename ...Constraints>
		struct GetFirstFloatType;

		template<typename FirstConstraint, typename ...OtherConstraints>
		struct GetFirstFloatType<FirstConstraint, OtherConstraints...> {
			using type = typename GetFirstFloatType<typename std::remove_reference<typename std::remove_const<FirstConstraint>::type>::type, OtherConstraints...>::type;
		};

		template<typename FloatType, typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks, typename ...OtherConstraints>
		struct GetFirstFloatType<Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>, OtherConstraints...> {
			using type = FloatType;
		};

		template<>
		struct GetFirstFloatType<> {
			using type = NullType;
		};


		/**
		 * Checks that all of the added constraints are of the same type CommonFloatType.
		 */
		template<typename CommonFloatType, typename ...Constraints>
		struct FloatTypeMatch;

		template<typename CommonFloatType>
		struct FloatTypeMatch<CommonFloatType> {
			enum {
				value = 1
			};
		};

		template<typename FloatType, typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks, typename CommonFloatType, typename ...OtherConstraints>
		struct FloatTypeMatch<CommonFloatType, Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>, OtherConstraints...> {
			enum {
				value = EqualTypes<FloatType, CommonFloatType>::value && FloatTypeMatch<CommonFloatType, OtherConstraints...>::value
			};
		};

		template<typename CommonFloatType, typename CurrentConstraint, typename ...OtherConstraints>
		struct FloatTypeMatch<CommonFloatType, CurrentConstraint, OtherConstraints...> {
			enum {
				value = FloatTypeMatch<CommonFloatType, typename std::remove_reference<typename std::remove_const<CurrentConstraint>::type>::type, OtherConstraints...>::value
			};
		};

		template<typename CommonFloatType, typename ...Constraints>
		static void checkFloatTypeMatch(Type2Type<CommonFloatType>, Constraints&&... constraints) {
			static_assert(FloatTypeMatch<CommonFloatType, Constraints...>::value > 0, "All constraints should have the same FloatType precision type.");
		}


		/**
		 * Computes the total number of constraints, used in the optimization.
		 */
		template<typename FloatType, typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks, typename ...Constraints>
		static unsigned computeNumUsedConstraints(Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>& constraint, Constraints&&... constraints) {
			using CostFunctionSignature = typename Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>::CostFunctionInterfaceSignature;

			if (constraint.isUsedInOptimization())
				return 1 + computeNumUsedConstraints(std::forward<Constraints>(constraints)...);
			else
				return computeNumUsedConstraints(std::forward<Constraints>(constraints)...);
		}

		static unsigned computeNumUsedConstraints() {
			return 0;
		}

	} // namespace solver_proc
} // namespace solo