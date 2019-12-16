#pragma once
#include <common_utils/meta_structures/BasicTypes.h>

#include "solo/data_structures/DenseMatrix.h"
#include "CostFunctionInterface.h"
#include "solo/data_structures/DataHolder.h"
#include "solo/data_structures/MemoryProcessing.h"
#include "solo/data_structures/IndexMatrix.h"

namespace solo {

	/**
	 * Constraint class contains all necessary information to evaluate a certain cost function.
	 * The FloatType type decides about the floating-point type used for parameter estimation and residual
	 * computation. Could be either float or double.
	 * At compile-time, you need to provide the cost function signature (as a template parameter).
	 * At run-time, you need to provide the number of the residuals the cost function will be evaluated at.
	 */
	template<typename FloatType, typename CostFunction, typename LocalDataHolder = DataHolder<>, typename GlobalDataHolder = DataHolder<>, unsigned NumBlocks = 1>
	class Constraint {
	public:
		using CostFunctionInterfaceSignature = typename CostFunctionInterfaceType<CostFunction>::type;

	private:
		unsigned m_nResiduals{ 0 };
		DenseMatrixWrapper<int> m_indexMatrixWrapper;
		LocalDataHolder m_localDataHolder;
		GlobalDataHolder m_globalDataHolder;

	public:
		Constraint(unsigned nResiduals) : m_nResiduals{ nResiduals } {	}

		/**
		 * We disable copy/move/assignment/move assignment operations.
		 */
		Constraint(const Constraint& other) = delete;
		Constraint(Constraint&& other) = delete;
		Constraint& operator=(const Constraint& other) = delete;
		Constraint& operator=(Constraint&& other) = delete;

		/**
		 * Adds the parameter index matrix to the constraint, as a raw integer array, stored column-major.
		 * The index matrix should have one column for each parameter dimension (for all parameters), and
		 * it should have the same number of rows as the number of resiudals of the constraint.
		 */
		template<typename MemoryStorageType>
		void addIndexMatrix(IndexMatrix<MemoryStorageType> indexMatrix) {
			m_indexMatrixWrapper.wrapMemory(indexMatrix.getData(), m_nResiduals, getTotalParamDim(), Type2Type<MemoryStorageType>());
		}

		/**
		 * Adds a pointer to the local data at the given index. 
		 * The data array length should be the same as the number of residuals. Each execution of the
		 * cost function will get only one corresponding element of the data array.
		 * The data can be allocated on the host or device memory (additional flag signalizes which memory
		 * the data is situated on, the default choice is host memory).
		 */
		template<unsigned Idx, typename T>
		void addLocalData(T* data) {
			addLocalData<Idx>(data, Type2Type<MemoryTypeCPU>());
		}

		template<unsigned Idx, typename T>
		void addLocalData(T* data, Type2Type<MemoryTypeCPU>) {
			static_assert(Idx < LocalDataHolder::Size::value, "Local data index out of range.");
			m_localDataHolder.getDataTuple()[I<Idx>()].wrapMemory(const_cast<typename std::remove_const<T>::type*>(data), m_nResiduals, Type2Type<MemoryTypeCPU>());
		}

		template<unsigned Idx, typename T>
		void addLocalData(T* data, Type2Type<MemoryTypeCUDA>) {
			static_assert(Idx < LocalDataHolder::Size::value, "Local data index out of range.");
			m_localDataHolder.getDataTuple()[I<Idx>()].wrapMemory(const_cast<typename std::remove_const<T>::type*>(data), m_nResiduals, Type2Type<MemoryTypeCUDA>());
		}

		template<unsigned Idx, typename T>
		void addLocalData(T* data, Type2Type<MemoryTypeCustom>) {
			static_assert(Idx < LocalDataHolder::Size::value, "Local data index out of range.");
			std::cerr << "Local data doesn't support custom memory type. Use global data instead." << endl;
			exit(-1);
		}

		/**
		 * Adds a pointer to the local data, given as a tuple of pointers with a start offset.
		 * The data array length should be the same as the number of residuals. Each execution of the
		 * cost function will get only one corresponding element of the data array.
		 * The data can be allocated on the host or device memory (additional flag signalizes which memory
		 * the data is situated on, the default choice is host memory).
		 */
		template<unsigned Offset, typename PointerTList>
		void addLocalData(const Tuple<PointerTList>& data) {
			addLocalData<Offset>(data, Type2Type<MemoryTypeCPU>());
		}

		template<unsigned Offset, typename PointerTList, typename MemoryStorageType>
		void addLocalData(const Tuple<PointerTList>& data, Type2Type<MemoryStorageType>) {
			static_assert(Offset + TypeListLength<PointerTList>::value - 1 < LocalDataHolder::Size::value, "Local data index out of range.");
			static_for<TypeListLength<PointerTList>::value, InitializePointersFromTuple>(m_localDataHolder, data, m_nResiduals, Unsigned2Type<Offset>(), Type2Type<MemoryStorageType>());
		}

		template<unsigned Offset, typename PointerTList>
		void addLocalData(const Tuple<PointerTList>& data, Type2Type<MemoryTypeCustom>) {
			static_assert(Offset + TypeListLength<PointerTList>::value - 1 < LocalDataHolder::Size::value, "Local data index out of range.");
			std::cerr << "Local data doesn't support custom memory type. Use global data instead." << endl;
			exit(-1);
		}

		/**
		 * Adds a pointer to the global data at the given index.
		 * Since the length of the data array can be arbitrary, you need to provide it as an argument size.
		 * Each execution of the cost function will get the access to the whole data array, i.e. directly
		 * the data pointer.
		 * The data can be allocated on the host or device memory (additional flag signalizes which memory
		 * the data is situated on, the default choice is host memory).
		 */
		template<unsigned Idx, typename T>
		void addGlobalData(T* data, unsigned size) {
			addGlobalData<Idx>(data, size, Type2Type<MemoryTypeCPU>());
		}

		template<unsigned Idx, typename T, typename MemoryStoragetType>
		void addGlobalData(T* data, unsigned size, Type2Type<MemoryStoragetType>) {
			static_assert(Idx < GlobalDataHolder::Size::value, "Global data index out of range.");
			m_globalDataHolder.getDataTuple()[I<Idx>()].wrapMemory(const_cast<typename std::remove_const<T>::type*>(data), size, Type2Type<MemoryStoragetType>());
		}

		/**
		 * Adds a pointer to the global data, given as a tuple of pointers with a start offset.
		 * Since the length of the data array can be arbitrary, you need to provide it as an argument size.
		 * Each execution of the cost function will get the access to the whole data array, i.e. directly
		 * the data pointer.
		 * The data can be allocated on the host or device memory (additional flag signalizes which memory
		 * the data is situated on, the default choice is host memory).
		 */
		template<unsigned Offset, typename PointerTList>
		void addGlobalData(const Tuple<PointerTList>& data, unsigned size) {
			addGlobalData<Offset>(data, size, Type2Type<MemoryTypeCPU>());
		}

		template<unsigned Offset, typename PointerTList, typename MemoryStorageType>
		void addGlobalData(const Tuple<PointerTList>& data, unsigned size, Type2Type<MemoryStorageType>) {
			static_assert(Offset + TypeListLength<PointerTList>::value - 1 < GlobalDataHolder::Size::value, "Global data index out of range.");
			static_for<TypeListLength<PointerTList>::value, InitializePointersFromTuple>(m_globalDataHolder, data, size, Unsigned2Type<Offset>(), Type2Type<MemoryStorageType>());
		}

		/**
		 * Sets the flag about whether we use the constraint in our optimization problem
		 * solving, or not. If set to false, the constraint will be skipped in the optimization,
		 * although it is added to the Solve method.
		 * That is useful when we want to decide at runtime, which constraints we want to
		 * use. We don't need to specify different Solve(...) calls, for each combination,
		 * inside if-else clauses, we just run one Solve(...) call with all possible constraints
		 * listed, but some constraints can have the non-used flag turned on.
		 * By default, the constraint is used in the optimization.
		 */
		void useInOptimization(bool bUsedInOptimization) {
			m_bUsedInOptimization = bUsedInOptimization;
		}

		/**
		 * Getters.
		 */
		unsigned getNumResiduals() const {
			return m_nResiduals;
		}

		unsigned getResidualDim() const {
			return GetResidualDim<CostFunctionInterfaceSignature>::value;
		}

		unsigned getTotalParamDim() const {
			return GetTotalParamDim<CostFunctionInterfaceSignature>::value;
		}

		unsigned getNumParams() const {
			return GetNumParams<CostFunctionInterfaceSignature>::value;
		}

		template<unsigned paramId>
		constexpr unsigned getSpecificParamDim(I<paramId>) const {
			return GetSpecificParamDim<paramId, CostFunctionInterfaceSignature>::value;
		}
		
		DenseMatrixWrapper<int>& getIndexMatrix() {
			return m_indexMatrixWrapper;
		}

		auto getLocalData() -> decltype(m_localDataHolder.getDataTuple()) {
			return m_localDataHolder.getDataTuple();
		}

		auto getGlobalData() -> decltype(m_globalDataHolder.getDataTuple()) {
			return m_globalDataHolder.getDataTuple();
		}

		size_t getLocalDataByteSize() {
			return m_localDataHolder.getByteSize();
		}

		size_t getGlobalDataByteSize() {
			return m_globalDataHolder.getByteSize();
		}

		bool isUsedInOptimization() const {
			return m_bUsedInOptimization;
		}

	private:
		bool m_bUsedInOptimization{ true };

		/**
		 * Helper method for storing local/global data.
		 */
		struct InitializePointersFromTuple {
			template<int Idx, unsigned GlobalOffset, typename DataHolder, typename PointerTList, typename MemoryStorageType>
			static void f(DataHolder& dataHolder, const Tuple<PointerTList>& pointerTuple, unsigned size, Unsigned2Type<GlobalOffset>, Type2Type<MemoryStorageType>) {
				using PointerType = typename TypeAt<Idx, PointerTList>::type;
				dataHolder.getDataTuple()[I<GlobalOffset + Idx>()].wrapMemory(
					const_cast<typename std::remove_const<PointerType>::type>(pointerTuple[I<Idx>()]), size, Type2Type<MemoryStorageType>()
				);
			}
		};		
	};


	/**
	 * Returns the FloatType of a Constraint object.
	 */
	template<typename ConstraintType>
	struct ExtractFloatType;

	template<typename FloatType, typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks>
	struct ExtractFloatType<Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>> {
		using type = FloatType;
	};

	
	/**
	 * Returns the CostFunction of a Constraint object.
	 */
	template<typename ConstraintType>
	struct ExtractCostFunction;

	template<typename FloatType, typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks>
	struct ExtractCostFunction<Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>> {
		using type = CostFunction;
	};

	
	/**
	 * Returns the LocalData of a Constraint object.
	 */
	template<typename ConstraintType>
	struct ExtractLocalData;

	template<typename FloatType, typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks>
	struct ExtractLocalData<Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>> {
		using type = LocalData;
	};


	/**
	 * Returns the GlobalData of a Constraint object.
	 */
	template<typename ConstraintType>
	struct ExtractGlobalData;

	template<typename FloatType, typename CostFunction, typename LocalData, typename GlobalData, unsigned NumBlocks>
	struct ExtractGlobalData<Constraint<FloatType, CostFunction, LocalData, GlobalData, NumBlocks>> {
		using type = GlobalData;
	};

} // namespace solo