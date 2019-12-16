#pragma once
#include "common_utils/RuntimeAssertion.h"
#include "common_utils/Common.h"
#include "common_utils/meta_structures/BasicTypes.h"
#include "MemoryType.h"

namespace common_utils {

	/**
	 * The class is an interface to the existing CPU (host), GPU (device) or custom (unknown) memory. 
	 * It offers methods to replicate the memory to CUDA (device) or CPU (host) memory.
	 * If the given memory is custom, the method doesn't replicate any memory (i.e. could be used for
	 * GPU textures).
	 */
	template<typename T>
	class MemoryWrapper {
	public:
		using Type = T;

		/**
		 * Default constructor.
		 */
		MemoryWrapper() = default;

		/**
		 * Initialization with raw array in the host/CPU memory.
		 * Important: The input array is not copied. You need to be careful to not destroy it before the 
		 * processing ends.
		 */
		MemoryWrapper(T* dataHost, size_t size, Type2Type<MemoryTypeCPU>) {
			wrapMemory(dataHost, size, Type2Type<MemoryTypeCPU>());
		}

		/**
		 * Initialization with raw array in the device/GPU memory.
		 * Important: The input array is not copied. You need to be careful to not destroy it before the
		 * processing ends.
		 */
		MemoryWrapper(T* dataDevice, size_t size, Type2Type<MemoryTypeCUDA>) {
			wrapMemory(dataDevice, size, Type2Type<MemoryTypeCUDA>());
		}

		/**
		 * Initialization with raw array in the custom/unknown memory.
		 * Important: The input array is not copied. You need to be careful to not destroy it before the
		 * processing ends.
		 */
		MemoryWrapper(T* dataCustom, size_t size, Type2Type<MemoryTypeCustom>) {
			wrapMemory(dataCustom, size, Type2Type<MemoryTypeCustom>());
		}

		/**
		 * Destructor.
		 * We only clean device memory, only if we use GPU evaluation of contraints (otherwise, the
		 * memory wrapper shouldn't posess any device memory).
		 */
		~MemoryWrapper() {
			if (m_wrappedMemoryType == MemoryType::CUDA_MEMORY) {
				// We free the host memory only if the device memory was wrapped (and if host memory
				// was allocated/updated).
				freeHost();
				m_dataDevice = nullptr;
			}

			if (m_wrappedMemoryType == MemoryType::CPU_MEMORY) {
				// We free the device memory only if the host memory was wrapped (and if device memory
				// was allocated/updated).
				#ifdef COMPILE_CUDA
				freeDevice();
				#endif
				m_dataHost = nullptr;
			}
		}

		/**
		 * We disable copy/move/assignment/move assignment operations.
		 */
		MemoryWrapper(const MemoryWrapper& other) = delete;
		MemoryWrapper(MemoryWrapper&& other) = delete;
		MemoryWrapper& operator=(const MemoryWrapper& other) = delete;
		MemoryWrapper& operator=(MemoryWrapper&& other) = delete;

		/**
		 * The memory container points to the same host memory as is given by the input pointer, i.e. 
		 * host memory is not copied. The device memory is not updated. The flag is set that signals 
		 * that host memory is updated and device memory is not.
		 */
		void wrapMemory(T* dataHost, unsigned size, Type2Type<MemoryTypeCPU>) {
			// Clean the previous memory, if necessary.
			if (m_wrappedMemoryType == MemoryType::CPU_MEMORY) {
				#ifdef COMPILE_CUDA
				freeDevice();
				#endif
			}
			else if (m_wrappedMemoryType == MemoryType::CUDA_MEMORY) {
				freeHost();
			}

			// Set the host data pointer.
			m_size = size;
			m_dataHost = dataHost;
			m_wrappedMemoryType = MemoryType::CPU_MEMORY;
			m_bUpdatedHost = true;
			m_bUpdatedDevice = false;
		}

		/**
		 * The memory container points to the same device memory as is given by the input pointer, i.e.
		 * device memory is not copied. The host memory is not allocated, since the device wrapper can
		 * only be used with GPU constraint evaluation, therefore the host memory is not needed. 
		 */
		void wrapMemory(T* dataDevice, unsigned size, Type2Type<MemoryTypeCUDA>) {
			// Clean the previous memory, if necessary.
			if (m_wrappedMemoryType == MemoryType::CPU_MEMORY) {
				#ifdef COMPILE_CUDA
				freeDevice();
				#endif
			}
			else if (m_wrappedMemoryType == MemoryType::CUDA_MEMORY) {
				freeHost();
			}

			// Set the device data pointer.
			m_size = size;
			m_dataDevice = dataDevice;
			m_wrappedMemoryType = MemoryType::CUDA_MEMORY;
			m_bUpdatedHost = false;
			m_bUpdatedDevice = true;
		}

		/**
		 * The memory container points to the same custom memory as is given by the input pointer, the
		 * custom memory is not copied.
		 * Important: The updating of the custom memory is not possible, so the user needs to take care
		 * that it is in the right form (either in the device memory for GPU evaluation or host memory
		 * for CPU evaluation).
		 */
		void wrapMemory(T* dataCustom, unsigned size, Type2Type<MemoryTypeCustom>) {
			// Clean the previous memory, if necessary.
			if (m_wrappedMemoryType == MemoryType::CPU_MEMORY) {
				#ifdef COMPILE_CUDA
				freeDevice();
				#endif
			}
			else if (m_wrappedMemoryType == MemoryType::CUDA_MEMORY) {
				freeHost();
			}

			// We set the host data pointer to point to the custom memory.
			m_size = size;
			m_dataHost = dataCustom;
			m_wrappedMemoryType = MemoryType::CUSTOM_MEMORY;
			m_bUpdatedHost = false;
			m_bUpdatedDevice = false;
		}

		/**
		 * Returns the pointer to the data, deciding about the memory location (host or device) either at 
		 * run-time or compile-time.
		 */
		CPU_AND_GPU T* getData(MemoryType memoryType) {
			switch (memoryType) {
			case MemoryType::CPU_MEMORY:
				return m_dataHost;
			#ifdef COMPILE_CUDA
			case MemoryType::CUDA_MEMORY:
				return m_dataDevice;
			#endif
			case MemoryType::CUSTOM_MEMORY:
				return m_dataHost;
			default:
				runtime_assert(false, "Unsupported memory type.");
				return nullptr;
			}
		}

		CPU_AND_GPU T* getData(Type2Type<MemoryTypeCPU>) {
			return m_dataHost;
		}

		CPU_AND_GPU const T* getData(Type2Type<MemoryTypeCPU>) const {
			return m_dataHost;
		}

		CPU_AND_GPU T* getData(Type2Type<MemoryTypeCUDA>) {
			return m_dataDevice;
		}

		CPU_AND_GPU const T* getData(Type2Type<MemoryTypeCUDA>) const {
			return m_dataDevice;
		}

		CPU_AND_GPU T* getData(Type2Type<MemoryTypeCustom>) {
			// We store the custom memory pointer as host memory pointer.
			return m_dataHost; 
		}

		CPU_AND_GPU const T* getData(Type2Type<MemoryTypeCustom>) const {
			// We store the custom memory pointer as host memory pointer.
			return m_dataHost;
		}

		/**
		 * Copies the data from one memory type to another (host = CPU, device = CUDA).
		 * Important: Can be called only from CUDA code.
		 */
		void copyHostToDevice();
		void copyDeviceToHost();

		/**
		 * Returns the size of the data.
		 */
		CPU_AND_GPU size_t getSize() const { return m_size; }

		/**
		 * Returns the byte size of the data.
		 */
		CPU_AND_GPU size_t getByteSize() const { return m_size * sizeof(T); }

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
		 * Checks whether the wrapped memory is custom memory.
		 */
		CPU_AND_GPU bool isWrappingCustomMemory() const {
			return m_wrappedMemoryType == MemoryType::CUSTOM_MEMORY;
		}

		CPU_AND_GPU MemoryType wrappedMemoryType() const {
			return m_wrappedMemoryType;
		}

	private:
		size_t m_size{ 0 };
		T* m_dataHost{ nullptr };
		T* m_dataDevice{ nullptr };

		MemoryType m_wrappedMemoryType{ MemoryType::CPU_MEMORY };
		bool m_bUpdatedHost{ false };
		bool m_bUpdatedDevice{ false };

		/**
		 * Checks for memory type allocation.
		 */
		CPU_AND_GPU bool isAllocatedHost() const { return m_dataHost != nullptr; }
		CPU_AND_GPU bool isAllocatedDevice() const { return m_dataDevice != nullptr; }

		/**
		 * Helper methods for allocating and freeing device memory.
		 * Important: Can only be called if we use GPU evaluation of contraints.
		 */
		void allocateDevice(long size);
		void freeDevice();

		/**
		 * Helper methods for allocating and freeing host memory.
		 */
		void allocateHost(long size) {
			runtime_assert(m_dataHost == nullptr, "We can only allocate host memory if previous memory was freed.");
			m_dataHost = static_cast<T*>(malloc(size * sizeof(T)));
			runtime_assert(m_dataHost != nullptr, "Host memory allocation failed.");
		}

		void freeHost() {
			if (m_dataHost) {
				free(m_dataHost);
				m_dataHost = nullptr;
			}
		}
	};

} // namespace common_utils