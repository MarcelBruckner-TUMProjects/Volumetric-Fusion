#pragma once
#include "common_utils/RuntimeAssertion.h"
#include "common_utils/Common.h"
#include "common_utils/meta_structures/BasicTypes.h"
#include "MemoryType.h"

namespace common_utils {

	/**
	 * Stores a memory chunk of type T. 
	 * It can store both CPU (host) and CUDA (device) memory.
	 */
	template<typename T>
	class MemoryContainer {
	public:
		using Type = T;

		/**
		 * Default constructor.
		 */
		MemoryContainer() = default;

		/**
		 * Initialization with raw array in the host/CPU memory.
		 * Important: The input array is copied, but only to the CPU memory. The GPU memory is not allocated.
		 */
		MemoryContainer(const T* data, size_t size) : m_size{ size } {
			allocateHost(size);
			copyHostToHost(data);
		}

		/**
		 * Destructor.
		 */
		~MemoryContainer() {
			freeHost();
#			ifdef COMPILE_CUDA
			freeDevice();
#			endif
		}

		/**
		 * Copy constructor.
		 */
		MemoryContainer(const MemoryContainer& other) :
			m_size{ other.m_size },
			m_bUpdatedHost{ other.m_bUpdatedHost },
			m_bUpdatedDevice{ other.m_bUpdatedDevice }
		{
			if (m_size > 0) {
				if (other.isAllocatedHost()) {
					allocateHost(m_size);
					copyHostToHost(other.m_dataHost);
				}
#				ifdef COMPILE_CUDA
				if (other.isAllocatedDevice()) {
					allocateDevice(m_size);
					copyDeviceToDevice(other.m_dataDevice);
				}
#				endif
			}
		}

		/**
		 * Move constructor.
		 */
		MemoryContainer(MemoryContainer&& other) :
			m_size{ other.m_size },
			m_dataHost{ other.m_dataHost },
			m_dataDevice{ other.m_dataDevice },
			m_bUpdatedHost{ other.m_bUpdatedHost },
			m_bUpdatedDevice{ other.m_bUpdatedDevice }
		{
			other.m_dataHost   = nullptr;
			other.m_dataDevice = nullptr;
		}

		/**
		 * Copy assignment operator.
		 */
		MemoryContainer& operator=(const MemoryContainer& other) {
			if (m_size > 0) {
				freeHost();
#				ifdef COMPILE_CUDA
				freeDevice();
#				endif
			}

			m_bUpdatedHost   = other.m_bUpdatedHost;
			m_bUpdatedDevice = other.m_bUpdatedDevice;
			m_size           = other.m_size;

			if (m_size > 0) {
				if (other.isAllocatedHost()) {
					allocateHost(m_size);
					copyHostToHost(other.m_dataHost);
				}
#				ifdef COMPILE_CUDA
				if (other.isAllocatedDevice()) { 
					allocateDevice(m_size);
					copyDeviceToDevice(other.m_dataDevice);
				}
#				endif
			}
			
			return *this;
		}

		/**
		 * Move assignment operator.
		 */
		MemoryContainer& operator=(MemoryContainer&& other) {
			if (m_size > 0) {
				freeHost();
#				ifdef COMPILE_CUDA
				freeDevice();
#				endif
			}

			m_size             = other.m_size;
			m_dataHost         = other.m_dataHost;
			m_dataDevice       = other.m_dataDevice;
			m_bUpdatedHost     = other.m_bUpdatedHost;
			m_bUpdatedDevice   = other.m_bUpdatedDevice;

			other.m_dataHost   = nullptr;
			other.m_dataDevice = nullptr;
			return *this;
		}

		/**
		 * Returns the pointer to the data, deciding about the memory location (host or device) either at run-time or compile-time.
		 */
		CPU_AND_GPU T* getData(MemoryType memoryType) {
			switch (memoryType) {
			case MemoryType::CPU_MEMORY:
				return m_dataHost;
#			ifdef COMPILE_CUDA
			case MemoryType::CUDA_MEMORY:
				return m_dataDevice;
#			endif
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

		CPU_AND_GPU T* h() {
			return m_dataHost;
		}

		CPU_AND_GPU const T* h() const {
			return m_dataHost;
		}

		CPU_AND_GPU T* d() {
			return m_dataDevice;
		}

		CPU_AND_GPU const T* d() const {
			return m_dataDevice;
		}

		/**
		 * Returns the reference to the element at the given index.
		 */
		CPU_AND_GPU T& getElement(unsigned i, Type2Type<MemoryTypeCPU>) {
			runtime_assert(i < m_size, "Index out of range");
			runtime_assert(isAllocatedHost(), "Host memory is not allocated");
			return m_dataHost[i];
		}

		CPU_AND_GPU const T& getElement(unsigned i, Type2Type<MemoryTypeCPU>) const {
			runtime_assert(i < m_size, "Index out of range");
			runtime_assert(isAllocatedHost(), "Host memory is not allocated");
			return m_dataHost[i];
		}

		CPU_AND_GPU T& getElement(unsigned i, Type2Type<MemoryTypeCUDA>) {
			runtime_assert(i < m_size, "Index out of range");
			runtime_assert(isAllocatedDevice(), "Device memory is not allocated");
			return m_dataDevice[i];
		}

		CPU_AND_GPU const T& getElement(unsigned i, Type2Type<MemoryTypeCUDA>) const {
			runtime_assert(i < m_size, "Index out of range");
			runtime_assert(isAllocatedDevice(), "Device memory is not allocated");
			return m_dataDevice[i];
		}

		CPU_AND_GPU T& h(unsigned i) {
			runtime_assert(i < m_size, "Index out of range");
			runtime_assert(isAllocatedHost(), "Host memory is not allocated");
			return m_dataHost[i];
		}

		CPU_AND_GPU const T& h(unsigned i) const {
			runtime_assert(i < m_size, "Index out of range");
			runtime_assert(isAllocatedHost(), "Host memory is not allocated");
			return m_dataHost[i];
		}

		CPU_AND_GPU T& d(unsigned i) {
			runtime_assert(i < m_size, "Index out of range");
			runtime_assert(isAllocatedDevice(), "Device memory is not allocated");
			return m_dataDevice[i];
		}

		CPU_AND_GPU const T& d(unsigned i) const {
			runtime_assert(i < m_size, "Index out of range");
			runtime_assert(isAllocatedDevice(), "Device memory is not allocated");
			return m_dataDevice[i];
		}

		/**
		 * Releases the memory, allocated by the container (both host and device), and resets
		 * the memory size to 0.
		 */
		void clear() {
#			ifdef COMPILE_CUDA
			allocate(0, true, true);
#			else
			allocate(0, true, false);
#			endif
		}

		/**
		 * Allocates the given size of the chosen memory type. If the size is different as before,
		 * the other kinds of memory are deallocated. 
		 */
		void allocate(unsigned size, bool bAllocateHost = true, bool bAllocateDevice = false) {
			if (bAllocateHost) allocate(size, Type2Type<MemoryTypeCPU>());
#			ifdef COMPILE_CUDA
			if (bAllocateDevice) allocate(size, Type2Type<MemoryTypeCUDA>());
#			else
			runtime_assert(!bAllocateDevice, "Device memory of memory container can only be allocated if compiled with flag COMPILE_CUDA.");
#			endif
		}
		
		void allocate(unsigned size, Type2Type<MemoryTypeCPU>) {
			if (size != m_size) {
				// If the size is different, we always release both host and device memory and 
				// allocate only host memory.
				freeHost();
#				ifdef COMPILE_CUDA
				freeDevice();
#				endif

				if (size > 0) allocateHost(size);
				m_size = size;
			}
			else if (!isAllocatedHost()) {
				// If the size is the same as before, but host is not allocated yet, we only 
				// allocate host.
				if (size > 0) allocateHost(size);
			}
		}

		void allocate(unsigned size, Type2Type<MemoryTypeCUDA>);

		/**
		 * Copies the data from one memory type to another (host = CPU, device = CUDA).
		 * Important: Can be called only from CUDA code.
		 */
		void copyHostToDevice();
		void copyDeviceToHost();

		/**
		 * Checks for memory type allocation.
		 */
		CPU_AND_GPU bool isAllocatedHost() const { return m_dataHost != nullptr; }
		CPU_AND_GPU bool isAllocatedDevice() const { return m_dataDevice != nullptr; }

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

	private:
		size_t m_size{ 0 };
		T* m_dataHost{ nullptr };
		T* m_dataDevice{ nullptr };

		bool m_bUpdatedHost{ false };
		bool m_bUpdatedDevice{ false };

		/**
		 * Helper methods for allocating and freeing memory.
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

		void allocateDevice(long size);
		void freeDevice();

		/**
		 * Copies the data between the same memory type (host = CPU, device = CUDA).
		 */
		void copyDeviceToDevice(const T* otherDevicePointer);

		void copyHostToHost(const T* otherHostPointer) {
			runtime_assert(isAllocatedHost(), "Host memory needs to be allocated.");
			memcpy(m_dataHost, otherHostPointer, m_size * sizeof(T));
		}
	};

} // namespace common_utils