#pragma once
#include "common_utils/Common.h"
#include "MemoryContainer.h"

namespace common_utils {

	template <typename T>
	void MemoryContainer<T>::allocate(unsigned size, Type2Type<MemoryTypeCUDA>) {
		if (size != m_size) {
			// If the size is different, we always release both host and device memory and 
			// allocate only device memory.
			freeHost();
			freeDevice();

			if (size > 0) allocateDevice(size);
			m_size = size;
		}
		else if (!isAllocatedDevice()) {
			// If the size is the same as before, but device is not allocated yet, we only 
			// allocate device.
			if (size > 0) allocateDevice(size);
		}
	}

	template <typename T>
	void MemoryContainer<T>::copyHostToDevice() {
		runtime_assert(isAllocatedHost(), "Host memory needs to be allocated.");
		if (!isAllocatedDevice()) {
			allocateDevice(m_size);
		}
		CUDA_SAFE_CALL(cudaMemcpy(m_dataDevice, m_dataHost, m_size * sizeof(T), cudaMemcpyHostToDevice));
	}

	template <typename T>
	void MemoryContainer<T>::copyDeviceToHost() {
		runtime_assert(isAllocatedDevice(), "Device memory needs to be allocated.");
		if (!isAllocatedHost()) {
			allocateHost(m_size);
		}
		CUDA_SAFE_CALL(cudaMemcpy(m_dataHost, m_dataDevice, m_size * sizeof(T), cudaMemcpyDeviceToHost));
	}

	template <typename T>
	void MemoryContainer<T>::allocateDevice(long size) {
		runtime_assert(m_dataDevice == nullptr, "We can only allocate device memory if previous memory was freed.");
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dataDevice, size * sizeof(T)));
	}

	template <typename T>
	void MemoryContainer<T>::freeDevice() {
		if (m_dataDevice) {
			CUDA_SAFE_CALL(cudaFree(m_dataDevice));
			m_dataDevice = nullptr;
		}
	}

	template <typename T>
	void MemoryContainer<T>::copyDeviceToDevice(const T* otherDevicePointer) {
		runtime_assert(isAllocatedDevice(), "Device memory needs to be allocated.");
		CUDA_SAFE_CALL(cudaMemcpy(m_dataDevice, otherDevicePointer, m_size * sizeof(T), cudaMemcpyDeviceToDevice));
	}

} // namespace common_utils