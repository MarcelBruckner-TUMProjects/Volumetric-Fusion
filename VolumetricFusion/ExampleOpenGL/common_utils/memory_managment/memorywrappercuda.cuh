#pragma once
#include "MemoryWrapper.h"
#include "common_utils/Common.h"

namespace common_utils {

	template <typename T>
	void MemoryWrapper<T>::allocateDevice(long size) {
		runtime_assert(m_dataDevice == nullptr, "We can only allocate device memory if previous memory was freed.");
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dataDevice, size * sizeof(T)));
	}

	template <typename T>
	void MemoryWrapper<T>::freeDevice() {
		if (m_dataDevice) {
			CUDA_SAFE_CALL(cudaFree(m_dataDevice));
			m_dataDevice = nullptr;
		}
	}

	template <typename T>
	void MemoryWrapper<T>::copyHostToDevice() {
		runtime_assert(m_wrappedMemoryType != MemoryType::CUSTOM_MEMORY, "Updating custom memory is not possible.");
		runtime_assert(isAllocatedHost(), "Host memory needs to be allocated.");
		if (!isAllocatedDevice()) {
			allocateDevice(m_size);
		}
		CUDA_SAFE_CALL(cudaMemcpy(m_dataDevice, m_dataHost, m_size * sizeof(T), cudaMemcpyHostToDevice));
	}

	template <typename T>
	void MemoryWrapper<T>::copyDeviceToHost() {
		runtime_assert(m_wrappedMemoryType != MemoryType::CUSTOM_MEMORY, "Updating custom memory is not possible.");
		runtime_assert(isAllocatedDevice(), "Device memory needs to be allocated.");
		if (!isAllocatedHost()) {
			allocateHost(m_size);
		}
		CUDA_SAFE_CALL(cudaMemcpy(m_dataHost, m_dataDevice, m_size * sizeof(T), cudaMemcpyDeviceToHost));
	}

} // namespace common_utils