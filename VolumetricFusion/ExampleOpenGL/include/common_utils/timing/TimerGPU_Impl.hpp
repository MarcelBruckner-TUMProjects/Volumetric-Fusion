#pragma once
#include "Common.h"
#include "TimerGPU.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace common_utils {

	struct PrivateTimerGPU {
		cudaEvent_t start;
		cudaEvent_t stop;
	};

	TimerGPU::TimerGPU() {
		m_privateTimer = new PrivateTimerGPU{};
		CUDA_SAFE_CALL(cudaEventCreate(&(m_privateTimer->start)));
		CUDA_SAFE_CALL(cudaEventCreate(&(m_privateTimer->stop)));
		restart();
	}
	
	TimerGPU::~TimerGPU() { delete m_privateTimer; }
	
	void TimerGPU::restart() {	
		CUDA_SAFE_CALL(cudaEventRecord(m_privateTimer->start, 0));
	}

	double TimerGPU::getElapsedTime() const {
		float time;
		CUDA_SAFE_CALL(cudaEventRecord(m_privateTimer->stop, 0));
		CUDA_SAFE_CALL(cudaEventSynchronize(m_privateTimer->stop));
		CUDA_SAFE_CALL(cudaEventElapsedTime(&time, m_privateTimer->start, m_privateTimer->stop));
		return double(time) / 1000.0;
	}

} // namespace common_utils


