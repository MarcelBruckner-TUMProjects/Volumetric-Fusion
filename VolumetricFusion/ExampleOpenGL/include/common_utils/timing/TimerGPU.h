#pragma once

#ifdef TIME_EXECUTION
#define TIME_GPU_START(timerName) \
	common_utils::TimerGPU timerName ; 
#define TIME_GPU_STOP(timerName) \
	std::cout << std::setw(50) << std::left << (std::string( #timerName ) + " (s): ") << std::setprecision(6) << timerName##.getElapsedTime() << std::endl;
#else
#define TIME_GPU_START(timerName) 
#define TIME_GPU_STOP(timerName) 
#endif

namespace common_utils {

	struct PrivateTimerGPU;

	class TimerGPU {
	private:
		PrivateTimerGPU* m_privateTimer;

	public:
		TimerGPU();
		~TimerGPU();

		/* (Re)starts the timer. */
		void restart();

		/* Returns the elapsed time (from latest timer start) in seconds. */
		double getElapsedTime() const;
	};

} // namespace common_utils


