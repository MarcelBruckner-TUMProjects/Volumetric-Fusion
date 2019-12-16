#pragma once

#ifdef TIME_EXECUTION
#define TIME_CPU_START(timerName) \
	common_utils::TimerCPU timerName ; 
#define TIME_CPU_STOP(timerName) \
	std::cout << std::setw(50) << std::left << (std::string( #timerName ) + " (s): ") << std::setprecision(6) << timerName##.getElapsedTime() << std::endl;
#else
#define TIME_CPU_START(timerName) 
#define TIME_CPU_STOP(timerName) 
#endif

namespace common_utils {
	
	class TimerCPU {
	private:
		double m_startTime{ 0.0 };

	public:
		TimerCPU() : m_startTime{ getTime() } {}

		/* (Re)starts the timer. */
		void restart() {
			m_startTime = getTime();
		}

		/* Returns the elapsed time (from latest timer start) in seconds. */
		double getElapsedTime() const {
			return getTime() - m_startTime;
		}

	private:
		/* Returns the time in seconds. */
		double getTime() const;
	};

} // namespace common_utils


