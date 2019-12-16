#pragma once
#include <map>

#include "TimerCPU.h"
#include "common_utils/RuntimeAssertion.h"
#include "common_utils/Utils.h"

#ifndef TIME             
#define TIME(title) common_utils::timing::start(title); for (bool bExecute = true; bExecute; common_utils::timing::stop(title), bExecute = false)
#endif

namespace common_utils {
	namespace timing {

		struct TimingSession {
		public:
			void start() {
				m_timer.restart();
			}

			void stop() {
				addMeasurement(m_timer.getElapsedTime());
			}

			double getAverageDuration() const {
				return m_averageDuration;
			}

		private:
			double m_averageDuration{ 0.f };
			unsigned m_nMeasurements{ 0 };
			TimerCPU m_timer;

			void addMeasurement(double duration) {
				m_averageDuration = (double(m_nMeasurements) / (m_nMeasurements + 1))  * m_averageDuration + duration / (m_nMeasurements + 1);
				m_nMeasurements++;
			}
		};

		/**
		 * Class that abstracts away the executing timing.
		 * Important: For efficiency reasons it is not thread-safe.
		 */
		class Timing {
		public:
			static Timing& getInstance() {
				static Timing s;
				return s;
			}

			static Timing& get() {
				return getInstance();
			}

			void start(std::string sessionName) {
				auto it = m_timingSessions.find(sessionName);
				if (it != m_timingSessions.end()) {
					it->second.start();
				}
				else {
					m_timingSessions[sessionName] = TimingSession{};
					m_timingSessions[sessionName].start();
				}
			}

			void stop(std::string sessionName) {
				auto it = m_timingSessions.find(sessionName);
				if (it != m_timingSessions.end()) {
					it->second.stop();
				}
				else {
					runtime_assert(false, "You stopped a timing session that wasn't started before: " + sessionName);
				}
			}

			double averageTime(std::string sessionName) {
				auto it = m_timingSessions.find(sessionName);
				if (it != m_timingSessions.end()) {
					return it->second.getAverageDuration();
				}
				else {
					runtime_assert(false, "You wanted an average time of a session that wasn't timed before: " + sessionName);
					return -1.0;
				}
			}

			void report(std::string filters) {
				cout << endl << "TIMING REPORT:" << endl;
				vector<std::string> filterArray = split(filters, ';');
				for (const auto& it : m_timingSessions) {
					const std::string& sessionName = it.first;
					const TimingSession& session = it.second;

					bool bPrinted = false;
					for (auto filter : filterArray) {
						if (sessionName.find(filter) == 0 && !bPrinted) {
							cout << std::setw(60) << std::left << sessionName << std::setprecision(6) << session.getAverageDuration() << endl;
							bPrinted = true;
						}
					}
					if (filters.empty())
						cout << std::setw(80) << std::left << sessionName << std::setprecision(6) << session.getAverageDuration() << endl;
				}
				cout << endl;
			}

			/**
			 * If enabled, the start and stop of every timer is printed out to standard output, which
			 * makes debugging easier.
			 */
			void debugExecution(bool bDebugExecution) {
				m_bDebugExecution = bDebugExecution;
			}

			bool debugExecution() {
				return m_bDebugExecution;
			}

			/**
			 * Indents for debugging.
			 */
			void increaseIndent() { m_indentSize++; }
			void decreaseIndent() { m_indentSize--; }
			std::string getIndent() { return std::string(2 * m_indentSize, ' '); }

		private:
			std::map<std::string, TimingSession> m_timingSessions;

			bool m_bDebugExecution;
			unsigned m_indentSize;
		};

		/**
		 * Starts a timing session.
		 * @param	sessionName		Name of the session (e.g. correspondence_finding)
		 */
		inline void start(std::string sessionName) {
			if (Timing::get().debugExecution()) {
				cout << Timing::get().getIndent() << "START: " << sessionName << endl;
				Timing::get().increaseIndent();
			}
			Timing::get().start(sessionName);
		}

		/**
		 * Stops a timing session.
		 * @param	sessionName		Name of the session (e.g. correspondence_finding)
		 */
		inline void stop(std::string sessionName) {
			Timing::get().stop(sessionName);
			if (Timing::get().debugExecution()) {
				Timing::get().decreaseIndent();
				cout << Timing::get().getIndent() << "STOP: " << sessionName << endl;
			}
		}

		/**
		 * Returns the average execution time for the given session.
		 * @param	sessionName		Name of the session (e.g. correspondence_finding)
		 */
		inline double averageTime(std::string sessionName) {
			return Timing::get().averageTime(sessionName);
		}

		/**
		 * Prints out a timing report.
		 */
		inline void print(std::string filters = "") {
			Timing::get().report(filters);
		}

		/**
		 * Enables/disables automatic debugging (start and stop of every timer is printed out to
		 * standard output, which simplifies debugging).
		 */
		inline void debugExecution(bool bDebugExecution) {
			Timing::get().debugExecution(bDebugExecution);
		}

	} // namespace timing
} // namespace common_utils