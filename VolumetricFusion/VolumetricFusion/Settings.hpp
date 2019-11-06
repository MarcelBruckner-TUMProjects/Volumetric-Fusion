#pragma once
#include <VolumetricFusion\Enums.hpp>
using namespace enums;

namespace vc {
	namespace settings {
		class States {
		public:
			CaptureState captureState;
			RenderState renderState;

			States(CaptureState captureState = CaptureState::STREAMING, RenderState renderState = RenderState::ONLY_COLOR) {
				this->captureState = captureState;
				this->renderState = renderState;
			}
		};

		class OutputFolders {
		public:
			std::string capturesFolder;
			std::string recordingsFolder;
			std::string charucoFolder;

			OutputFolders(std::string capturesFolder = "captures/", std::string recordingsFolder = "single_stream_recording/", std::string charucoFolder = "charuco/") {
				this->capturesFolder = capturesFolder;
				this->recordingsFolder = recordingsFolder;
				this->charucoFolder = charucoFolder;
			}
		};

		class MarkerSettings {
		public:
			float squareLength;
			float markerLength;

			MarkerSettings(float squareLength = 0.04f, float markerLength = 0.02f) {
				this->squareLength = squareLength;
				this->markerLength = markerLength;
			}
		};
	}
}