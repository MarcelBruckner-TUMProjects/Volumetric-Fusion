#pragma once
#if APPLE
#include "Enums.hpp"
#else
#include "Enums.hpp"
#endif

using namespace vc::enums;
namespace vc::settings {
	class State {
	public:
		CaptureState captureState;
		RenderState renderState;

		State(CaptureState captureState = CaptureState::STREAMING, RenderState renderState = RenderState::ONLY_COLOR) {
			this->captureState = captureState;
			this->renderState = renderState;
		}
	};

	class FolderSettings {
	public:
		std::string capturesFolder;
		std::string recordingsFolder;
		std::string charucoFolder;

		FolderSettings(std::string capturesFolder = "captures/", std::string recordingsFolder = "single_stream_recording/", std::string charucoFolder = "charuco/") {
			this->capturesFolder = capturesFolder;
			this->recordingsFolder = recordingsFolder;
			this->charucoFolder = charucoFolder;
		}
	};
}