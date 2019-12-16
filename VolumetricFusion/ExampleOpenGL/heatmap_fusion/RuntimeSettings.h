#ifndef RUNTIMESETTINGS_H
#define RUNTIMESETTINGS_H

#include <string>

namespace heatmap_fusion {

	class Settings {
	public:
		std::string s_shaderPath{ PROJECT_DIR + std::string("/shaders") };

		static Settings& getInstance() {
			static Settings s;
			return s;
		}

		static Settings& get() {
			return getInstance();
		}
	};

} // namespace heatmap_fusion

#endif //RUNTIMESETTINGS_H