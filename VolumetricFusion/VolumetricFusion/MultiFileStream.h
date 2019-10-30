void NewFunction(std::string& captures_folder, std::map<int, std::shared_ptr<rs2::pipeline>>& pipelines, std::map<int, rs2::frame>& colorized_depth_frames, std::map<int, rs2::points>& filtered_points);

void initialize(window& window_main, int& w2, int& h2, std::vector<std::string>& stream_names, const int& width_half, const int& height_half, const float& width, const float& height);

void finalize();
