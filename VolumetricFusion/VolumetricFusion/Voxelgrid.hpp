#ifndef _VOXELGRID_HEADER_
#define _VOXELGRID_HEADER_

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <VolumetricFusion\shader.hpp>
#include <unordered_map>


namespace vc::fusion {

	float single_cube[] = {
		-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
		 0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
		 0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
		 0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
		-0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
		-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,

		-0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
		 0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
		 0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
		 0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
		-0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
		-0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,

		-0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
		-0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
		-0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
		-0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
		-0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
		-0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,

		 0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
		 0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
		 0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
		 0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
		 0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
		 0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,

		-0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
		 0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
		 0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
		 0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
		-0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
		-0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,

		-0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
		 0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
		 0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
		 0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
		-0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
		-0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f
	};

	class Voxelgrid {
	private:
		float resolution;
		glm::vec3 size;
		glm::vec3 sizeNormalized;
		glm::vec3 sizeHalf;
		glm::vec3 origin;

		std::vector<float> points;
		GLuint VBO_grid, VAO_grid;
		vc::rendering::Shader* gridShader;

		GLuint VBO_cubes, VAO_cubes, VBO_sdfs, VBO_weights;
		vc::rendering::Shader* cubeShader;

		GLuint VBO_lamp, VAO_lamp;
		vc::rendering::Shader* lampShader;

		float* voxel_grid_tsdf = nullptr;
		float* voxel_grid_weight = nullptr;

		float* tsdf;
		float* weights;
		int integratedFrames = 0;

		glm::vec3 totalMin = glm::vec3((float)INT_MAX);
		glm::vec3 totalMax = glm::vec3((float)INT_MIN);

		std::map<int, std::vector<int>> integratedFramesPerPipeline;

		int num_gridPoints;

	public:

		Voxelgrid(const float resolution = 0.1, const glm::vec3 size = glm::vec3(5.0f), const glm::vec3 origin = glm::vec3(0.0f)) {
			this->resolution = resolution;
			this->origin = origin;
			this->size = size;
			this->sizeHalf = size / 2.0f;
			this->sizeNormalized = size / resolution;
			this->sizeNormalized += glm::vec3(1.0f);

			this->num_gridPoints = (sizeNormalized.x * sizeNormalized.y * sizeNormalized.z);
					   
			for (float i = 0; i < sizeNormalized.x; i++)
			{
				auto x = i * resolution - (size.x / 2.0f) + origin.x;
				for (float j = 0; j < sizeNormalized.y; j++)
				{
					auto y = j * resolution - (size.y / 2.0f) + origin.y;
					for (float k = 0; k < sizeNormalized.z; k++)
					{
						auto z = k * resolution - (size.z / 2.0f) + origin.z;

						points.push_back(x);
						points.push_back(y);
						points.push_back(z);
					}
				}
			}

			reset();

			initializeOpenGL();
		}

		~Voxelgrid() {
			glDeleteVertexArrays(1, &VAO_grid);
			glDeleteBuffers(1, &VBO_grid);

			glDeleteVertexArrays(1, &VAO_cubes);
			glDeleteBuffers(1, &VBO_cubes);
			glDeleteBuffers(1, &VBO_sdfs);
			glDeleteBuffers(1, &VBO_weights);

			glDeleteVertexArrays(1, &VAO_lamp);
			glDeleteBuffers(1, &VBO_lamp);
		}

		void initializeOpenGL() {

			gridShader = new vc::rendering::VertexFragmentShader("shader/voxelgrid.vs", "shader/voxelgrid.fs");
			glGenVertexArrays(1, &VAO_grid);
			glGenBuffers(1, &VBO_grid);
			glBindVertexArray(VAO_grid);
			glBindBuffer(GL_ARRAY_BUFFER, VBO_grid);
			glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), points.data(), GL_STATIC_DRAW);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);

			cubeShader = new vc::rendering::VertexFragmentShader("shader/voxelgrid_cube.vert", "shader/voxelgrid_cube.frag", "shader/voxelgrid_cube.geom");
			//cubeShader = new vc::rendering::Shader("shader/voxelgrid_cube.vert", "shader/voxelgrid_cube.frag");
			glGenVertexArrays(1, &VAO_cubes);
			glGenBuffers(1, &VBO_cubes);
			glGenBuffers(1, &VBO_sdfs);
			glGenBuffers(1, &VBO_weights);
			glBindVertexArray(VAO_cubes);
			glBindBuffer(GL_ARRAY_BUFFER, VBO_cubes);
			glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(float), points.data(), GL_STATIC_DRAW);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);

			lampShader = new vc::rendering::VertexFragmentShader("shader/lamp.vert", "shader/lamp.frag");
			glGenVertexArrays(1, &VAO_lamp);
			glGenBuffers(1, &VBO_lamp);
			glBindVertexArray(VAO_lamp);
			glBindBuffer(GL_ARRAY_BUFFER, VBO_lamp);
			glBufferData(GL_ARRAY_BUFFER, sizeof(single_cube), single_cube, GL_STATIC_DRAW);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);

			glBindVertexArray(0);

			// Enable blending
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

			// Enable point size
			glEnable(GL_PROGRAM_POINT_SIZE);

			//glEnable(GL_DEPTH_TEST);
			//glDepthFunc(GL_LESS);
			//glDepthRange(0.0f, 50.0f);
			//glEnable(GL_STENCIL_TEST);
		}

		std::vector<float> *getPoints() {
			return &points;
		}

		float* getTSDFValues() {
			return tsdf;
		}

		float* getWeights() {
			return weights;
		}

		void renderGrid(glm::mat4 model, glm::mat4 view, glm::mat4 projection) {
			gridShader->use();
			gridShader->setVec3("size", size);
			gridShader->setMat4("model", model);
			gridShader->setMat4("view", view);
			gridShader->setMat4("projection", projection);
			glBindVertexArray(VAO_grid);
			glDrawArrays(GL_POINTS, 0, points.size());
			glBindVertexArray(0);
		}

		int hashFunc(glm::vec3 v) {
			int hash = (int)v.z * (int)sizeNormalized.x * (int)sizeNormalized.y + (int)v.y * (int)sizeNormalized.x + (int)v.x;
			//std::cout << v.x << "," << v.y << "," << v.z << ": " << hash << std::endl;
			return hash;
		}

		void renderField(glm::mat4 model, glm::mat4 view, glm::mat4 projection) {
			if (integratedFrames <= 0) {
				return;
			}
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			//view = glm::mat4(
			//	0.942055, -0.227064, 0.246930, 0.000000,
			//	0.000000, 0.736097, 0.676876, 0.000000,
			//	-0.335458, -0.637655, 0.693444, 0.000000,
			//	-0.389230, -0.007644, -0.465928, 1.000000
			//);
			//projection = glm::mat4(
			//	2.453140, 0.000000, 0.000000, 0.000000,
			//	0.000000, 3.270853, 0.000000, 0.000000,
			//	0.000000, 0.000000, -1.002002, -1.000000,
			//	0.000000, 0.000000, -0.200200, 0.000000
			//);

			cubeShader->use();

			glBindVertexArray(VAO_cubes);

			// bind the sdf values
			glBindBuffer(GL_ARRAY_BUFFER, VBO_sdfs);
			glBufferData(GL_ARRAY_BUFFER, num_gridPoints * sizeof(float), tsdf, GL_STREAM_DRAW);
			glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);
			glEnableVertexAttribArray(1);

			// bind the weights
			glBindBuffer(GL_ARRAY_BUFFER, VBO_weights);
			glBufferData(GL_ARRAY_BUFFER, num_gridPoints * sizeof(float), weights, GL_STREAM_DRAW);
			glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);
			glEnableVertexAttribArray(2);

			cubeShader->setMat4("model", model);
			cubeShader->setMat4("view", view);
			cubeShader->setMat4("projection", projection);
			cubeShader->setFloat("resolution", this->resolution * 0.75f);
			//glm::vec3 lightPos(1.2f, 1.0f, 2.0f);
			glm::vec3 lightPos(0.0f, 0.0f, 0.0f);
			cubeShader->setVec3("objectColor", 1.0f, 0.5f, 0.31f);
			cubeShader->setVec3("lightColor", 1.0f, 1.0f, 1.0f);
			cubeShader->setVec3("lightPos", lightPos);
			glDrawArrays(GL_POINTS, 0, points.size());  
			//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

			glBindVertexArray(0);

			// Render the light source
			lampShader->use();
			model = glm::mat4(1.0f);
			model = glm::translate(model, lightPos);
			model = glm::scale(model, glm::vec3(0.2f)); // a smaller cube
			lampShader->setMat4("model", model);
			lampShader->setMat4("view", view);
			lampShader->setMat4("projection", projection);
			glBindVertexArray(VAO_lamp);
			glDrawArrays(GL_TRIANGLES, 0, 36);

			glBindVertexArray(0);
		}

		glm::vec3 hashFuncInv(int hash) {
			int x = hash % (int)sizeNormalized.x;
			float y = (int)(hash / sizeNormalized.x) % (int)sizeNormalized.y;
			float z = (int)(hash / (sizeNormalized.x * sizeNormalized.y)) % (int)sizeNormalized.z;

			glm::vec3 result = glm::vec3(x, y, z);
			result -= sizeHalf;
			
			std::cout << hash << ":\t" << result.x << "\t" << result.y << "\t" << result.z << std::endl;

			return result;
		}


		void reset() {
			delete[] tsdf;
			delete[] weights;

			tsdf = new float[num_gridPoints];
			weights = new float[num_gridPoints];

			memset(tsdf, 0, num_gridPoints * sizeof(float));
			memset(weights, 0, num_gridPoints * sizeof(float));

			integratedFrames = 0;

			totalMin = glm::vec3((float)INT_MAX);
			totalMax = glm::vec3((float)INT_MIN);
		}

		void integrateFrameCPU(const std::shared_ptr<vc::capture::CaptureDevice> pipeline, glm::mat4 relativeTransformation, const int pipelineId, const int frameId) {
			std::cout << "Integrating " << pipelineId << " - Frame: " << frameId << std::endl;
			
			if (integratedFramesPerPipeline.count(pipelineId) <= 0) {
				integratedFramesPerPipeline[pipelineId] = std::vector<int>();
			}
			else {
				// does not really work
				if (std::find(integratedFramesPerPipeline[pipelineId].begin(), integratedFramesPerPipeline[pipelineId].end(), frameId) != integratedFramesPerPipeline[pipelineId].end()) {
					//std::cout << "Already integrated." << std::endl << std::endl;
					//return;
				}
			}

			pipeline->depth_camera->K;

			integratedFramesPerPipeline[pipelineId].push_back(frameId);

			auto depthFrame = pipeline->data->filteredDepthFrames;
			auto points = pipeline->data->pointclouds.calculate(depthFrame);
			if (!points) {
				return;
			}
			//const rs2::vertex* vertices_f = pipeline->data->points.get_vertices();
			const float* vertices_f = reinterpret_cast<const float*>(points.get_vertices());

			// insert them into the voxel grid (point by point)
			// yes, it is fucking slow
			auto size_half = size / 2.0f;

			//weights[84529] = 1.0f;
			//tsdf[84529] = 1.0f;
			//integratedFrames++;
			//return;

			for (int i = 0; i < num_gridPoints; i++) {
				auto index = i * 3;
				glm::vec3 vertex = glm::make_vec3(vertices_f + index);
				auto v = glm::vec4(vertex, 1.0);
				auto transformedVertex = relativeTransformation * v;

				// check distance and out of bounds
				// transformed vertex shifted, normalize position to positive range in order to be able to calculate the index
				auto tvs = transformedVertex + glm::vec4(size_half.x, size_half.y, size_half.z, 1.0f);
				if (tvs.x < 0 || tvs.y < 0 || tvs.z < 0 || tvs.x > size.x || tvs.y > size.y || tvs.z > size.z) {
					continue;
				}

				glm::vec3 pt_grid = tvs / resolution;
				
				int volume_idx = hashFunc(pt_grid);
				if (volume_idx < 0 || volume_idx >= num_gridPoints) {
					//std::cout << "ERROR: volume_idx out of range (" << volume_idx << ")" << std::endl;
					continue;
				}

				//float weight_old = voxel_grid_weight[volume_idx];
				float dist = fmin(1.0f, sqrtf((v.x * v.x) + (v.y * v.y) + (v.z * v.z)));
				float weight_old = weights[volume_idx];
				float weight_new = weight_old + 1.0f;
				weights[volume_idx] = weight_new;
				float tsdf_new = (tsdf[volume_idx] * weight_old + dist) / weight_new;
				tsdf[volume_idx] = 1.0f;
				//weights[volume_idx] = weight_new;
				//tsdf[volume_idx] = (tsdf[volume_idx] * weight_old + dist) / weight_new;
			}

			integratedFrames++;
		}
	};
}

#endif