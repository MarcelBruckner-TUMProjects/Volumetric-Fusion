#pragma once

#ifndef _ICP_HEADER
#define _ICP_HEADER

#include "ceres/autodiff_cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"
#include "glog/logging.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "CharacteristicPoints.hpp"
#include "OptimizationProblem.hpp"
#include "CeresOptimizationProblem.hpp"
#include "PointCorrespondenceError.hpp"
#include "ReprojectionError.hpp"
#include "BundleAdjustment.hpp"
#include "NearestNeighbor.hpp"
#include <VolumetricFusion\CaptureDevice.hpp>
#include "../../../include/librealsense2/hpp/rs_processing.hpp"
#include "../../../include/librealsense2/hpp/rs_pipeline.hpp"

namespace vc::optimization {

	class PointToPointConstraint {
	public:
		PointToPointConstraint(const ceres::Vector& sourcePoint, const ceres::Vector& targetPoint, const float weight) :
			m_sourcePoint{ sourcePoint },
			m_targetPoint{ targetPoint },
			m_weight{ weight }
		{ }

		template <typename T>
		bool operator()(const T* const pose, T* residuals) const {

			const T* scale = pose;
			const T* rotation = pose + 3; 
			const T* translation = pose + 6;
			T source[3];
			T output[3];

			//std::cout << "Point source: " << T(m_sourcePoint(0)) << " " << T(m_sourcePoint(1)) << " " << T(m_sourcePoint(2)) << std::endl;
			//std::cout << "Point target: " << T(m_targetPoint(0)) << " " << T(m_targetPoint(1)) << " " << T(m_targetPoint(2)) << std::endl << std::endl;

			source[0] = scale[0] * T(m_sourcePoint(0));
			source[1] = scale[1] * T(m_sourcePoint(1));
			source[2] = scale[2] * T(m_sourcePoint(2));

			T point[3];

			ceres::AngleAxisRotatePoint(rotation, source, point);

			point[0] = point[0] + translation[0];
			point[1] = point[1] + translation[1];
			point[2] = point[2] + translation[2];

			T res1 = point[0] - T(m_targetPoint(0));
			T res2 = point[1] - T(m_targetPoint(1));
			T res3 = point[2] - T(m_targetPoint(2));

			residuals[0] = T(sqrt(LAMBDA)) * T(sqrt(m_weight)) * res1;
			residuals[1] = T(sqrt(LAMBDA)) * T(sqrt(m_weight)) * res2;
			residuals[2] = T(sqrt(LAMBDA)) * T(sqrt(m_weight)) * res3;

			return true;
		}

		static ceres::CostFunction* create(const ceres::Vector& sourcePoint, const ceres::Vector& targetPoint, const float weight) {
			return new ceres::AutoDiffCostFunction<PointToPointConstraint, 3, 9>(new PointToPointConstraint(sourcePoint, targetPoint, weight));
		}

	protected:
		const ceres::Vector m_sourcePoint;
		const ceres::Vector m_targetPoint;
		const float m_weight;
		const float LAMBDA = 0.1f;
	};

	template <typename T>
	class PoseIncrement {
	public:
		explicit PoseIncrement(T* const array) : m_array{ array } { }

		void setZero() {
			for (int i = 0; i < 9; ++i)
				m_array[i] = T(0);
		}

		T* getData() const {
			return m_array;
		}

	private:
		T* m_array;
	};

	class ICP: public virtual CeresOptimizationProblem {

	public:
		ICP(bool verbose = false, long sleepDuration = -1l) 
			:CeresOptimizationProblem(verbose, sleepDuration),
			m_nearestNeighborSearch{ std::make_unique<NearestNeighborSearchFlann>() }
			//,m_nIterations{ 50 }
		{
			translations = std::vector<std::vector<std::vector<double>>>(4);
			rotations = std::vector<std::vector<std::vector<double>>>(4);
			scales = std::vector<std::vector<std::vector<double>>>(4);
			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					translations[i].push_back(std::vector<double> { 0.0, 0.0, 0.0 });
					rotations[i].push_back(std::vector<double> { 0.0, 2 * M_PI, 0.0 });
					scales[i].push_back(std::vector<double> {1.0, 1.0, 1.0  });
				}
			}
		}

		//void setNbOfIterations(unsigned nIterations) {
		//	m_nIterations = nIterations;
		//}

		std::vector<Eigen::Vector3d> transformPoints(std::vector<Eigen::Vector3d> sourcePoints, Eigen::Matrix4d& pose) {
			
			std::vector<Eigen::Vector3d> transformedPoints;
			transformedPoints.reserve(sourcePoints.size());

			for(const auto& point : sourcePoints) {
				Eigen::Vector4d homogeneousPoint(point(0), point(1), point(2), 1.0);

				Eigen::Vector4d transformedPoint = pose * homogeneousPoint;

				Eigen::Vector3d returnPoint(transformedPoint(0), transformedPoint(1), transformedPoint(2));

				transformedPoints.push_back(returnPoint);
			}

			return transformedPoints;
		}

		//void savePLY() {

		//	//pipelines[0]->data->pointclouds;
		//	//std::shared_ptr<rs2::pipeline> pipe = pipelines[0]->pipeline->;
		//	//pipelines[0]->data->filteredColorFrames
		//	
		//	

		//	
		//	//pointcloud pc = rs2::context().create_pointcloud();
		//	//rs2::points points;

		//	//std::cout << pipelines[0]->data->filteredDepthFrames.as<rs2::video_frame>().get_data_size() << std::endl;
		//	std::cout << pipelines[0]->data->filteredDepthFrames.get_data() << std::endl;

		//	//pipelines[0]->data->pointclouds.
		//	//points.export_to_ply("temp.ply", color);// << std::endl;

		//	pipelines[0]->data->pointclouds.calculate(pipelines[0]->data->filteredDepthFrames).export_to_ply("temp1.ply", pipelines[0]->data->filteredColorFrames.as<rs2::video_frame>());// << std::endl;
		//	//std::cout << pipelines[0]->data->points.get_vertices() << std::endl;

		//	//auto frames = pipelines[0]->pipeline->wait_for_frames();
		//	//auto depth = frames.get_depth_frame();
		//	//auto color = frames.get_color_frame();

		//	//std::cout << depth.get_height() << std::endl;
		//	//std::cout << depth.get_data() << std::endl;
		//}


		bool vc::optimization::OptimizationProblem::specific_optimize() {

			if (!hasInitialization) {
				initialize();
				if (!hasInitialization) {
					return false;
				}
			}

			if (!solveErrorFunction()) {
				return false;
			}

			return true;
		}

		void initialize() {
			vc::optimization::BundleAdjustment bundleAdjustment = vc::optimization::BundleAdjustment(verbose);
			bundleAdjustment.characteristicPoints = characteristicPoints;
			if (bundleAdjustment.optimizeOnPoints()) {
				bestTransformations = bundleAdjustment.bestTransformations;
				currentRotations = bundleAdjustment.currentRotations;
				currentTranslations = bundleAdjustment.currentTranslations;
				currentScales = bundleAdjustment.currentScales;
				hasInitialization = true;
				setup();
			}
			else {
				hasInitialization = false;
			}
		}

		std::vector<Eigen::Vector3d> convertToEigenVector(rs2::points points) {
			
			//int points_size = points.size();
			int points_size = 10000;

			std::vector<Eigen::Vector3d> eigen_points;
			eigen_points.reserve(points_size);

			auto vertices = points.get_vertices();

			for (int i = 0; i < points_size; i++) {
				Eigen::Vector3d point; 
				point << vertices[i].x, vertices[i].y, vertices[i].z;
				
				eigen_points.push_back(point);
			}

			return eigen_points;

		}

		bool solveErrorFunction() {

			std::vector<Eigen::Matrix4d> initialTransformations(OptimizationProblem::bestTransformations);
			bool onlyOnce = true;

			for (int from = 0; from < characteristicPoints.size(); from++) {
				
				//ACharacteristicPoints fromPoints = characteristicPoints[from];

				auto fromDepth = pipelines[from]->data->filteredDepthFrames;
				auto fromColor = pipelines[from]->data->filteredColorFrames.as<rs2::video_frame>();
				
				pipelines[from]->data->pointclouds.map_to(fromColor);
				auto fromPoints = pipelines[from]->data->pointclouds.calculate(fromDepth);
				//auto fromVertices = fromPoints.get_vertices();

				for (int to = 0; to < characteristicPoints.size(); to++) {

					if (from == to) {
						continue;
					}

					//ACharacteristicPoints toPoints = characteristicPoints[to];

					auto toDepth = pipelines[to]->data->filteredDepthFrames;
					auto toColor = pipelines[to]->data->filteredColorFrames.as<rs2::video_frame>();

					pipelines[to]->data->pointclouds.map_to(toColor);
					auto toPoints = pipelines[to]->data->pointclouds.calculate(toDepth);
					//auto toVertices = toPoints.get_vertices();

					if (onlyOnce) {
						fromPoints.export_to_ply("temp1.ply", fromColor);
						toPoints.export_to_ply("temp2.ply", toColor);
						onlyOnce = false;
					}

					double pose[9];

					std::vector<Eigen::Vector3d> toPointsEigen = convertToEigenVector(toPoints);
					std::vector<Eigen::Vector3d> fromPointsEigen = convertToEigenVector(fromPoints);

					//Eigen::Matrix4d::Identity()
					//bestTransformations[to]
					// getCurrentTransformation(from, to)
					estimatePose(fromPointsEigen, toPointsEigen, bestTransformations[to], pose);

					std::cout << "Scale: ";
					for (int i = 0; i < 3; i++) {
						std::cout << pose[i] << " ";
						scales[from][to].push_back(pose[i]);
					}
					std::cout << std::endl;

					std::cout << "Rotation: ";
					for (int i = 0; i < 3; i++) {
						std::cout << pose[i+3] << " ";
						rotations[from][to].push_back(pose[i+3]);
					}
					std::cout << std::endl;

					std::cout << "Translation: ";
					for (int i = 0; i < 3; i++) {
						std::cout << pose[i+6] << " ";
						translations[from][to].push_back(pose[i+6]);
					}
					std::cout << std::endl;
				}
			}

			calculateTransformations();

			//std::cout << "ICP" << std::endl;
			//std::cout << vc::utils::toString("Initial", initialTransformations);
			//std::cout << vc::utils::toString("Final", bestTransformations);
			//std::cout << std::endl;

			return true;
		}

		void evaluate() {
			bestTransformations[0] = Eigen::Matrix4d::Identity();
			bestErrors[0] = 0;

			for (int i = 1; i < characteristicPoints.size(); i++)
			{
				auto paths = enumerate(i);

				std::cout << vc::utils::toString(paths);
				for (auto& path : paths)
				{

					//double error = 0;

					//for (int i = path.size() - 1; i >= 1; i--)
					//{
					//	error += calculateRelativeError(path[i], path[i - 1]);
					//}

					//if (error <= bestErrors[i]) {
					//bestErrors[i] = error;

						bestTransformations[i] = pathToMatrix(path);
						std::cout << vc::utils::toString("Transformation " + std::to_string(i), bestTransformations[i]);
					//}
				}
			}
		}

		void estimatePose(std::vector<Eigen::Vector3d>& source, std::vector<Eigen::Vector3d>& target, Eigen::Matrix4d initialPose, double *incrementArray) {

			m_nearestNeighborSearch->buildIndex(target);

			Eigen::Matrix4d estimatedPose = initialPose;

			auto poseIncrement = PoseIncrement<double>(incrementArray);
			poseIncrement.setZero();

			std::cout << "Matching points ..." << std::endl;
			clock_t begin = clock();
				
			auto transformedPoints = transformPoints(source, estimatedPose);
			auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);

			clock_t end = clock();
			std::cout << "Total number of matches " << matches.size() << std::endl;
			double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
			std::cout << "FLANN query matches completed in " << elapsedSecs << " seconds." << std::endl;

			ceres::Problem problem;
			prepareConstraints(transformedPoints, target, matches, poseIncrement, problem);

			// Configure options for the solver.
			ceres::Solver::Options options;
			configureSolver(options);
			ceres::Solver::Summary summary;

			ceres::Solve(options, &problem, &summary);
			std::cout << summary.BriefReport() << std::endl;
			//std::cout << summary.FullReport() << std::endl;

			// Update the current pose estimate (we always update the pose from the left, using left-increment notation).
			//Eigen::Matrix4d matrix = PoseIncrement<double>::convertToMatrix(poseIncrement);
			//estimatedPose = PoseIncrement<double>::convertToMatrix(poseIncrement) * estimatedPose;
			incrementArray = poseIncrement.getData();
		}

	private:
		//unsigned m_nIterations;
		std::unique_ptr<NearestNeighborSearch> m_nearestNeighborSearch;
		
		void configureSolver(ceres::Solver::Options& options) {
			// Ceres options.
			options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
			options.use_nonmonotonic_steps = false;
			options.linear_solver_type = ceres::DENSE_QR;
			options.minimizer_progress_to_stdout = 0;
			options.max_num_iterations = 50;
			options.update_state_every_iteration = true;
			options.num_threads = 8;
		}

		void prepareConstraints(std::vector<Eigen::Vector3d>& sourcePoints, std::vector<Eigen::Vector3d>& targetPoints, const std::vector<Match> matches, const PoseIncrement<double>& poseIncrement, ceres::Problem& problem) const {
			
			const unsigned nPoints = sourcePoints.size();

			for (unsigned i = 0; i < nPoints; ++i) {
				const auto match = matches[i];
				if (match.idx >= 0) {
					const auto& sourcePoint = sourcePoints[i];
					const auto& targetPoint = targetPoints[match.idx];

					//std::cout << "Point source: " << relativePoint(0) << " " << relativePoint(1) << " " << relativePoint(2) << std::endl;
					//std::cout << "Point target: " << basePoint(0) << " " << basePoint(1) << " " << basePoint(2) << std::endl << std::endl;

					ceres::CostFunction * cf = vc::optimization::PointToPointConstraint::create(sourcePoint, targetPoint, 0.5f);
					problem.AddResidualBlock(cf, nullptr, poseIncrement.getData());
				}
			}
		}

	};

	class MockICP : public ICP, public MockOptimizationProblem {
	private:
		void setupMock() {
			MockOptimizationProblem::setupMock();
			//setup();
		}

	public:
		bool vc::optimization::OptimizationProblem::specific_optimize() {
			setupMock();

			ICP::specific_optimize();

			return true;
		}
	};
}

#endif // !_ICP_HEADER