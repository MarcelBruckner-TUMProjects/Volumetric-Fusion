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
#include <VolumetricFusion\CaptureDevice.hpp>

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

			const T* rotation = pose; 
			const T* translation = pose + 3;

			T source[3];
			T output[3];

			//std::cout << "Point source: " << T(m_sourcePoint(0)) << " " << T(m_sourcePoint(1)) << " " << T(m_sourcePoint(2)) << std::endl;
			//std::cout << "Point target: " << T(m_targetPoint(0)) << " " << T(m_targetPoint(1)) << " " << T(m_targetPoint(2)) << std::endl << std::endl;

			source[0] = T(m_sourcePoint(0));
			source[1] = T(m_sourcePoint(1));
			source[2] = T(m_sourcePoint(2));

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
			return new ceres::AutoDiffCostFunction<PointToPointConstraint, 3, 6>(new PointToPointConstraint(sourcePoint, targetPoint, weight));
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
			for (int i = 0; i < 6; ++i)
				m_array[i] = T(0);
		}

		T* getData() const {
			return m_array;
		}

		/**
		 * Applies the pose increment onto the input point and produces transformed output point.
		 * Important: The memory for both 3D points (input and output) needs to be reserved (i.e. on the stack)
		 * beforehand).
		 */
		void apply(T* inputPoint, T* outputPoint) const {
			// pose[0,1,2] is angle-axis rotation.
			// pose[3,4,5] is translation.
			const T* rotation = m_array + 0;
			const T* translation = m_array + 3;

			T temp[3];

			ceres::AngleAxisRotatePoint(rotation, inputPoint, temp);

			outputPoint[0] = temp[0] + translation[0];
			outputPoint[1] = temp[1] + translation[1];
			outputPoint[2] = temp[2] + translation[2];
		}

		/**
		 * Converts the pose increment with rotation in SO3 notation and translation as 3D vector into
		 * transformation 4x4 matrix.
		 */
		static Eigen::Matrix4d convertToMatrix(const PoseIncrement<double>& poseIncrement) {
			// pose[0,1,2] is angle-axis rotation.
			// pose[3,4,5] is translation.
			double* pose = poseIncrement.getData();
			double* rotation = pose;
			double* translation = pose + 3;

			// Convert the rotation from SO3 to matrix notation (with column-major storage).
			double rotationMatrix[9];
			ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);

			// Create the 4x4 transformation matrix.
			Eigen::Matrix4d matrix;
			matrix.setIdentity();
			matrix(0, 0) = float(rotationMatrix[0]);	matrix(0, 1) = float(rotationMatrix[3]);	matrix(0, 2) = float(rotationMatrix[6]);	matrix(0, 3) = float(translation[0]);
			matrix(1, 0) = float(rotationMatrix[1]);	matrix(1, 1) = float(rotationMatrix[4]);	matrix(1, 2) = float(rotationMatrix[7]);	matrix(1, 3) = float(translation[1]);
			matrix(2, 0) = float(rotationMatrix[2]);	matrix(2, 1) = float(rotationMatrix[5]);	matrix(2, 2) = float(rotationMatrix[8]);	matrix(2, 3) = float(translation[2]);

			return matrix;
		}

	private:
		T* m_array;
	};

	class ICP: public virtual CeresOptimizationProblem {

		bool hasInitialization = false;
		bool needsRecalculation = true;

		const int num_translation_parameters = 3;
		const int num_rotation_parameters = 3;
		const int num_scale_parameters = 3;
		const int num_intrinsic_parameters = 4;
		const int num_distCoeff_parameters = 4;

		std::vector<std::vector<double>> translations;
		std::vector<std::vector<double>> rotations;
		std::vector<std::vector<double>> scales;

		std::vector<std::vector<double>> intrinsics;
		std::vector<std::vector<double>> distCoeffs;

	protected:

		bool init(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines) {
			if (!OptimizationProblem::init(pipelines)) {
				return false;
			}

			//for (int i = 0; i < pipelines.size(); i++) {
			//    std::vector<double> translation;
			//    for (int j = 0; j < num_translation_parameters; j++)
			//    {
			//        translation.emplace_back(pipelines[i]->chArUco->translation[j]);
			//    }

			//    std::vector<double> rotation;
			//    for (int j = 0; j < num_rotation_parameters; j++)
			//    {
			//        rotation.emplace_back(pipelines[i]->chArUco->rotation[j]);
			//    }

			//    translations[i] = (translation);
			//    rotations[i] = (rotation);
			//    scales[i] = std::vector<double>{ 1.0, 1.0, 1.0 };
			//}

			needsRecalculation = true;

			//calculateRelativeTransformations();
			return true;
		}

	public:
		ICP(bool verbose = false, long sleepDuration = -1l) 
			:CeresOptimizationProblem(verbose, sleepDuration),
			m_nIterations{ 20 }
		{
			for (int i = 0; i < 4; i++)
			{
				translations.push_back(std::vector<double> { 0.0, 0.0, 0.0 });
				rotations.push_back(std::vector<double> { 0.0, 2 * M_PI, 0.0 });
				scales.push_back(std::vector<double> {1.0, 1.0, 1.0  });

				intrinsics.push_back(std::vector<double> { 0.0, 0.0, 0.0, 0.0 });
				distCoeffs.push_back(std::vector<double> { 0.0, 0.0, 0.0, 0.0 });
			}
		}

		void setNbOfIterations(unsigned nIterations) {
			m_nIterations = nIterations;
		}

		std::map<unsigned long long, Eigen::Vector4d> transformPoints(std::map<unsigned long long, Eigen::Vector4d>& sourcePoints, std::vector<unsigned long long>& matchingHashes, Eigen::Matrix4d& pose) {
			
			std::map<unsigned long long, Eigen::Vector4d>& transformedPoints = std::map<unsigned long long, Eigen::Vector4d>();
			
			for(auto& hash : matchingHashes) {
				Eigen::Vector4d point = pose * sourcePoints[hash];
				transformedPoints[hash] = point;
			}

			return transformedPoints;
		}

		bool specific_optimize() {
			
			if (!hasInitialization) {
				initialize();
				//return false;
			}

			if (!solveErrorFunction()) {
				return false;
			}

			return true;

		}

		void calculateTransformations() {
			for (int i = 0; i < translations.size(); i++) {
				currentTranslations[i] = getTranslationMatrix(i);
				currentRotations[i] = getRotationMatrix(i);
				currentScales[i] = getScaleMatrix(i);
			}

			needsRecalculation = false;
		}

		Eigen::Matrix4d getTransformation(int camera_index) {
			if (needsRecalculation) {
				calculateTransformations();
			}
			return getCurrentTransformation(camera_index);
		}

		Eigen::Matrix4d getRotationMatrix(int camera_index) {
			try {
				Eigen::Vector3d rotationVector(
					rotations.at(camera_index).at(0),
					rotations.at(camera_index).at(1),
					rotations.at(camera_index).at(2)
				);
				return generateTransformationMatrix(0.0, 0.0, 0.0, rotationVector.norm(), rotationVector.normalized());
			}
			catch (std::out_of_range&) {
				return Eigen::Matrix4d::Identity();
			}
			catch (std::exception&) {
				return Eigen::Matrix4d::Identity();
			}
		}

		Eigen::Matrix4d getTranslationMatrix(int camera_index) {
			try {
				return generateTransformationMatrix(
					translations.at(camera_index).at(0),
					translations.at(camera_index).at(1),
					translations.at(camera_index).at(2),
					0.0, Eigen::Vector3d::Identity()
				);
			}
			catch (std::out_of_range&) {
				return Eigen::Matrix4d::Identity();
			}
			catch (std::exception&) {
				return Eigen::Matrix4d::Identity();
			}
		}

		Eigen::Matrix4d getScaleMatrix(int camera_index) {
			try {
				return generateScaleMatrix(
					scales.at(camera_index).at(0),
					scales.at(camera_index).at(1),
					scales.at(camera_index).at(2)
				);
			}
			catch (std::out_of_range&) {
				return Eigen::Matrix4d::Identity();
			}
			catch (std::exception&) {
				return Eigen::Matrix4d::Identity();
			}
		}

		void setup() {
			for (int i = 0; i < 4; i++)
			{
				Eigen::Vector3d translation = currentTranslations[i].block<3, 1>(0, 3);
				double angle = Eigen::AngleAxisd(currentRotations[i].block<3, 3>(0, 0)).angle();
				Eigen::Vector3d rotation = Eigen::AngleAxisd(currentRotations[i].block<3, 3>(0, 0)).axis().normalized();
				rotation *= angle;
				Eigen::Vector3d scale = currentScales[i].diagonal().block<3, 1>(0, 0);

				for (int j = 0; j < 3; j++)
				{
					translations[i][j] = translation[j];
					rotations[i][j] = rotation[j];
					scales[i][j] = scale[j];
				}
			}
			calculateTransformations();
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

		bool solveErrorFunction() {

			//std::vector<Eigen::Matrix4d> initialTransformations = bestTransformations;
			std::vector<Eigen::Matrix4d> initialTransformations(OptimizationProblem::bestTransformations);

			int baseId = 0;

			for (int relativeId = 1; relativeId < characteristicPoints.size(); relativeId++) {
								
				Eigen::Matrix4d pose = estimatePose(characteristicPoints[relativeId], characteristicPoints[baseId], Eigen::Matrix4d::Identity());

				Eigen::Matrix3d rotationMatrix = pose.block<3, 3>(0, 0);
				Eigen::Vector3d translation = pose.block<3, 1>(0, 3);
				
				double rotation[9];
				double angle_axis[3];

				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 3; j++) {
						rotation[i*3 + j] = rotationMatrix(j, i);
					}
				}

				ceres::RotationMatrixToAngleAxis(rotation, angle_axis);

				for (int i = 0; i < 3; i++) {
					rotations[relativeId].push_back(angle_axis[i]);
				}

				for (int i = 0; i < 3; i++) {
					translations[relativeId].push_back(translation(i));
				}
			}

			calculateTransformations();

			std::cout << "ICP" << std::endl;
			std::cout << vc::utils::toString("Initial", initialTransformations);
			std::cout << vc::utils::toString("Final", bestTransformations);
			std::cout << std::endl;

			return true;
		}

		Eigen::Matrix4d estimatePose(ACharacteristicPoints& source, ACharacteristicPoints& target, Eigen::Matrix4d initialPose) {

			std::vector<unsigned long long> matchingHashes = vc::utils::findOverlap(source.getHashes(verbose), target.getHashes(verbose));

			auto& sourcePoints = source.getFilteredPoints(matchingHashes, verbose);
			auto& targetPoints = target.getFilteredPoints(matchingHashes, verbose);

			Eigen::Matrix4d estimatedPose = initialPose;

			double incrementArray[6];
			auto poseIncrement = PoseIncrement<double>(incrementArray);
			poseIncrement.setZero();

			for (int i = 0; i < m_nIterations; ++i) {

				//std::cout << "Iteration: " << i << std::endl;

				auto transformedPoints = transformPoints(sourcePoints, matchingHashes, estimatedPose);

				ceres::Problem problem;
				prepareConstraints(transformedPoints, targetPoints, matchingHashes, poseIncrement, problem);

				// Configure options for the solver.
				ceres::Solver::Options options;
				configureSolver(options);
				ceres::Solver::Summary summary;

				ceres::Solve(options, &problem, &summary);
				//std::cout << summary.BriefReport() << std::endl;
				//std::cout << summary.FullReport() << std::endl;

				// Update the current pose estimate (we always update the pose from the left, using left-increment notation).
				Eigen::Matrix4d matrix = PoseIncrement<double>::convertToMatrix(poseIncrement);
				estimatedPose = PoseIncrement<double>::convertToMatrix(poseIncrement) * estimatedPose;
				poseIncrement.setZero();

				//std::cout << "Optimization iteration done." << std::endl;
			}

			//std::cout << "Pose: " << estimatedPose << std::endl << std::endl;

			return estimatedPose;
		}

	private:
		unsigned m_nIterations;
		
		void configureSolver(ceres::Solver::Options& options) {
			// Ceres options.
			options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
			options.use_nonmonotonic_steps = false;
			options.linear_solver_type = ceres::DENSE_QR;
			options.minimizer_progress_to_stdout = 0;
			options.max_num_iterations = 1;
			options.update_state_every_iteration = true;
			options.num_threads = 8;
		}

		void prepareConstraints(std::map<unsigned long long, Eigen::Vector4d>& sourcePoints, std::map<unsigned long long, Eigen::Vector4d>& targetPoints, std::vector<unsigned long long>& matchingHashes, const PoseIncrement<double>& poseIncrement, ceres::Problem& problem) const {
			
			//const unsigned nPoints = sourcePoints.size();

			for (auto& hash : matchingHashes)
			{
				auto relativePoint = sourcePoints[hash];
				auto basePoint = targetPoints[hash];

				bool valid = true;
				for (int i = 0; i < 3; i++)
				{
					if (std::abs(relativePoint[i]) > 10e5) {
						valid = false;
						break;
					}
				}
				if (!valid) {
					continue;
				}

				for (int i = 0; i < 3; i++)
				{
					if (std::abs(basePoint[i]) > 10e5) {
						valid = false;
						break;
					}
				}

				if (!valid) {
					continue;
				}

				//std::cout << "Point source: " << relativePoint(0) << " " << relativePoint(1) << " " << relativePoint(2) << std::endl;
				//std::cout << "Point target: " << basePoint(0) << " " << basePoint(1) << " " << basePoint(2) << std::endl << std::endl;

				ceres::CostFunction* cf = vc::optimization::PointToPointConstraint::create(relativePoint, basePoint, 0.5f);
				problem.AddResidualBlock(cf, nullptr, poseIncrement.getData());
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