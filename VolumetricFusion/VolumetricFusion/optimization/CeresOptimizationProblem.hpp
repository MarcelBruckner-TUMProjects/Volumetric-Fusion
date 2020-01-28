#pragma once
#ifndef _CERES_OPTIMIZATION_HEADER
#define _CERES_OPTIMIZATION_HEADER

#include "CharacteristicPoints.hpp"
#include "OptimizationProblem.hpp"
#include "PointCorrespondenceError.hpp"
#include "ReprojectionError.hpp"

namespace vc::optimization {

	class CeresOptimizationProblem : public OptimizationProblem {
	
	private:

	protected:
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

		CeresOptimizationProblem(bool verbose = false, long sleepDuration = -1l)
			:OptimizationProblem(verbose, sleepDuration) {

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

		bool vc::optimization::OptimizationProblem::specific_optimize() {
			if (!hasInitialization) {
				initialize();
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

		void clear() {
			OptimizationProblem::clear();
			needsRecalculation = true;
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



		virtual bool solveErrorFunction() = 0;
		
		virtual void initialize() = 0;

	};
}

#endif