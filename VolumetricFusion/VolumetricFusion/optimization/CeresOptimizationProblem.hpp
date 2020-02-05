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

		std::vector<std::vector<std::vector<double>>> translations;
		std::vector<std::vector<std::vector<double>>> rotations;
		std::vector<std::vector<std::vector<double>>> scales;

		std::vector<std::vector<std::vector<double>>> intrinsics;
		std::vector<std::vector<std::vector<double>>> distCoeffs;

		bool init(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines) {
			if (!OptimizationProblem::init(pipelines)) {
				return false;
			}

			needsRecalculation = true;

			return true;
		}



	public:

		CeresOptimizationProblem(bool verbose = false, long sleepDuration = -1l)
			:OptimizationProblem(verbose, sleepDuration) {

		}

		void clear() {
			OptimizationProblem::clear();
			needsRecalculation = true;
		}

		void reset() {
			OptimizationProblem::reset();
			setup();
			//hasProcrustesInitialization = true;
			hasInitialization = false;
		}

		Eigen::Matrix4d getTransformation(int from, int to) {
			if (needsRecalculation) {
				calculateTransformations();
			}
			return OptimizationProblem::getCurrentTransformation(from, to);
		}

		Eigen::Matrix4d getRotationMatrix(int from, int to) {
			try {
				Eigen::Vector3d rotationVector(
					rotations.at(from).at(to).at(0),
					rotations.at(from).at(to).at(1),
					rotations.at(from).at(to).at(2)
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

		Eigen::Matrix4d getTranslationMatrix(int from, int to) {
			try {
				return generateTransformationMatrix(
					translations.at(from).at(to).at(0),
					translations.at(from).at(to).at(1),
					translations.at(from).at(to).at(2),
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

		Eigen::Matrix4d getScaleMatrix(int from, int to) {
			try {
				return generateScaleMatrix(
					scales.at(from).at(to).at(0),
					scales.at(from).at(to).at(1),
					scales.at(from).at(to).at(2)
				);
			}
			catch (std::out_of_range&) {
				return Eigen::Matrix4d::Identity();
			}
			catch (std::exception&) {
				return Eigen::Matrix4d::Identity();
			}
		}

		//bool vc::optimization::OptimizationProblem::specific_optimize() {
		//	if (!hasInitialization) {
		//		initialize();
		//	}

		//	if (!solveErrorFunction()) {
		//		return false;
		//	}

		//	return true;
		//}

		void calculateTransformations() {
			for (int from = 0; from < translations.size(); from++) {
				for (int to = 0; to < translations.size(); to++)
				{
					if (from == to || !(from < 3 && to < 3)) {
						continue;
					}
					//std::stringstream ss;
					//std::cout << vc::utils::asHeader("Pre recalculation");
					//ss << "(" << from << ", " << to << ")";
					//std::cout << vc::utils::asHeader("translations" + ss.str()) << vc::utils::toString(translations[from][to]);
					//std::cout << vc::utils::asHeader("rotations" + ss.str()) << vc::utils::toString(rotations[from][to]);
					//std::cout << vc::utils::asHeader("scales" + ss.str()) << vc::utils::toString(scales[from][to]);

					//std::cout << vc::utils::toString("translations" + ss.str(), currentTranslations[from][to]);
					//std::cout << vc::utils::toString("rotations" + ss.str(), currentRotations[from][to]);
					//std::cout << vc::utils::toString("scales" + ss.str(), currentScales[from][to]);

					currentTranslations[from][to] = getTranslationMatrix(from, to);
					currentRotations[from][to] = getRotationMatrix(from, to);
					currentScales[from][to] = getScaleMatrix(from, to);

					//std::cout << vc::utils::asHeader("Post recalculation");
					//ss = std::stringstream();
					//ss << "(" << from << ", " << to << ")";
					//std::cout << vc::utils::asHeader("translations" + ss.str()) << vc::utils::toString(translations[from][to]);
					//std::cout << vc::utils::asHeader("rotations" + ss.str()) << vc::utils::toString(rotations[from][to]);
					//std::cout << vc::utils::asHeader("scales" + ss.str()) << vc::utils::toString(scales[from][to]);

					//std::cout << vc::utils::toString("translations" + ss.str(), currentTranslations[from][to]);
					//std::cout << vc::utils::toString("rotations" + ss.str(), currentRotations[from][to]);
					//std::cout << vc::utils::toString("scales" + ss.str(), currentScales[from][to]);

					//ss = std::stringstream();
				}
			}

			needsRecalculation = false;
		}

		void setup() {
			for (int from = 0; from < 4; from++)
			{
				for (int to = 0; to < 4; to++)
				{
					Eigen::Vector3d translation = currentTranslations[from][to].block<3, 1>(0, 3);
					double angle = Eigen::AngleAxisd(currentRotations[from][to].block<3, 3>(0, 0)).angle();
					Eigen::Vector3d rotation = Eigen::AngleAxisd(currentRotations[from][to].block<3, 3>(0, 0)).axis().normalized();
					rotation *= angle;
					Eigen::Vector3d scale = currentScales[from][to].diagonal().block<3, 1>(0, 0);

					for (int k = 0; k < 3; k++)
					{
						translations[from][to][k] = translation[k];
						rotations[from][to][k] = rotation[k];
						scales[from][to][k] = scale[k];
					}
				}
			}
			calculateTransformations();
		}



		virtual bool solveErrorFunction() = 0;
		
		virtual void initialize() = 0;

	};
}

#endif