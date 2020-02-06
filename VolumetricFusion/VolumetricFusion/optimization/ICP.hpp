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
#include "PointCorrespondenceError.hpp"
#include "ReprojectionError.hpp"
#include "BundleAdjustment.hpp"
#include <VolumetricFusion\CaptureDevice.hpp>

namespace vc::optimization {

	class ICP: public BundleAdjustment {

	public:
		ICP(bool verbose = false, long sleepDuration = -1l) 
			:BundleAdjustment(verbose, sleepDuration)
		{ }

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

		bool vc::optimization::OptimizationProblem::specific_optimize() {
			if (!hasInitialization) {
				initialize();
				if (!hasInitialization) {
					return false;
				}
			}

			setCharacteristicPoints(pipelines);

			if (!solveErrorFunction()) {
				return false;
			}

			return true;
		}

		void setCharacteristicPoints(std::vector<std::shared_ptr<vc::capture::CaptureDevice>> pipelines, bool* activeOnes = new bool[4]{ true, true, true, true }) {
			characteristicPoints.clear();

			std::unique_ptr<NearestNeighborSearch> m_nearestNeighborSearch = std::make_unique<NearestNeighborSearchFlann>();

			std::vector<vc::optimization::ACharacteristicPoints> current(pipelines.size());
			for (int i = 0; i < pipelines.size(); i++)
			{
				if (activeOnes[i]) {
					current[i] = FLANNCharacteristicPoints(pipelines[i], i, bestTransformations[i], m_nearestNeighborSearch);
				}
			}
			characteristicPoints = current;
		}

	private:
		

	};
}

#endif // !_ICP_HEADER