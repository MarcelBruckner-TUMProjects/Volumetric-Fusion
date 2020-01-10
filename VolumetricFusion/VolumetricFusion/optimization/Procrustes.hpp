#pragma once
#ifndef _PROCRUSTES_HEADER
#define _PROCRUSTES_HEADER

#include "ceres/autodiff_cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"
#include "glog/logging.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <sstream>

#include "OptimizationProblem.hpp"
#include "PointCorrespondenceError.hpp"
#include "ReprojectionError.hpp"
#include "../Utils.hpp"
#include "../CaptureDevice.hpp"

namespace vc::optimization {
	class Procrustes : virtual public OptimizationProblem {
    public:
        Eigen::Matrix4d getRelativeTransformation(int camera_index) {
            return relativeTransformations[camera_index];
        }

        bool vc::optimization::OptimizationProblem::specific_optimize(std::vector<ACharacteristicPoints> characteristicPoints) {

            std::map<int, Eigen::Vector4d> centers;
            std::map<int, double> distances;

            std::map<int, Eigen::Matrix4d> rotations;

            for (int i = 0; i < characteristicPoints.size(); i++)
            {
                centers[i] = characteristicPoints[i].getCenterOfGravity();
                distances[i] = characteristicPoints[i].getAverageDistance();
            }

            for (int i = 1; i < characteristicPoints.size(); i++)
            {
                rotations[i] = characteristicPoints[i].estimateRotation(characteristicPoints[0].getFlattenedPoints(), centers[0]);

                Eigen::Matrix4d relativeTranslation = Eigen::Matrix4d::Identity();

                auto r = centers[0] - rotations[i] * centers[i];
                relativeTranslation.block<3, 1>(0, 3) = (r).block<3, 1>(0, 0);

                auto finalTransformation = relativeTranslation * rotations[i];
                
                std::cout << "Final transformation" << std::endl << finalTransformation << std::endl;
                
                relativeTransformations[i] = finalTransformation;
            }

            return true;
        }
    };

    class MockProcrustes : public Procrustes, public MockOptimizationProblem {
    public:
        ~MockProcrustes() {
            calculateFinalError();
        }

        bool vc::optimization::OptimizationProblem::specific_optimize(std::vector<ACharacteristicPoints> characteristicPoints) {
            setupMock();

            Procrustes::specific_optimize(mockCharacteristicPoints);

            std::cout << vc::utils::toString("Expected", expectedTransformations);

            return true;
        }
    };
}

#endif // !_PROCRUSTES_HEADER
