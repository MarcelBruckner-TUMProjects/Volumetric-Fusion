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
            return transformations[camera_index];
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

            transformations[0] = Eigen::Matrix4d::Identity();

            for (int i = 1; i < characteristicPoints.size(); i++)
            {
                rotations[i] = characteristicPoints[i].estimateRotation(characteristicPoints[0].getFlattenedPoints(), centers[0]).inverse();


                //std::cout << vc::utils::toString("centers[0]", centers[0]);
                //std::cout << vc::utils::toString("centers[1]", centers[1]);

                Eigen::Vector4d translation = centers[0] - rotations[i] * centers[i];

                //std::cout << vc::utils::toString("Translation", translation);

                Eigen::Matrix4d finalTrans = generateTransformationMatrix(translation, 0);

                //std::cout << vc::utils::toString("Translation", finalTrans);

                auto finalTransformation = (finalTrans * rotations[i]).inverse();
                
                //std::cout << "Final transformation" << std::endl << finalTransformation << std::endl;
                
                transformations[i] = finalTransformation;
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

            return true;
        }
    };
}

#endif // !_PROCRUSTES_HEADER
