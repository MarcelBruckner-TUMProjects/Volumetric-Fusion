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
        Procrustes(bool verbose = false, long sleepDuration = -1l) : OptimizationProblem(verbose, sleepDuration)
        {}

        bool vc::optimization::OptimizationProblem::specific_optimize() {

            currentTranslations[0] = Eigen::Matrix4d::Identity();
            currentRotations[0] = Eigen::Matrix4d::Identity();
            currentScales[0] = Eigen::Matrix4d::Identity();

            for (int i = 1; i < characteristicPoints.size(); i++)
            {
                if (!calculateRelativetranformation(characteristicPoints[i], characteristicPoints[0], i)) {
                    return false;
                }
            }

            return true;
        }

        bool calculateRelativetranformation(ACharacteristicPoints& source, ACharacteristicPoints& target, int index) {
            std::vector<unsigned long long> matchingHashes = vc::utils::findOverlap(source.getHashes(verbose), target.getHashes(verbose));

            if (matchingHashes.size() <= 4) {
                std::cerr << "At least 5 points are needed for Procrustes. Provided: " << matchingHashes.size() << std::endl;

                currentTranslations[index] = Eigen::Matrix4d::Identity();
                currentRotations[index] = Eigen::Matrix4d::Identity();
                currentScales[index] = Eigen::Matrix4d::Identity();
                return false;
            }

            auto& sourcePoints = source.getFilteredPoints(matchingHashes, verbose);
            auto& targetPoints = target.getFilteredPoints(matchingHashes, verbose);

            auto& sourceMean = getCenterOfGravity(sourcePoints, verbose);
            auto& targetMean = getCenterOfGravity(targetPoints, verbose);

            double sourceDistance = getAverageDistance(sourcePoints, sourceMean, verbose);
            double targetDistance = getAverageDistance(targetPoints, targetMean, verbose);

            auto& rotation = estimateRotation(
                matchingHashes,
                sourcePoints, sourceMean, sourceDistance,
                targetPoints, targetMean, targetDistance,
                verbose
            );

            std::stringstream ss;

            Eigen::Vector4d translation = sourceMean - rotation * targetMean;
            Eigen::Matrix4d finalTrans = generateTransformationMatrix(translation, 0);

            ss << vc::utils::toString("Translation", finalTrans);

            Eigen::Matrix4d scaling = (sourceDistance / targetDistance) * Eigen::Matrix4d::Identity();
            scaling.bottomRightCorner<1, 1>() << 1.0;
            
            ss << vc::utils::toString("Scaling", scaling);

            auto finalTransformation = Eigen::Matrix4d(finalTrans * rotation * scaling);

            ss << vc::utils::toString("Final transformation", finalTransformation);

            if (verbose) {
                std::cout << ss.str();
            }

            currentTranslations[index] = finalTrans;
            currentRotations[index] = rotation;
            currentScales[index] = scaling;

            return true;
        }

        Eigen::Matrix4d estimateRotation(std::vector<unsigned long long> hashes,
            std::map<unsigned long long, Eigen::Vector4d> sourcePoints, Eigen::Vector4d sourceMean, double sourceDistance,
            std::map<unsigned long long, Eigen::Vector4d> targetPoints, Eigen::Vector4d targetMean, double targetDistance,
            bool verbose = true) {

            std::stringstream ss;

            ss << vc::utils::toString("Uncentered Source points:", sourcePoints) << std::endl;
            ss << vc::utils::toString("Uncentered Target points:", targetPoints) << std::endl;
            
            std::map<unsigned long long, Eigen::Vector4d> filteredSourcePoints;
            std::map<unsigned long long, Eigen::Vector4d> filteredTargetPoints;

            for (auto& hash : hashes)
            {
                filteredSourcePoints[hash] = (sourcePoints[hash] - sourceMean) / sourceDistance * targetDistance;
                filteredTargetPoints[hash] = targetPoints[hash] - targetMean;
            }

            ss << vc::utils::toString("Mean centered Source points:", filteredSourcePoints) << std::endl;
            ss << vc::utils::toString("Mean centered Target points:", filteredTargetPoints) << std::endl;

            Eigen::Matrix4d A = Eigen::Matrix4d();
            A <<
                filteredTargetPoints[hashes[4]] - filteredTargetPoints[hashes[0]],
                filteredTargetPoints[hashes[3]] - filteredTargetPoints[hashes[0]],
                filteredTargetPoints[hashes[2]] - filteredTargetPoints[hashes[0]],
                filteredTargetPoints[hashes[1]] - filteredTargetPoints[hashes[0]];
            Eigen::Matrix4d B = Eigen::Matrix4d();
            B <<
                filteredSourcePoints[hashes[4]] - filteredSourcePoints[hashes[0]],
                filteredSourcePoints[hashes[3]] - filteredSourcePoints[hashes[0]],
                filteredSourcePoints[hashes[2]] - filteredSourcePoints[hashes[0]],
                filteredSourcePoints[hashes[1]] - filteredSourcePoints[hashes[0]];

            Eigen::JacobiSVD<Eigen::Matrix4d> svd = Eigen::JacobiSVD<Eigen::Matrix4d>(B * A.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);

            auto result = Eigen::Matrix4d(svd.matrixU() * svd.matrixV().transpose());

            ss << vc::utils::toString("Rotation: ", result);

            if (verbose) {
                std::cout << ss.str();
            }

            return result;

            //return Eigen::Matrix4d::Identity();
        }

        Eigen::Vector4d getCenterOfGravity(std::map<unsigned long long, Eigen::Vector4d> points, bool verbose = true) {
            Eigen::Vector4d centerOfGravity(0, 0, 0, 0);

            for (auto& point : points)
            {
                centerOfGravity += point.second;
            }

            centerOfGravity /= points.size();

            if (verbose) {
                std::cout << vc::utils::asHeader("Center of gravity:") << centerOfGravity << std::endl;
            }
            return centerOfGravity;
        }

        double getAverageDistance(std::map<unsigned long long, Eigen::Vector4d> points, Eigen::Vector4d mean, bool verbose = true) {
            double distance = 0;

            for (auto& point : points)
            {
                distance += (mean - point.second).norm();
            }

            distance /= points.size();

            if (verbose) {
                std::cout << vc::utils::asHeader("Average distance:") << distance << std::endl;
            }

            return distance;
        }
    };

    class MockProcrustes : public Procrustes, public MockOptimizationProblem {
    public:
        ~MockProcrustes(){
            calculateRelativeError(1, 0);
        }

        bool vc::optimization::OptimizationProblem::specific_optimize() {
            setupMock();

            Procrustes::specific_optimize();

            return true;
        }
    };
}

#endif // !_PROCRUSTES_HEADER