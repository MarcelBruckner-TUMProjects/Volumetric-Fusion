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

            bool success = true;

            for (int i = 0; i < characteristicPoints.size(); i++)
            {
                for (int j = 0; j < characteristicPoints.size(); j++)
                {
                    if (i == j) {
                        currentTranslations[i][j] = Eigen::Matrix4d::Identity();
                        currentRotations[i][j] = Eigen::Matrix4d::Identity();
                        currentScales[i][j] = Eigen::Matrix4d::Identity();
                    }
                    if (!calculateRelativetranformation(characteristicPoints[i], characteristicPoints[j], &currentTranslations[i][j], &currentRotations[i][j], &currentScales[i][j])) {
                        success = false;
                    }
                }
            }

            return success;
        }

        bool calculateRelativetranformation(ACharacteristicPoints& source, ACharacteristicPoints& target, Eigen::Matrix4d* finalTranslation, Eigen::Matrix4d* finalRotation, Eigen::Matrix4d* finalScale) {
            std::vector<unsigned long long> matchingHashes = vc::utils::findOverlap(source.getHashes(verbose), target.getHashes(verbose));

            if (matchingHashes.size() <= 4) {
                std::cerr << "At least 5 points are needed for Procrustes. Provided: " << matchingHashes.size() << std::endl;

                *finalTranslation   = Eigen::Matrix4d::Zero();
                *finalRotation      = Eigen::Matrix4d::Zero();
                *finalScale         = Eigen::Matrix4d::Zero();
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

            Eigen::Vector4d _translation = sourceMean - rotation * targetMean;
            Eigen::Matrix4d translation = generateTransformationMatrix(_translation, 0);

            ss << vc::utils::toString("Translation", translation);

            Eigen::Matrix4d scaling = (sourceDistance / targetDistance) * Eigen::Matrix4d::Identity();
            scaling.bottomRightCorner<1, 1>() << 1.0;
            
            ss << vc::utils::toString("Scaling", scaling);

            auto finalTransformation = Eigen::Matrix4d(translation * rotation * scaling);

            ss << vc::utils::toString("Final transformation", finalTransformation);

            if (verbose) {
                std::cout << ss.str();
            }

            *finalTranslation = translation;
            *finalRotation = rotation;
            *finalScale = scaling;

            return true;
        }

        Eigen::Matrix4d estimateRotation(std::vector<unsigned long long> hashes,
            std::map<unsigned long long, Eigen::Vector4d> sourcePoints, Eigen::Vector4d sourceMean, double sourceDistance,
            std::map<unsigned long long, Eigen::Vector4d> targetPoints, Eigen::Vector4d targetMean, double targetDistance,
            bool verbose = true) {

            std::stringstream ss;

            const unsigned int nPoints = hashes.size();
            Eigen::MatrixXd sourceMatrix(nPoints, 3);
            Eigen::MatrixXd targetMatrix(nPoints, 3);

            for (int i = 0; i < nPoints; i++)
            {
                sourceMatrix.block<1, 3>(i, 0) = (sourcePoints[hashes[i]] - sourceMean).block<3,1>(0,0).transpose();
                targetMatrix.block<1, 3>(i, 0) = (targetPoints[hashes[i]] - targetMean).block<3, 1>(0, 0).transpose();
            }

            Eigen::Matrix3d A = sourceMatrix.transpose() * targetMatrix;
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

            const Eigen::Matrix3d& U = svd.matrixU();
            const Eigen::Matrix3d& V = svd.matrixV();

            const float d = (U * V.transpose()).determinant();
            Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
            D(2,2) = d;

            Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
            result.block<3,3>(0,0) = U* D* V.transpose();

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