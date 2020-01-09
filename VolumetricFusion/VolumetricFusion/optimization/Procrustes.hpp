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
	class Procrustes : public vc::optimization::OptimizationProblem {

        bool vc::optimization::OptimizationProblem::specific_optimize(std::vector<ACharacteristicPoints> characteristicPoints) {
           
            return true;
        }
    };
}

#endif // !_PROCRUSTES_HEADER
