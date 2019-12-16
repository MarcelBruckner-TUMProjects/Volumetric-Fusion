#pragma once
#include "matrix_lib/utils/LibIncludeCPU.h"
#include "matrix_lib/matrix_structures/MatrixStructuresInclude.h"

namespace matrix_lib {

	namespace pose_proc {

		/**
		 * Returns a skew-symetric matrix of vector v.
		 */
		template<typename T>
		CPU_AND_GPU Mat3<T> hat(const Vec3<T>& v) {
			Mat3<T> matrix = {
				T(0),	-v.z(),	v.y(),
				v.z(),	T(0),	-v.x(),
				-v.y(),	v.x(),	T(0)
			};
			return matrix;
		}

	} // namespace pose_proc

} // namespace matrix_lib
