#pragma once
#include "matrix_lib/utils/LibIncludeCPU.h"
#include "matrix_lib/utils/Promotion.h"
#include "matrix_lib/matrix_structures/MatrixStructuresInclude.h"

namespace matrix_lib {

	/**
	 * Abstract pose class.
	 * These methods need to be implemented by all specializations.
	 */
	template<typename T, int PType>
	class Pose {
	public:
		/**
		 * Applies the pose on the given point.
		 */
		template<typename U> CPU_AND_GPU Vec3<Promote<T, U>> apply(const Vec3<U>& point) const;
		template<typename U> CPU_AND_GPU Vec3<Promote<T, U>> operator*(const Vec3<U>& point) const;

		/**
		 * Rotates the given point.
		 */
		template<typename U> CPU_AND_GPU Vec3<Promote<T, U>> rotate(const Vec3<U>& point) const;

		/**
		 * Translates the given point.
		 */
		template<typename U> CPU_AND_GPU Vec3<Promote<T, U>> translate(const Vec3<U>& point) const;

		/**
		 * Computes a 4x4 pose matrix. Needs to be generated (therefore slower).
		 */
		CPU_AND_GPU Mat4<T> matrix() const;

		/**
		 * Updates the current pose to match the given pose.
		 */
		CPU_AND_GPU void update(const Pose<T, PType>& pose);

		/**
		 * Static constructors. 
		 * The input parameters are given in matrix notation.
		 */
		CPU_AND_GPU static Pose<T, PType> identity();
		CPU_AND_GPU static Pose<T, PType> translation(const Vec3<T>& translation);
		CPU_AND_GPU static Pose<T, PType> rotation(const Mat3<T>& rotation);
		CPU_AND_GPU static Pose<T, PType> pose(const Mat4<T>& pose);
		CPU_AND_GPU static Pose<T, PType> pose(const Mat3<T>& rotation, const Vec3<T>& translation);

		/**
		 * Interpolates the given poses with given interpolation weights.
		 * The interpolation weights need to sum up to 1.
		 */
		CPU_AND_GPU static Pose<T, PType> interpolate(const vector<Pose<T, PType>>& poses, const vector<T>& weights);
	};

} // namespace matrix_lib
