#pragma once
#include "matrix_lib/utils/Promotion.h"

namespace matrix_lib {

	/**
	 * Abstract pose increment class.
	 * These methods need to be implemented by all specializations.
	 */
	template<typename T, int PType>
	class PoseIncrement {
	public:
		/**
		 * Returns the pointer to the raw data.
		 */
		T* getData();

		/**
		 * Applies the pose on the given point.
		 */
		template<typename U> Vec3<Promote<T, U>> apply(const Vec3<U>& point) const;
		template<typename U> Vec3<Promote<T, U>> operator*(const Vec3<U>& point) const;

		/**
		 * Rotates the given point.
		 */
		template<typename U> Vec3<Promote<T, U>> rotate(const Vec3<U>& point) const;

		/**
		* Translates the given point.
		*/
		template<typename U> Vec3<Promote<T, U>> translate(const Vec3<U>& point) const;

		/**
		 * Computes a 4x4 pose matrix. Needs to be generated (therefore slower).
		 */
		Mat4<T> matrix() const;

		/**
		 * Resets the pose increment to identity rotation and zero translation.
		 */
		void reset();

		/**
		 * Static constructors.
		 * The input parameters are given in matrix notation.
		 */
		static PoseIncrement<T, PType> identity();
		static PoseIncrement<T, PType> translation(const Vec3<T>& translation);
		static PoseIncrement<T, PType> rotation(const Mat3<T>& rotation);
		static PoseIncrement<T, PType> pose(const Mat4<T>& pose);
		static PoseIncrement<T, PType> pose(const Mat3<T>& rotation, const Vec3<T>& translation);

		/**
		 * Interpolates the given poses with given interpolation weights.
		 * The interpolation weights need to sum up to 1.
		 */
		CPU_AND_GPU static PoseIncrement<T, PType> interpolate(const vector<PoseIncrement<T, PType>>& poses, const vector<T>& weights);
	};

}
