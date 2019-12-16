#ifndef UTILS_H
#define UTILS_H

#include <common_utils/Common.h>
#include <matrix_lib/matrix_structures/MatrixStructuresInclude.h>

using namespace matrix_lib;

namespace heatmap_fusion {
	namespace utils {

		inline CPU_AND_GPU float angleDifference(const Vec3f& v1, const Vec3f& v2) {
			float normalizer = v1.length() * v2.length();
			if (normalizer == 0) {
				return 0.f;
			}
			else {
				float val = ((v1 | v2) / normalizer);
				if (val >= 1.f) {
					return 0.f;
				}
				if (val <= -1.f) {
					return math_proc::PI;
				}
				return acos(val);
			}
		}

		inline CPU_AND_GPU Mat4f faceTransform(const Vec3f& vA, const Vec3f& vB) {
			auto a = vA.getNormalized();
			auto b = vB.getNormalized();
			auto axis = b ^ a;
			float angle = math_proc::radiansToDegrees(angleDifference(a, b));

			if (angle == 0.0f) {  // No rotation
				return Mat4f::identity();
			}
			else if (axis.lengthSq() == 0.0f) {  // Need any perpendicular axis
				float dotX = Vec3f::dot(Vec3f(1, 0, 0), a);
				if (std::abs(dotX) != 1.0) {
					axis = Vec3f(1, 0, 0) - dotX * a;
				}
				else {
					axis = Vec3f(0, 1, 0) - Vec3f::dot(Vec3f(0, 1, 0), a) * a;
				}
				axis.normalize();
			}

			return Mat4f::rotation(axis, -angle);
		}

	} // namespace utils
} // namespace heatmap_fusion

#endif // !UTILS_H
