#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

#include <matrix_lib/matrix_structures/MatrixStructuresInclude.h>

using namespace matrix_lib;

namespace heatmap_fusion {

	template<class FloatType>
	class BoundingBox3 {
	public:

		BoundingBox3() {
			reset();
		}

		BoundingBox3(const BoundingBox3& other) : minB(other.minB), maxB(other.maxB) {}

		explicit BoundingBox3(const std::vector<Vec3<FloatType>>& verts) {
			reset();
			for (const auto &v : verts)
				include(v);
		}

		BoundingBox3(const Vec3<FloatType>& p0, const Vec3<FloatType>& p1, const Vec3<FloatType>& p2) {
			reset();
			include(p0);
			include(p1);
			include(p2);
		}

		BoundingBox3(const Vec3<FloatType>& minBound, const Vec3<FloatType>& maxBound) {
			reset();
			minB = minBound;
			maxB = maxBound;
		}

		void reset() {
			minX = minY = minZ = std::numeric_limits<FloatType>::max();
			maxX = maxY = maxZ = -std::numeric_limits<FloatType>::max();
		}

		void include(const BoundingBox3& other) {
			if (other.minX < minX)	minX = other.minX;
			if (other.minY < minY)	minY = other.minY;
			if (other.minZ < minZ)	minZ = other.minZ;

			if (other.maxX > maxX)	maxX = other.maxX;
			if (other.maxY > maxY)	maxY = other.maxY;
			if (other.maxZ > maxZ)	maxZ = other.maxZ;
		}

		void include(const Vec3<FloatType>& v) {
			if (v.x() < minX)	minX = v.x();
			if (v.y() < minY)	minY = v.y();
			if (v.z() < minZ)	minZ = v.z();

			if (v.x() > maxX)	maxX = v.x();
			if (v.y() > maxY)	maxY = v.y();
			if (v.z() > maxZ)	maxZ = v.z();
		}

		void include(const std::vector<Vec3<FloatType>>& v) {
			for (const auto &p : v)
				include(p);
		}

		FloatType getMaxExtent() const {
			FloatType d0 = maxX - minX;
			FloatType d1 = maxY - minY;
			FloatType d2 = maxZ - minZ;
			return math::max(d0, d1, d2);
		}

		FloatType getExtentX() const {
			return maxX - minX;
		}

		FloatType getExtentY() const {
			return maxY - minY;
		}

		FloatType getExtentZ() const {
			return maxZ - minZ;
		}

		Vec3<FloatType> getExtent() const {
			return Vec3<FloatType>(maxX - minX, maxY - minY, maxZ - minZ);
		}

		Vec3<FloatType> getMin() const {
			return Vec3<FloatType>(minX, minY, minZ);
		}

		Vec3<FloatType> getMax() const {
			return Vec3<FloatType>(maxX, maxY, maxZ);
		}

		Vec3<FloatType> getCenter() const {
			Vec3<FloatType> center = getMin() + getMax();
			center *= (FloatType)0.5;
			return center;
		}

		void setMin(const Vec3<FloatType>& minValue) {
			minX = minValue.x();
			minY = minValue.y();
			minZ = minValue.z();
		}

		void setMax(const Vec3<FloatType>& maxValue) {
			maxX = maxValue.x();
			maxY = maxValue.y();
			maxZ = maxValue.z();
		}

		void setMinX(FloatType v) { minX = v; }
		void setMinY(FloatType v) { minY = v; }
		void setMinZ(FloatType v) { minZ = v; }
		void setMaxX(FloatType v) { maxX = v; }
		void setMaxY(FloatType v) { maxY = v; }
		void setMaxZ(FloatType v) { maxZ = v; }

		FloatType getMinX() const { return minX; }
		FloatType getMinY() const { return minY; }
		FloatType getMinZ() const { return minZ; }
		FloatType getMaxX() const { return maxX; }
		FloatType getMaxY() const { return maxY; }
		FloatType getMaxZ() const { return maxZ; }

	protected:
		union {
			struct {
				Vec3<FloatType> minB;
				Vec3<FloatType> maxB;
			};
			struct {
				FloatType minX, minY, minZ;
				FloatType maxX, maxY, maxZ;
			};
			FloatType parameters[6];
		};
	};

	typedef BoundingBox3<float> BoundingBox3f;
	typedef BoundingBox3<double> BoundingBox3d;
	typedef BoundingBox3<int> BoundingBox3i;

} // namespace heatmap_fusion

#endif // !BOUNDINGBOX_H
