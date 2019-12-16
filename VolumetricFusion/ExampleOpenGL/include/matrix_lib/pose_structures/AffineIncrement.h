#pragma once
#include "PoseIncrement.h"
#include "PoseType.h"

namespace matrix_lib {

	/**
	 * Affine pose increment, where affine matrix is a 3x3 matrix and translation is 3x1 vector.
	 * This pose increment is used for optimization of affine poses.
	 */
 	template<typename T>
	class PoseIncrement<T, PoseType::AFFINE> {
	public:
		enum { Type = PoseType::AFFINE };

		/**
		 * Default constructor (identity transformation).
		 */
		CPU_AND_GPU PoseIncrement() { reset(); }
		
		/**
		 * Constructor from a parameter array.
		 * Important: The size of data needs to be at least 12 * sizeof(T).
		 */
		CPU_AND_GPU PoseIncrement(const T* data) :
			m_affineMatrix{ data },
			m_translation{ data + 9 }
		{ }

		/**
		 * Explicit constructor.
		 */
		CPU_AND_GPU PoseIncrement(const Mat4<T>& matrix) :
			m_affineMatrix{ matrix.getMatrix3x3() },
			m_translation{ matrix.getTranslation() }
		{ }

		CPU_AND_GPU PoseIncrement(const Mat3<T>& affineMatrix, const Vec3<T>& translation) :
			m_affineMatrix{ affineMatrix },
			m_translation{ translation } 
		{ }

		/**
		 * Initialize with float4.
		 */
		CPU_AND_GPU PoseIncrement(const float4& matrixXRow, const float4& matrixYRow, const float4& matrixZRow, const float4& translation) :
			m_affineMatrix{ matrixXRow, matrixYRow, matrixZRow },
			m_translation{ translation }
		{ }

		/**
		 * Copy data from other template type.
		 */
		template<typename U, int PoseType>
		friend class PoseIncrement;

		template <class U>
		CPU_AND_GPU PoseIncrement(const PoseIncrement<U, PoseType::AFFINE>& other) :
			m_affineMatrix{ other.m_affineMatrix },
			m_translation{ other.m_translation }
		{ }

		/**
		 * Interface implementation.
		 */
		CPU_AND_GPU T* getData() {
			// The data structure is packed, therefore the translation follows continuously in the memory.
			return m_affineMatrix.getData();
		}

		template <typename U>
		CPU_AND_GPU Vec3<Promote<T, U>> apply(const Vec3<U>& point) const {
			return m_affineMatrix * point + m_translation;
		}

		template <typename U>
		CPU_AND_GPU Vec3<Promote<T, U>> operator*(const Vec3<U>& point) const {
			return apply(point);
		}

		template <typename U>
		CPU_AND_GPU Vec3<Promote<T, U>> rotate(const Vec3<U>& point) const {
			return m_affineMatrix * point;
		}

		template <typename U>
		CPU_AND_GPU Vec3<Promote<T, U>> translate(const Vec3<U>& point) const {
			return point + m_translation;
		}

		CPU_AND_GPU Mat4<T> matrix() const {
			return Mat4<T>{ m_affineMatrix, m_translation };
		}

		CPU_AND_GPU void reset() { m_affineMatrix.setIdentity(); m_translation = T(0); }

		CPU_AND_GPU static PoseIncrement<T, PoseType::AFFINE> interpolate(const vector<PoseIncrement<T, PoseType::AFFINE>>& poses, const vector<T>& weights) {
			runtime_assert(poses.size() == weights.size(), "Number of poses and weights should be the same.");
			PoseIncrement<T, PoseType::AFFINE> interpolatedPose{ Mat3<T>::zero(), Vec3<T>{ 0, 0, 0 } };

			const unsigned nPoses = poses.size();
			for (auto i = 0; i < nPoses; ++i) {
				interpolatedPose.m_affineMatrix += weights[i] * poses[i].m_affineMatrix;
				interpolatedPose.m_translation += weights[i] * poses[i].m_translation;
			}

			return interpolatedPose;
		}

		CPU_AND_GPU static PoseIncrement<T, PoseType::AFFINE> identity() { return PoseIncrement<T, PoseType::AFFINE>{}; }
		CPU_AND_GPU static PoseIncrement<T, PoseType::AFFINE> translation(const Vec3<T>& translation) { return PoseIncrement<T, PoseType::AFFINE>{ Mat3<T>::identity(), translation }; }
		CPU_AND_GPU static PoseIncrement<T, PoseType::AFFINE> rotation(const Mat3<T>& rotation) { return PoseIncrement<T, PoseType::AFFINE>{ rotation, Vec3<T>{} }; }
		CPU_AND_GPU static PoseIncrement<T, PoseType::AFFINE> pose(const Mat4<T>& pose) { return PoseIncrement<T, PoseType::AFFINE>{ pose }; }
		CPU_AND_GPU static PoseIncrement<T, PoseType::AFFINE> pose(const Mat3<T>& rotation, const Vec3<T>& translation) { return PoseIncrement<T, PoseType::AFFINE>{ rotation, translation }; }

		/**
		 * Stream output.
		 */
		CPU_AND_GPU friend std::ostream& operator<<(std::ostream& os, const PoseIncrement<T, PoseType::AFFINE>& obj) {
			return os << "affineMatrix = " << obj.m_affineMatrix << ", translation = " << obj.m_translation;
		}

		/**
		 * Getters.
		 */
		CPU_AND_GPU const Mat3<T>& getAffineMatrix() const { return m_affineMatrix; }
		CPU_AND_GPU Mat3<T>& getAffineMatrix() { return m_affineMatrix; }
		CPU_AND_GPU const Vec3<T>& getTranslation() const { return m_translation; }
		CPU_AND_GPU Vec3<T>& getTranslation() { return m_translation; }

		/**
		 * Indexing operators.
		 */
		CPU_AND_GPU Mat3<T>& at(I<0>) { return m_affineMatrix; }
		CPU_AND_GPU const Mat3<T>& at(I<0>) const { return m_affineMatrix; }
		CPU_AND_GPU Vec3<T>& at(I<1>) { return m_translation; }
		CPU_AND_GPU const Vec3<T>& at(I<1>) const { return m_translation; }

		template<unsigned i>
		CPU_AND_GPU T& operator[](I<i>) {
			static_assert(i < 12, "Index out of bounds.");
			return at(I<i / 9>())[I<i % 9>()];
		}

		template<unsigned i>
		CPU_AND_GPU const T& operator[](I<i>) const {
			static_assert(i < 12, "Index out of bounds.");
			return at(I<i / 9>())[I<i % 9>()];
		}

	private:
		Mat3<T> m_affineMatrix;
		Vec3<T> m_translation;
	};

}

