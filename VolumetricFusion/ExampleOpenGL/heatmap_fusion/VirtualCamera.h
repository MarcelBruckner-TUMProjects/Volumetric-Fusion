#ifndef VIRTUALCAMERA_H
#define VIRTUALCAMERA_H

#include <matrix_lib/matrix_structures/MatrixStructuresInclude.h>

using namespace matrix_lib;

namespace heatmap_fusion {

	/**
	 * Virtual camera containts controls to move around when visualizing a scene/shape.
	 * A right-hand vision style coordinate system is used: +x -> right, +y -> down, +z -> forward.
	 */
	template<typename FloatType>
	class VirtualCamera {
	public:
		VirtualCamera() = default;

		/**
		 * Constructor from extrinsic (world-to-camera) and intrinsic (vision projection)
		 * matrices.
		 * @param	worldToCamera		Camera pose, i.e. rigid pose from world to camera CS
		 * @param	visionProjection	Intrinsic camera matrix, given as 4x4 matrix (can also include
		 *								skew parameter, etc.)
		 * @param	width				Image/window width
		 * @param	height				Image/window height
		 * @param	zNear				Distance to the near plane
		 * @param	zFar				Distance to the far plane
		 */
		VirtualCamera(
			const Mat4<FloatType>& worldToCamera,
			const Mat4<FloatType> visionProjection,
			const int width, const int height,
			const FloatType zNear, const FloatType zFar
		) {
			Mat4<FloatType> cameraToWorld = worldToCamera.getInverse();
			
			m_eye   = Vec3<FloatType>(cameraToWorld(0, 3), cameraToWorld(1, 3), cameraToWorld(2, 3));
			m_right = Vec3<FloatType>(cameraToWorld(0, 0), cameraToWorld(1, 0), cameraToWorld(2, 0));
			m_up    = Vec3<FloatType>(cameraToWorld(0, 1), cameraToWorld(1, 1), cameraToWorld(2, 1));
			m_look  = Vec3<FloatType>(cameraToWorld(0, 2), cameraToWorld(1, 2), cameraToWorld(2, 2));

			runtime_assert(math_proc::floatEqual(m_look, (m_right ^ m_up)), "World-to-camera pose should be rigid, i.e. include rotation.");

			m_width = width;
			m_height = height;
			m_zNear = zNear;
			m_zFar = zFar;

			m_visionProjection = visionProjection;
			m_graphicsProjection = projMatrix(visionProjection, width, height, zNear, zFar);
			update();
		}

		/**
		 * Constructor with graphics notation.
		 * Important: all vectors should be given in vision coordinate system.
		 * @param	eye				Eye (position) of the camera
		 * @param	lookDir			Camera look direction
		 * @param	cameraUp		Camera up vector
		 * @param	fieldOfView		Horizontal field of view, i.e. maximum viewing angle (in degrees)
		 * @param	width			Image/window width
		 * @param	height			Image/window height
		 * @param	zNear			Distance to the near plane
		 * @param	zFar			Distance to the far plane
		 */
		VirtualCamera(
			const Vec3<FloatType>& eye, 
			const Vec3<FloatType>& lookDir, 
			const Vec3<FloatType>& cameraUp,
			const FloatType fieldOfView,
			const int width, const int height,
			const FloatType zNear, const FloatType zFar
		) {
			m_eye = eye;
			m_look = lookDir.getNormalized();
			m_up = cameraUp.getNormalized();
			m_right = (m_up ^ m_look).getNormalized();

			runtime_assert(math_proc::floatEqual(m_look, (m_right ^ m_up)), "Orientation of camera vectors is wrong.");

			m_width = width;
			m_height = height;
			m_zNear = zNear;
			m_zFar = zFar;

			// Conversion from graphics into vision notation.
			FloatType aspect = FloatType(width) / height;
			FloatType t = tan(FloatType(0.5) * math_proc::degreesToRadians(fieldOfView));
			FloatType focalLengthX = FloatType(0.5) * FloatType(width) / t;
			FloatType focalLengthY = FloatType(0.5) * FloatType(height) / t * aspect;

			m_visionProjection = Mat4<FloatType>(
				focalLengthX, 0.0, FloatType(width - 1) / 2.0, 0.0,
				0.0, focalLengthY, FloatType(height - 1) / 2.0, 0.0,
				0.0, 0.0, 1.0, 0.0,
				0.0, 0.0, 0.0, 1.0
			);

			m_graphicsProjection = projMatrix(m_visionProjection, width, height, zNear, zFar);

			update();
		}

		/**
		 * Returns the world-to-camera matrix.
		 */
		Mat4<FloatType> getView() const {
			return m_view;
		}

		/**
		 * Returns the projection matrix (both for vision and graphics).
		 */
		Mat4<FloatType> getGraphicsProj() const {
			return m_graphicsProjection;
		}

		Mat4<FloatType> getVisionProj() const {
			return m_visionProjection;
		}

		/**
		 * Returns the the camera-projection matrix (world -> camera -> proj space).
		 * The projection is given in graphics notation (for DirectX).
		 */
		Mat4<FloatType> getViewProj() const {
			return m_viewProjection;
		}

		/**
		 * Returns the eye point.
		 */
		Vec3<FloatType> getEye() const {
			return m_eye;
		}

		/** 
		 * Returns the look direction.
		 */
		Vec3<FloatType> getLook() const {
			return m_look;
		}

		/**
		 * Returns the right direction.
		 */
		Vec3<FloatType> getRight() const {
			return m_right;
		}

		/**
		 * Returns the (camera) up direction.
		 */
		Vec3<FloatType> getUp() const {
			return m_up;
		}

		/**
		 * Returns the distance to the near plane.
		 */
		float getNearPlane() const {
			return m_zNear;
		}

		/**
		 * Returns the distance to the far plane.
		 */
		float getFarPlane() const {
			return m_zFar;
		}

		/**
		 * Constructs a screen ray; screen coordinates are in [0; 1].
		 */
		Vec3<FloatType> getScreenRay(FloatType screenX, FloatType screenY) const {
			Vec3<FloatType> perspectivePoint{
				math::linearMap((FloatType)0.0, (FloatType)1.0, (FloatType)-1.0, (FloatType)1.0, screenX),
				math::linearMap((FloatType)0.0, (FloatType)1.0, (FloatType)1.0, (FloatType)-1.0, screenY),
				(FloatType)-0.5
			};

			return getViewProj().getInverse() * perspectivePoint - m_eye;
		}

		/**
		 * Camera movement commands.
		 * Angle 'theta' should always be specified in degrees.
		 */
		template <class FloatType>
		void lookRight(FloatType theta) {
			applyTransform(Mat3<FloatType>::rotation(m_up, theta));
		}

		template <class FloatType>
		void lookUp(FloatType theta) {
			applyTransform(Mat3<FloatType>::rotation(m_right, -theta));
		}

		template <class FloatType>
		void roll(FloatType theta) {
			applyTransform(Mat3<FloatType>::rotation(m_look, theta));
		}

		template <class FloatType>
		void applyTransform(const Mat3<FloatType>& transform) {
			m_up = transform * m_up;
			m_right = transform * m_right;
			m_look = transform * m_look;
			update();
		}
		template <class FloatType>
		void applyTransform(const Mat4<FloatType>& transform) {
			const Mat3<FloatType> rot = transform.getRotation();
			m_up = rot * m_up;
			m_right = rot * m_right;
			m_look = rot * m_look;
			m_eye += transform.getTranslation();
			update();
		}

		template <class FloatType>
		void strafe(FloatType delta) {
			m_eye += m_right * delta;
			update();
		}

		template <class FloatType>
		void jump(FloatType delta) {
			m_eye += m_up * delta;
			update();
		}

		template <class FloatType>
		void move(FloatType delta) {
			m_eye += m_look * delta;
			update();
		}

		template <class FloatType>
		void translate(const Vec3<FloatType> &v) {
			m_eye += v;
			update();
		}

		/**
		 * Resets the view matrix given the camera view vectors.
		 */
		void reset(const Vec3<FloatType>& eye, const Vec3<FloatType>& lookDir, const Vec3<FloatType>& cameraUp) {
			m_eye   = eye;
			m_up    = cameraUp.getNormalized();
			m_look  = lookDir.getNormalized();
			m_right = (m_up ^ m_look).getNormalized();

			runtime_assert(math_proc::floatEqual(m_look, (m_right ^ m_up)), "Orientation of camera vectors is wrong.");

			update();
		}

	private:
		Vec3<FloatType> m_eye, m_right, m_look, m_up;
		Mat4<FloatType> m_graphicsProjection;
		Mat4<FloatType> m_visionProjection;
		Mat4<FloatType> m_view;
		Mat4<FloatType> m_viewProjection;

		FloatType m_zNear{ 0.f }, m_zFar{ 0.f };
		int m_width{ 0 }, m_height{ 0 };
		
		/**
		 * Constructs a graphics projection matrix.
		 * The OpenGL normalized device coordinates are assumed: [-1, 1] x [-1, 1] x [-1, 1].
		 */
		static Mat4<FloatType> projMatrix(const Mat4<FloatType>& visionProjection, unsigned width, unsigned height, FloatType zNear, FloatType zFar) {
			Mat4<FloatType> pixelToNDC(
				FloatType(2) / width, 0, FloatType(1) / width - 1, 0,
				0, -FloatType(2) / height, 1 - FloatType(1) / height, 0,
				0, 0, 2 * zFar / (zFar - zNear) - 1, 2 * zFar * zNear / (zNear - zFar),
				0, 0, 1, 0
			);

			return pixelToNDC * visionProjection;
		}

		/**
		 * Constructs a view matrix (world -> camera).
		 */
		static Mat4<FloatType> viewMatrix(const Vec3<FloatType>& eye, const Vec3<FloatType>& lookDir, const Vec3<FloatType>& up, const Vec3<FloatType>& right) {
			Vec3<FloatType> l = lookDir.getNormalized();
			Vec3<FloatType> u = up.getNormalized();
			Vec3<FloatType> r = right.getNormalized();

			return Mat4<FloatType>(
				r.x(), r.y(), r.z(), -Vec3<FloatType>::dot(r, eye),
				u.x(), u.y(), u.z(), -Vec3<FloatType>::dot(u, eye),
				l.x(), l.y(), l.z(), -Vec3<FloatType>::dot(l, eye),
				0, 0, 0, 1
			);
		}

		/**
		 * Updates the view and view-projection matrices.
		 * Needs to be called every time we change the view or projection parameters.
		 */
		void update() {
			m_view = viewMatrix(m_eye, m_look, m_up, m_right);
			m_viewProjection = m_graphicsProjection * m_view;
		}
	};

	using VirtualCameraf = VirtualCamera<float>;
	using VirtualCamerad = VirtualCamera<double>;

} // namespace heatmap_fusion

#endif // !VIRTUALCAMERA_H