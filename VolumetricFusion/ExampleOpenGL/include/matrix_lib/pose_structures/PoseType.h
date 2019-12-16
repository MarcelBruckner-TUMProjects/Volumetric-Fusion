#pragma once

namespace matrix_lib {

	/**
	 * The type of pose that is used for modelling deformations.
	 */
	struct PoseType {
		enum {
			SO3wT = 0,
			AFFINE = 1,
			QUATERNION = 2
		};
	};

} // namespace matrix_lib
