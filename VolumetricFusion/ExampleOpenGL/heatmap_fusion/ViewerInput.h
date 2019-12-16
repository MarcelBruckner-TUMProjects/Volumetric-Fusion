#ifndef VIEWERINPUT_H
#define VIEWERINPUT_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <set>
#include <matrix_lib/matrix_structures/MatrixStructuresInclude.h>

using namespace matrix_lib;

namespace heatmap_fusion {

	struct MouseState {
		Vec2f pos;

		bool leftPressed{ false };
		bool rightPressed{ false };
	};


	class ViewerInput {
	public:
		static ViewerInput& get() // Singleton is accessed via get()
		{
			static ViewerInput instance; // lazy singleton, instantiated on first use
			return instance;
		}

		// Callback methods.
		static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
			get().mouseButtonCallbackImpl(button, action);
		}

		void mouseButtonCallbackImpl(int button, int action) {
			if (button == GLFW_MOUSE_BUTTON_1) {
				if (action == GLFW_PRESS) {
					m_bMouseLeftDown = true;
					m_mouse.leftPressed = true;
				}
				else if (action == GLFW_RELEASE) {
					m_bMouseLeftUp = true;
					m_mouse.leftPressed = false;
				}
			}
			if (button == GLFW_MOUSE_BUTTON_1) {
				if (action == GLFW_PRESS) {
					m_bMouseLeftDown = true;
					m_mouse.rightPressed = true;
				}
				else if (action == GLFW_RELEASE) {
					m_bMouseLeftUp = true;
					m_mouse.rightPressed = false;
				}
			}
		}

		static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
			if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
				glfwSetWindowShouldClose(window, GLFW_TRUE);

			//here we access the instance via the singleton pattern and forward the callback to the instance method
			get().keyCallbackImpl(key, action);
		}

		void keyCallbackImpl(int key, int action) {
			if (action == GLFW_PRESS) {
				m_keysPressed[key] = true;
				m_keysDown.insert(key);
			}
			else if (action == GLFW_RELEASE) {
				m_keysPressed[key] = false;
				m_keysUp.insert(key);
			}
		}

		static void scrollCallback(GLFWwindow* window, double xOffset, double yOffset) {
			get().scrollCallbackImpl(xOffset, yOffset);
		}

		void scrollCallbackImpl(double xOffset, double yOffset) {
			m_scrollChangeX = xOffset;
			m_scrollChangeY = yOffset;
			m_bScrollChange = true;
		}

		static void cursorPosCallback(GLFWwindow* window, double x, double y) {
			get().cursorPosCallbackImpl(x, y);
		}

		void cursorPosCallbackImpl(double x, double y) {
			m_bMouseMoved = true;
			m_mouse.pos = Vec2f(x, y);
		}

		static void errorCallback(int error, const char* description) {
			fprintf(stderr, "Error: %s\n", description);
		}

		// Resets change events.
		void resetChangeEvents() {
			m_keysUp.clear();
			m_keysDown.clear();

			m_bMouseMoved = false;
			m_bMouseLeftUp = false;
			m_bMouseLeftDown = false;
			m_bMouseRightUp = false;
			m_bMouseRightDown = false;
			m_bScrollChange = false;

			m_mousePrev = m_mouse;
			m_scrollChangeX = 0.0;
			m_scrollChangeY = 0.0;
		}

		// Event querries.
		bool isKeyDown(int key) const {
			return m_keysDown.find(key) != m_keysDown.end();
		}

		bool isKeyUp(int key) const {
			return m_keysUp.find(key) != m_keysUp.end();
		}

		bool isKeyPressed(int key) const {
			return m_keysPressed.find(key) != m_keysPressed.end() && m_keysPressed.find(key)->second;
		}

		bool isMouseLeftDown() const {
			return m_bMouseLeftDown;
		}

		bool isMouseLeftUp() const {
			return m_bMouseLeftUp;
		}

		bool isMouseMoved() const {
			return m_bMouseMoved;
		}

		bool isScrollChanged() const {
			return m_bScrollChange;
		}

		const MouseState& getMouse() const {
			return m_mouse;
		}

		const MouseState& getMousePrev() const {
			return m_mousePrev;
		}

		double getScrollChangeX() const {
			return m_scrollChangeX;
		}

		double getScrollChangeY() const {
			return m_scrollChangeY;
		}

	private:
		std::set<int> m_keysDown;
		std::set<int> m_keysUp;
		std::map<int, bool> m_keysPressed;

		MouseState m_mousePrev;
		MouseState m_mouse;

		double m_scrollChangeX{ 0.0 };
		double m_scrollChangeY{ 0.0 };

		bool m_bMouseMoved{ false };
		bool m_bMouseLeftUp{ false };
		bool m_bMouseLeftDown{ false };
		bool m_bMouseRightUp{ false };
		bool m_bMouseRightDown{ false };
		bool m_bScrollChange{ false };

		ViewerInput(void) // private constructor necessary to allow only 1 instance
		{ }

		ViewerInput(ViewerInput const&); // prevent copies
		void operator=(ViewerInput const&); // prevent assignments
	};

} // namespace heatmap_fusion

#endif // !VIEWERINPUT_H
