#include "Main.h"

float vertices[] = {
	 0.5f,  0.5f, 0.0f,  // top right
	 0.5f, -0.5f, 0.0f,  // bottom right
	-0.5f, -0.5f, 0.0f,  // bottom left
	-0.5f,  0.5f, 0.0f   // top left 
};
unsigned int indices[] = {  // note that we start from 0!
	0, 1, 3,   // first triangle
	1, 2, 3    // second triangle
};

const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"}\0";
const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
"}\n\0";

void mainloop(window app) {
	// Declare depth colorizer for pretty visualization of depth data
	rs2::colorizer color_map;
	// Declare rates printer for showing streaming rates of the enabled streams.
	rs2::rates_printer printer;

	// Declare RealSense pipeline, encapsulating the actual device and sensors
	rs2::pipeline pipe;

	// Start streaming with default recommended configuration
	// The default video configuration contains Depth and Color streams
	// If a device is capable to stream IMU data, both Gyro and Accelerometer are enabled by default
	pipe.start();

	glViewport(0, 0, 800, 600);

	unsigned int shaderProgram = create_shader_program(vertexShaderSource, fragmentShaderSource);

	/* 
		Vertex buffer object 
	 
		Stores raw vertex data.
	*/
	unsigned int VBO;
	glGenBuffers(1, &VBO);

	/*
		Vertex array object
	 
		Stores pointers to vertex data. Removes need to override the vertex buffers 
		whenever we change object.
	*/
	unsigned int VAO;
	glGenVertexArrays(1, &VAO);

	/*
		Element buffer object

		== Index buffer.
	*/
	unsigned int EBO;
	glGenBuffers(1, &EBO);

	// ..:: Initialization code :: ..
	// 1. bind Vertex Array Object
	glBindVertexArray(VAO);
	// 2. copy our vertices array in a vertex buffer for OpenGL to use
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	// 3. copy our index array in a element buffer for OpenGL to use
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
	// 4. then set the vertex attributes pointers
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);


	/* Loop until the user closes the window */
	while (!glfwWindowShouldClose(app))
	{
		/* Render here */
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);


		//// 5. draw the object
		//glUseProgram(shaderProgram);
		//glBindVertexArray(VAO);
		//glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		//	glBindVertexArray(0);

		rs2::frameset data = pipe.wait_for_frames().    // Wait for next set of frames from the camera
			apply_filter(printer).     // Print each enabled stream frame rate
			apply_filter(color_map);   // Find and colorize the depth data

// The show method, when applied on frameset, break it to frames and upload each frame into a gl textures
// Each texture is displayed on different viewport according to it's stream unique id
		app.show(data);

		/* Swap front and back buffers */
		glfwSwapBuffers(app);

		/* Poll for and process events */
		glfwPollEvents();
	}
}

int main()
{
	/* Initialize the library */
	if (!glfwInit())
		return -1;

	rs2::log_to_console(RS2_LOG_SEVERITY_ERROR);
	
	/* Window hints for OpenGL version settings */
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	/* Error callback */
	glfwSetErrorCallback(error_callback);

	window app(1280, 720, "RealSense Capture Example");

	if (!app)
	{
		glfwTerminate();
		return -1;
	}

	/* Set keyboard callback */
	glfwSetKeyCallback(app, key_callback);

	/* Make the window's context current */
	glfwMakeContextCurrent(app);

	/* Include the correct OpenGL version via glad. https://learnopengl.com/Getting-started/Hello-Window */
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	glfwSetFramebufferSizeCallback(app, framebuffer_size_callback);

	mainloop(app);

	glfwTerminate();
	return 0;
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);
}
