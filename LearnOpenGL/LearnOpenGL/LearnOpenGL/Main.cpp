#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

const char* vertexShaderSource =
"#version 330 core" "\n"
"layout (location = 0) in vec3 aPos;""\n"
"void main()""\n"
"{""\n"
"	gl_Position = vec4(aPos, 1.0);""\n"
"}";

const char* fragmentShaderSource =
"#version 330 core" "\n"
"out vec4 FragColor;""\n"
"void main()""\n"
"{""\n"
"	FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);""\n"
"}""\n"
;

const char* fragmentShaderSource_yellow =
"#version 330 core" "\n"
"out vec4 FragColor;""\n"
"void main()""\n"
"{""\n"
"	FragColor = vec4(1.0f, 1.0f, 0.0f, 1.0f);""\n"
"}""\n"
;

int main() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(800, 600, "LearnOpenGL", NULL, NULL);
	if (window == NULL) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}
	glViewport(0, 0, 800, 600);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	float vertices[] = {
		 0.5f,  0.5f, 0.0f,  // top right
		 0.5f, -0.5f, 0.0f,  // bottom right
		-0.5f, -0.5f, 0.0f,  // bottom left
		-0.5f,  0.5f, 0.0f   // top left 
	};
	unsigned int indices[] = {  
		0, 1, 3,   
		1, 2, 3   
	};

	float vertices_ex1[] = {
		-1, 0, 0,
		-0.5f, 1.f, 0,
		0, 0, 0,

		0,0,0,
		0.5f, -1.f, 0,
		1, 0, 0
	};


	float vertices_ex2_1[] = {
		-1, 0, 0,
		-0.5f, 1.f, 0,
		0, 0, 0,
	};
	float vertices_ex2_2[] = {
		0,0,0,
		0.5f, -1.f, 0,
		1, 0, 0
	};

	unsigned int VBO_EX2_1, VBO_EX2_2, VAO_EX2_1, VAO_EX2_2;
	glGenVertexArrays(1, &VAO_EX2_1);
	glGenVertexArrays(1, &VAO_EX2_2);
	glGenBuffers(1, &VBO_EX2_1);
	glGenBuffers(1, &VBO_EX2_2);
	glBindVertexArray(VAO_EX2_1);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_EX2_1);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices_ex2_1), vertices_ex2_1, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glBindVertexArray(VAO_EX2_2);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_EX2_2);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices_ex2_2), vertices_ex2_2, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	unsigned int VBO, VAO, EBO;
	glGenBuffers(1, &VBO);
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glGenBuffers(1, &EBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	unsigned int vertexShader;
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);

	int success;
	char infoLog[512];
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	unsigned int fragmentShader;
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);

	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	unsigned int fragmentShader_yellow;
	fragmentShader_yellow = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader_yellow, 1, &fragmentShaderSource_yellow, NULL);
	glCompileShader(fragmentShader_yellow);

	glGetShaderiv(fragmentShader_yellow, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(fragmentShader_yellow, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	unsigned int shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::LINK_FAILED\n" << infoLog << std::endl;
	}

	unsigned int shaderProgram_yellow = glCreateProgram();
	glAttachShader(shaderProgram_yellow, vertexShader);
	glAttachShader(shaderProgram_yellow, fragmentShader_yellow);
	glLinkProgram(shaderProgram_yellow);

	glGetProgramiv(shaderProgram_yellow, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(shaderProgram_yellow, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::LINK_FAILED\n" << infoLog << std::endl;
	}
	
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	glDeleteShader(fragmentShader_yellow);

	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	while (!glfwWindowShouldClose(window)) {
		processInput(window);

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(shaderProgram);

		glBindVertexArray(VAO_EX2_1);
		glDrawArrays(GL_TRIANGLES, 0, 3);

		glUseProgram(shaderProgram_yellow);

		glBindVertexArray(VAO_EX2_2);
		glDrawArrays(GL_TRIANGLES, 0, 3);

		//glBindVertexArray(VAO);
		//glDrawArrays(GL_TRIANGLES, 0, 6);
		//glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}
	if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	}
	if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
}

