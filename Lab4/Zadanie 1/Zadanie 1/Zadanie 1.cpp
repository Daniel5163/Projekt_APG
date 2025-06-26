#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

struct ShaderParams {
	float ambientStrength = 0.3f;
	float specularStrength = 0.5f;
	int lightColorIndex = 0;
	float shadowBias = 0.005f;
	float shadowDarkness = 0.5f;
	bool shadowsEnabled = true;
	float cubeAlpha = 0.7f;
	bool showLightSource = true;

	std::vector<glm::vec3> lightColors = {
	{1.0f, 0.9f, 0.7f},
	{1.0f, 0.5f, 0.5f},
	{0.5f, 1.0f, 0.5f},
	{0.5f, 0.5f, 1.0f},
	{1.0f, 1.0f, 0.5f},
	{1.0f, 0.5f, 1.0f}
	};
};

ShaderParams shaderParams;

class Shader {
public:
	GLuint ID;

	Shader(const char* vertexSource, const char* fragmentSource) {
		GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertexShader, 1, &vertexSource, NULL);
		glCompileShader(vertexShader);
		checkCompileErrors(vertexShader, "VERTEX");

		GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
		glCompileShader(fragmentShader);
		checkCompileErrors(fragmentShader, "FRAGMENT");

		ID = glCreateProgram();
		glAttachShader(ID, vertexShader);
		glAttachShader(ID, fragmentShader);
		glLinkProgram(ID);
		checkCompileErrors(ID, "PROGRAM");

		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);
	}

	void use() { glUseProgram(ID); }

	void setMat4(const std::string& name, const glm::mat4& mat) const {
		glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
	}

	void setVec3(const std::string& name, const glm::vec3& value) const {
		glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
	}

	void setInt(const std::string& name, int value) const {
		glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
	}

	void setFloat(const std::string& name, float value) const {
		glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
	}

	void setAlpha(float alpha) const {
		glUniform1f(glGetUniformLocation(ID, "alpha"), alpha);
	}

private:
	void checkCompileErrors(GLuint shader, std::string type) {
		GLint success;
		GLchar infoLog[1024];
		if (type != "PROGRAM") {
			glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
			if (!success) {
				glGetShaderInfoLog(shader, 1024, NULL, infoLog);
				std::cout << "ERROR::SHADER_COMPILATION_ERROR: " << type << "\n" << infoLog << "\n";
			}
		}
		else {
			glGetProgramiv(shader, GL_LINK_STATUS, &success);
			if (!success) {
				glGetProgramInfoLog(shader, 1024, NULL, infoLog);
				std::cout << "ERROR::PROGRAM_LINKING_ERROR: " << type << "\n" << infoLog << "\n";
			}
		}
	}
};

class Camera {
public:
	glm::vec3 Position;
	glm::vec3 Front;
	glm::vec3 Up;
	float Yaw;
	float Pitch;
	float Speed;

	Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 5.0f))
		: Front(glm::vec3(0.0f, 0.0f, -1.0f)), Up(glm::vec3(0.0f, 1.0f, 0.0f)),
		Yaw(-90.0f), Pitch(0.0f), Speed(2.5f), Position(position) {
	}

	glm::mat4 GetViewMatrix() {
		return glm::lookAt(Position, Position + Front, Up);
	}

	void ProcessKeyboard(int direction, float deltaTime) {
		float velocity = Speed * deltaTime;
		if (direction == 0) Position += Front * velocity;
		if (direction == 1) Position -= Front * velocity;
		if (direction == 2) Position -= glm::normalize(glm::cross(Front, Up)) * velocity;
		if (direction == 3) Position += glm::normalize(glm::cross(Front, Up)) * velocity;
		if (direction == 4) Position += Up * velocity;
		if (direction == 5) Position -= Up * velocity;
	}

	void ProcessMouseMovement(float xoffset, float yoffset) {
		float sensitivity = 0.1f;
		xoffset *= sensitivity;
		yoffset *= sensitivity;

		Yaw += xoffset;
		Pitch += yoffset;

		if (Pitch > 89.0f) Pitch = 89.0f;
		if (Pitch < -89.0f) Pitch = -89.0f;

		updateCameraVectors();
	}

private:
	void updateCameraVectors() {
		glm::vec3 front;
		front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
		front.y = sin(glm::radians(Pitch));
		front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
		Front = glm::normalize(front);
	}
};

class Mesh {
public:
	GLuint VAO, VBO, EBO;
	std::vector<float> vertices;
	std::vector<unsigned int> indices;

	Mesh(std::vector<float> verts, std::vector<unsigned int> inds) : vertices(verts), indices(inds) {
		setupMesh();
	}

	void Draw() {
		glBindVertexArray(VAO);
		glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
	}

protected:
	void setupMesh() {
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glGenBuffers(1, &EBO);

		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));

		glBindVertexArray(0);
	}
};

class Sphere : public Mesh {
public:
	Sphere(float radius = 1.0f, int sectors = 36, int stacks = 18)
		: Mesh(generateVertices(radius, sectors, stacks),
			generateIndices(sectors, stacks)) {
	}

private:
	static std::vector<float> generateVertices(float radius, int sectors, int stacks) {
		std::vector<float> sphere_vertices;
		const float PI = 3.14159265358979323846f;
		const float lengthInv = 1.0f / radius;
		float sectorStep = 2 * PI / sectors;
		float stackStep = PI / stacks;
		float sectorAngle, stackAngle;

		for (int i = 0; i <= stacks; ++i) {
			stackAngle = PI / 2 - i * stackStep;
			float xy = radius * cosf(stackAngle);
			float z_coord = radius * sinf(stackAngle);

			for (int j = 0; j <= sectors; ++j) {
				sectorAngle = j * sectorStep;
				float x_coord = xy * cosf(sectorAngle);
				float y_coord = xy * sinf(sectorAngle);
				sphere_vertices.insert(sphere_vertices.end(), { x_coord, y_coord, z_coord, x_coord * lengthInv, y_coord * lengthInv, z_coord * lengthInv });
			}
		}
		return sphere_vertices;
	}

	static std::vector<unsigned int> generateIndices(int sectors, int stacks) {
		std::vector<unsigned int> sphere_indices;
		unsigned int k1, k2;
		for (int i = 0; i < stacks; ++i) {
			k1 = i * (sectors + 1);
			k2 = k1 + sectors + 1;
			for (int j = 0; j < sectors; ++j, ++k1, ++k2) {
				if (i != 0) {
					sphere_indices.insert(sphere_indices.end(), { k1, k2, k1 + 1 });
				}
				if (i != (stacks - 1)) {
					sphere_indices.insert(sphere_indices.end(), { k1 + 1, k2, k2 + 1 });
				}
			}
		}
		return sphere_indices;
	}
};

class ShadowMap {
public:
	GLuint FBO, texture;
	const unsigned int width = 1024, height = 1024;

	ShadowMap() {
		glGenFramebuffers(1, &FBO);
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
		glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

		glBindFramebuffer(GL_FRAMEBUFFER, FBO);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture, 0);
		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void BindForWriting() {
		glViewport(0, 0, width, height);
		glBindFramebuffer(GL_FRAMEBUFFER, FBO);
		glClear(GL_DEPTH_BUFFER_BIT);
	}

	void BindForReading(GLenum textureUnit) {
		glActiveTexture(textureUnit);
		glBindTexture(GL_TEXTURE_2D, texture);
	}
};

Camera camera;
float lastX = 400.0f, lastY = 300.0f;
bool firstMouse = true;
float deltaTime = 0.0f;
float lastFrame = 0.0f;

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
	if (firstMouse) {
		lastX = static_cast<float>(xpos);
		lastY = static_cast<float>(ypos);
		firstMouse = false;
	}

	float xoffset = static_cast<float>(xpos) - lastX;
	float yoffset = lastY - static_cast<float>(ypos);
	lastX = static_cast<float>(xpos);
	lastY = static_cast<float>(ypos);

	camera.ProcessMouseMovement(xoffset, yoffset);
}

void processInput(GLFWwindow* window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camera.ProcessKeyboard(0, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camera.ProcessKeyboard(1, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camera.ProcessKeyboard(2, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camera.ProcessKeyboard(3, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) camera.ProcessKeyboard(4, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) camera.ProcessKeyboard(5, deltaTime);

	static bool keysProcessed[6] = { false };
	static bool shadowKeysProcessed[5] = { false };
	static bool alphaKeysProcessed[2] = { false };
	static bool lightSourceKeyProcessed = false;

	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS && !keysProcessed[0]) {
		shaderParams.ambientStrength = std::max(0.0f, shaderParams.ambientStrength - 0.05f);
		//std::cout << "Sila ambientu: " << shaderParams.ambientStrength << std::endl;
		keysProcessed[0] = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_RELEASE) keysProcessed[0] = false;

	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS && !keysProcessed[1]) {
		shaderParams.ambientStrength = std::min(1.0f, shaderParams.ambientStrength + 0.05f);
		//std::cout << "Sila ambientu: " << shaderParams.ambientStrength << std::endl;
		keysProcessed[1] = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_E) == GLFW_RELEASE) keysProcessed[1] = false;

	if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS && !keysProcessed[2]) {
		shaderParams.specularStrength = std::max(0.0f, shaderParams.specularStrength - 0.05f);
		//std::cout << "rozblyski: " << shaderParams.specularStrength << std::endl;
		keysProcessed[2] = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_RELEASE) keysProcessed[2] = false;

	if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS && !keysProcessed[3]) {
		shaderParams.specularStrength = std::min(2.0f, shaderParams.specularStrength + 0.05f);
		//std::cout << "rozblyski: " << shaderParams.specularStrength << std::endl;
		keysProcessed[3] = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_C) == GLFW_RELEASE) keysProcessed[3] = false;

	if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS && !keysProcessed[4]) {
		shaderParams.lightColorIndex = (shaderParams.lightColorIndex + 1) % shaderParams.lightColors.size();
		//std::cout << "kolor numer: " << shaderParams.lightColorIndex << std::endl;
		keysProcessed[4] = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_L) == GLFW_RELEASE) keysProcessed[4] = false;

	if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS && !shadowKeysProcessed[0]) {
		shaderParams.shadowsEnabled = !shaderParams.shadowsEnabled;
		//std::cout << "cienie: " << (shaderParams.shadowsEnabled ? "ON" : "OFF") << std::endl;
		shadowKeysProcessed[0] = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_B) == GLFW_RELEASE) shadowKeysProcessed[0] = false;

	if (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS && !shadowKeysProcessed[1]) {
		shaderParams.shadowBias += 0.001f;
		//std::cout << "bias cieni: " << shaderParams.shadowBias << std::endl;
		shadowKeysProcessed[1] = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_N) == GLFW_RELEASE) shadowKeysProcessed[1] = false;

	if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS && !shadowKeysProcessed[2]) {
		shaderParams.shadowBias = std::max(0.0f, shaderParams.shadowBias - 0.001f);
		//std::cout << "bias cieni: " << shaderParams.shadowBias << std::endl;
		shadowKeysProcessed[2] = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_M) == GLFW_RELEASE) shadowKeysProcessed[2] = false;

	if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS && !shadowKeysProcessed[3]) {
		shaderParams.shadowDarkness = std::min(1.0f, shaderParams.shadowDarkness + 0.05f);
		//std::cout << "sila cienia: " << shaderParams.shadowDarkness << std::endl;
		shadowKeysProcessed[3] = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_K) == GLFW_RELEASE) shadowKeysProcessed[3] = false;

	if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS && !shadowKeysProcessed[4]) {
		shaderParams.shadowDarkness = std::max(0.0f, shaderParams.shadowDarkness - 0.05f);
		//std::cout << "sila cienia: " << shaderParams.shadowDarkness << std::endl;
		shadowKeysProcessed[4] = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_J) == GLFW_RELEASE) shadowKeysProcessed[4] = false;

	if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS && !alphaKeysProcessed[0]) {
		shaderParams.cubeAlpha = std::max(0.0f, shaderParams.cubeAlpha - 0.1f);
		//std::cout << "przezroczystosc kostki: " << shaderParams.cubeAlpha << std::endl;
		alphaKeysProcessed[0] = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_U) == GLFW_RELEASE) alphaKeysProcessed[0] = false;

	if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS && !alphaKeysProcessed[1]) {
		shaderParams.cubeAlpha = std::min(1.0f, shaderParams.cubeAlpha + 0.1f);
		//std::cout << "przezroczystosc kostki: " << shaderParams.cubeAlpha << std::endl;
		alphaKeysProcessed[1] = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_I) == GLFW_RELEASE) alphaKeysProcessed[1] = false;

	if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS && !lightSourceKeyProcessed) {
		shaderParams.showLightSource = !shaderParams.showLightSource;
		//std::cout << "czy slonce jest widoczne: " << (shaderParams.showLightSource ? "ON" : "OFF") << std::endl;
		lightSourceKeyProcessed = true;
	}
	else if (glfwGetKey(window, GLFW_KEY_O) == GLFW_RELEASE) lightSourceKeyProcessed = false;
}

const char* vertexShaderSource = R"glsl(
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

out vec3 FragPos;
out vec3 Normal;
out vec4 FragPosLightSpace;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 lightSpaceMatrix;

void main() {
FragPos = vec3(model * vec4(position, 1.0));
Normal = mat3(transpose(inverse(model))) * normal;
FragPosLightSpace = lightSpaceMatrix * vec4(FragPos, 1.0);
gl_Position = projection * view * vec4(FragPos, 1.0);
}
)glsl";

const char* fragmentShaderSource = R"glsl(
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec4 FragPosLightSpace;

out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform sampler2D shadowMap;
uniform float ambientStrength;
uniform float specularStrength;
uniform float shadowBias;
uniform float shadowDarkness;
uniform bool shadowsEnabled;
uniform float alpha;

float ShadowCalculation(vec4 fragPosLightSpace) {
if (!shadowsEnabled) return 0.0;

vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;

projCoords = projCoords * 0.5 + 0.5;

if (projCoords.z > 1.0 || projCoords.x < 0.0 || projCoords.x > 1.0 ||
projCoords.y < 0.0 || projCoords.y > 1.0) {
return 0.0;
}

float closestDepth = texture(shadowMap, projCoords.xy).r;

float currentDepth = projCoords.z;

float shadow = 0.0;
vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
for(int x = -1; x <= 1; ++x) {
for(int y = -1; y <= 1; ++y) {
float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;

shadow += currentDepth - shadowBias > pcfDepth ? 1.0 : 0.0;
}
}
shadow /= 9.0;

return shadow * shadowDarkness;

}

void main() {
vec3 ambient = ambientStrength * lightColor;

vec3 norm = normalize(Normal);
vec3 lightDir = normalize(lightPos - FragPos);

float diff = max(dot(norm, lightDir), 0.0);
vec3 diffuse = diff * lightColor;

vec3 viewDir = normalize(viewPos - FragPos);
vec3 reflectDir = reflect(-lightDir, norm);
float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
vec3 specular = specularStrength * spec * lightColor;

float shadow = ShadowCalculation(FragPosLightSpace);

vec3 objectColor = vec3(0.7, 0.2, 0.2);

vec3 lighting = ambient + (1.0 - shadow) * (diffuse + specular);
vec3 result = lighting * objectColor;

FragColor = vec4(result, alpha);
}
)glsl";

const char* shadowVertexShaderSource = R"glsl(
#version 330 core
layout(location = 0) in vec3 position;

uniform mat4 lightSpaceMatrix;
uniform mat4 model;

void main() {
gl_Position = lightSpaceMatrix * model * vec4(position, 1.0);
}
)glsl";

const char* shadowFragmentShaderSource = R"glsl(
#version 330 core

void main() {

}
)glsl";

const char* lightSphereVertexShaderSource = R"glsl(
#version 330 core
layout(location = 0) in vec3 position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
gl_Position = projection * view * model * vec4(position, 1.0);
}
)glsl";

const char* lightSphereFragmentShaderSource = R"glsl(
#version 330 core
out vec4 FragColor;
uniform vec3 lightColor;

void main() {

FragColor = vec4(lightColor, 1.0);
}
)glsl";

int main() {
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW" << std::endl;
		return -1;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	const int INITIAL_WINDOW_WIDTH = 800;
	const int INITIAL_WINDOW_HEIGHT = 600;
	GLFWwindow* window = glfwCreateWindow(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT, "Projekt Ciosmak Arazny ", NULL, NULL);
	if (!window) {
		std::cerr << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPosCallback(window, mouse_callback);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		std::cerr << "Failed to initialize GLEW" << std::endl;
		glfwTerminate();
		return -1;
	}

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	Shader mainShader(vertexShaderSource, fragmentShaderSource);
	Shader shadowShader(shadowVertexShaderSource, shadowFragmentShaderSource);
	Shader lightSphereShader(lightSphereVertexShaderSource, lightSphereFragmentShaderSource);

	ShadowMap shadowMap;

	std::vector<float> cubeVertices = {
	-0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f,
	0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f, -0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f, -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f,
	-0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f,
	0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f, -0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f, -0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f,
	-0.5f, 0.5f, 0.5f, -1.0f, 0.0f, 0.0f, -0.5f, 0.5f, -0.5f, -1.0f, 0.0f, 0.0f, -0.5f, -0.5f, -0.5f, -1.0f, 0.0f, 0.0f,
	-0.5f, -0.5f, -0.5f, -1.0f, 0.0f, 0.0f, -0.5f, -0.5f, 0.5f, -1.0f, 0.0f, 0.0f, -0.5f, 0.5f, 0.5f, -1.0f, 0.0f, 0.0f,
	0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f,
	0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 0.5f, -0.5f, 0.5f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f,
	-0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f, 0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f, 0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f,
	0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f, -0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f, -0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f,
	-0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f,
	0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f, -0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f, -0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f
	};
	std::vector<unsigned int> cubeIndices(36);
	for (unsigned int i = 0; i < 36; i++) cubeIndices[i] = i;
	Mesh cube(cubeVertices, cubeIndices);

	std::vector<float> floorVertices = {
	-5.0f, -1.0f, -5.0f, 0.0f, 1.0f, 0.0f,
	-5.0f, -1.0f, 5.0f, 0.0f, 1.0f, 0.0f,
	5.0f, -1.0f, 5.0f, 0.0f, 1.0f, 0.0f,
	5.0f, -1.0f, -5.0f, 0.0f, 1.0f, 0.0f
	};
	std::vector<unsigned int> floorIndices = { 0, 1, 2, 2, 3, 0 };
	Mesh floorMesh(floorVertices, floorIndices);

	Sphere lightSphere(0.2f);

	std::cout << "sterowanie:\n"
		<< "WASD: poruszanie sie\n"
		<< "SPACE/LSHIFT: gora dół\n"
		<< "Mouse: rozejrzyj sie\n"
		<< "Q/E: sila ambientu +/- \n"
		<< "Z/C: rozblyski +/- \n"
		<< "L: zmien kolor\n"
		<< "U/I: przezroczystosc +/- \n"
		<< "O: czy widac kulke ze sloncem\n"
		<< "B: cien wlacz/wylacz pod kostka\n"
		<< "N/M: bias +/- \n"
		<< "K/J: jak ciemny jest cien +/- \n"
		<< "ESC: wyjdz\n";

	while (!glfwWindowShouldClose(window)) {
		float currentFrame = static_cast<float>(glfwGetTime());
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		processInput(window);

		glm::vec3 lightPos(2.0f * sin(currentFrame), 1.5f, 2.0f * cos(currentFrame));
		glm::vec3 currentLightColor = shaderParams.lightColors[shaderParams.lightColorIndex];

		shadowMap.BindForWriting();

		float near_plane_shadow = 1.0f, far_plane_shadow = 20.0f;
		glm::mat4 lightProjection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, near_plane_shadow, far_plane_shadow);
		glm::mat4 lightView = glm::lookAt(lightPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 lightSpaceMatrix = lightProjection * lightView;

		shadowShader.use();
		shadowShader.setMat4("lightSpaceMatrix", lightSpaceMatrix);

		glm::mat4 model = glm::mat4(1.0f);
		model = glm::rotate(model, currentFrame * 0.5f, glm::vec3(0.5f, 1.0f, 0.0f));
		shadowShader.setMat4("model", model);
		cube.Draw();

		model = glm::mat4(1.0f);
		shadowShader.setMat4("model", model);
		floorMesh.Draw();

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		int screenWidth, screenHeight;
		glfwGetFramebufferSize(window, &screenWidth, &screenHeight);
		glViewport(0, 0, screenWidth, screenHeight);
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		mainShader.use();

		glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)screenWidth / (float)screenHeight, 0.1f, 100.0f);
		glm::mat4 view = camera.GetViewMatrix();

		mainShader.setMat4("projection", projection);
		mainShader.setMat4("view", view);
		mainShader.setMat4("lightSpaceMatrix", lightSpaceMatrix);
		mainShader.setVec3("lightPos", lightPos);
		mainShader.setVec3("viewPos", camera.Position);
		mainShader.setVec3("lightColor", currentLightColor);
		mainShader.setFloat("ambientStrength", shaderParams.ambientStrength);
		mainShader.setFloat("specularStrength", shaderParams.specularStrength);
		mainShader.setFloat("shadowBias", shaderParams.shadowBias);
		mainShader.setFloat("shadowDarkness", shaderParams.shadowDarkness);
		mainShader.setInt("shadowsEnabled", shaderParams.shadowsEnabled ? 1 : 0);

		shadowMap.BindForReading(GL_TEXTURE0);
		mainShader.setInt("shadowMap", 0);

		mainShader.setAlpha(1.0f);
		model = glm::mat4(1.0f);
		mainShader.setMat4("model", model);
		floorMesh.Draw();

		mainShader.setAlpha(shaderParams.cubeAlpha);
		model = glm::mat4(1.0f);
		model = glm::rotate(model, currentFrame * 0.5f, glm::vec3(0.5f, 1.0f, 0.0f));
		mainShader.setMat4("model", model);
		cube.Draw();

		bool shouldShowTheLightSphere = shaderParams.showLightSource;

		if (shouldShowTheLightSphere) {
			lightSphereShader.use();
			lightSphereShader.setMat4("projection", projection);
			lightSphereShader.setMat4("view", view);
			lightSphereShader.setVec3("lightColor", currentLightColor);

			glm::mat4 lightSphereModel = glm::mat4(1.0f);
			lightSphereModel = glm::translate(lightSphereModel, lightPos);
			lightSphereModel = glm::scale(lightSphereModel, glm::vec3(0.7f));
			lightSphereShader.setMat4("model", lightSphereModel);
			lightSphere.Draw();
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}