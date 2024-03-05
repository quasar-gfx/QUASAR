#include <iostream>

#include <stb_image.h>

#include <CubeMap.h>

const glm::mat4 CubeMap::captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
const glm::mat4 CubeMap::captureViews[] = {
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
};

CubeMap::CubeMap(unsigned int width, unsigned int height, CubeMapType cubeType) {
    glGenTextures(1, &ID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, ID);

    switch(cubeType) {
    case CUBE_MAP_SHADOW:
        for (int i = 0; i < 6; ++i){
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                            0, GL_DEPTH_COMPONENT, width, height, 0,
                            GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        break;

    case CUBE_MAP_HDR:
        for (int i = 0; i < 6; ++i){
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                            0, GL_RGB32F, width, height, 0,
                            GL_RGB, GL_FLOAT, NULL);
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        break;

    case CUBE_MAP_PREFILTER:
        for (unsigned int i = 0; i < 6; ++i){
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                            0, GL_RGB16F,
                            width, height, 0,
                            GL_RGB, GL_FLOAT, NULL);
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
        maxMipLevels = 5;
        break;

    default:
        throw std::runtime_error("Invalid CubeMapType");
        break;
    }
}

CubeMap::CubeMap(std::vector<std::string> faceFilePaths,
        GLenum format,
        GLint wrapS, GLint wrapT, GLint wrapR,
        GLint minFilter, GLint magFilter) {
    std::vector<CubeMapVertex> skyboxVertices = {
        { {-1.0f,  1.0f, -1.0f} },
        { {-1.0f, -1.0f, -1.0f} },
        { { 1.0f, -1.0f, -1.0f} },
        { { 1.0f, -1.0f, -1.0f} },
        { { 1.0f,  1.0f, -1.0f} },
        { {-1.0f,  1.0f, -1.0f} },

        { {-1.0f, -1.0f,  1.0f} },
        { {-1.0f, -1.0f, -1.0f} },
        { {-1.0f,  1.0f, -1.0f} },
        { {-1.0f,  1.0f, -1.0f} },
        { {-1.0f,  1.0f,  1.0f} },
        { {-1.0f, -1.0f,  1.0f} },

        { { 1.0f, -1.0f, -1.0f} },
        { { 1.0f, -1.0f,  1.0f} },
        { { 1.0f,  1.0f,  1.0f} },
        { { 1.0f,  1.0f,  1.0f} },
        { { 1.0f,  1.0f, -1.0f} },
        { { 1.0f, -1.0f, -1.0f} },

        { {-1.0f, -1.0f,  1.0f} },
        { {-1.0f,  1.0f,  1.0f} },
        { { 1.0f,  1.0f,  1.0f} },
        { { 1.0f,  1.0f,  1.0f} },
        { { 1.0f, -1.0f,  1.0f} },
        { {-1.0f, -1.0f,  1.0f} },

        { {-1.0f,  1.0f, -1.0f} },
        { { 1.0f,  1.0f, -1.0f} },
        { { 1.0f,  1.0f,  1.0f} },
        { { 1.0f,  1.0f,  1.0f} },
        { {-1.0f,  1.0f,  1.0f} },
        { {-1.0f,  1.0f, -1.0f} },

        { {-1.0f, -1.0f, -1.0f} },
        { {-1.0f, -1.0f,  1.0f} },
        { { 1.0f, -1.0f, -1.0f} },
        { { 1.0f, -1.0f, -1.0f} },
        { {-1.0f, -1.0f,  1.0f} },
        { { 1.0f, -1.0f,  1.0f} }
    };

    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);

    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);

    glBufferData(GL_ARRAY_BUFFER, skyboxVertices.size() * sizeof(CubeMapVertex), skyboxVertices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(CubeMapVertex), (void*)0);

    loadFromFiles(faceFilePaths, format, wrapS, wrapT, wrapR, minFilter, magFilter);
}

void CubeMap::loadFromFiles(std::vector<std::string> faceFilePaths,
            GLenum format,
            GLint wrapS, GLint wrapT, GLint wrapR,
            GLint minFilter, GLint magFilter) {
    glGenTextures(1, &ID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, ID);

    int width, height, nrChannels;
    for (unsigned int i = 0; i < faceFilePaths.size(); i++) {
        unsigned char *data = stbi_load(faceFilePaths[i].c_str(), &width, &height, &nrChannels, 0);
        if (data) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                        0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data
            );
            stbi_image_free(data);
        }
        else {
            throw std::runtime_error("Cubemap tex failed to load at path: " + faceFilePaths[i]);
            stbi_image_free(data);
        }
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, wrapS);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, wrapT);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, wrapR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, magFilter);
}

