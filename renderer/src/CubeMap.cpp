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
    this->cubeType = cubeType;

    glGenTextures(1, &ID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, ID);

    switch(cubeType) {
    case CUBE_MAP_STANDARD:
        for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        break;

    case CUBE_MAP_SHADOW:
        for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
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
        for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, NULL);
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        break;

    case CUBE_MAP_PREFILTER:
        for (unsigned int i = 0; i < NUM_CUBEMAP_FACES; i++) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, NULL);
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
        CubeMapType cubeType,
        GLenum format,
        GLint wrapS, GLint wrapT, GLint wrapR,
        GLint minFilter, GLint magFilter) {
    this->faceFilePaths = faceFilePaths;
    this->cubeType = cubeType;

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

void CubeMap::loadFromEquirectTexture(Shader &equirectToCubeMapShader, unsigned int width, unsigned int height, Texture &equirectTexture) {
    equirectToCubeMapShader.bind();

    equirectToCubeMapShader.setInt("equirectangularMap", 0);
    equirectToCubeMapShader.setMat4("projection", captureProjection);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, ID);

    equirectTexture.bind(0);
    glViewport(0, 0, width, height);

    for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
        equirectToCubeMapShader.setMat4("view", captureViews[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, equirectTexture.ID, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        drawCube();
    }

    equirectTexture.unbind();

    equirectToCubeMapShader.unbind();
}

void CubeMap::convolve(Shader &convolutionShader, unsigned int width, unsigned int height, TextureID envMapTexture) {
    convolutionShader.bind();

    convolutionShader.setInt("environmentMap", 0);
    convolutionShader.setMat4("projection", captureProjection);

    glViewport(0, 0, width, height);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, envMapTexture);

    for (int i = 0; i < NUM_CUBEMAP_FACES; i++){
        convolutionShader.setMat4("view", captureViews[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, ID, 0);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        drawCube();
    }

    convolutionShader.unbind();
}

void CubeMap::prefilter(Shader &prefilterShader, unsigned int width, unsigned int height, TextureID envMapTexture, Texture &captureRBO) {
    prefilterShader.bind();

    prefilterShader.setInt("environmentMap", 0);
    prefilterShader.setMat4("projection", captureProjection);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, envMapTexture);

    for (int mip = 0; mip < maxMipLevels; mip++) {
        unsigned int mipWidth  = static_cast<unsigned int>(width * std::pow(0.5f, mip));
        unsigned int mipHeight = static_cast<unsigned int>(height * std::pow(0.5f, mip));

        glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, mipWidth, mipHeight);
        glViewport(0, 0, mipWidth, mipHeight);

        for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
            prefilterShader.setMat4("view", captureViews[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, ID, mip);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            drawCube();
        }
    }

    prefilterShader.unbind();
}


void CubeMap::draw(Shader &shader, Camera* camera) {
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_FALSE);

    shader.bind();

    glm::mat4 view = glm::mat4(glm::mat3(camera->getViewMatrix()));
    shader.setMat4("view", view);
    shader.setMat4("projection", camera->getProjectionMatrix());

    drawCube();

    shader.unbind();

    // restore depth func
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);
}

void CubeMap::drawCube() {
    glBindVertexArray(quadVAO);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, ID);

    glDrawArrays(GL_TRIANGLES, 0, 36);

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    glBindVertexArray(0);
}
