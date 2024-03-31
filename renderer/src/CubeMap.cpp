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

void CubeMap::init(unsigned int width, unsigned int height, CubeMapType cubeType) {
    this->width = width;
    this->height = height;
    this->cubeType = cubeType;

    initBuffers();

    glGenTextures(1, &ID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, ID);

    switch(cubeType) {
    case CUBE_MAP_STANDARD:
        for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        break;

    case CUBE_MAP_SHADOW:
        for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        // glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
        // glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        // glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        break;

    case CUBE_MAP_HDR:
        for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA32F, width, height, 0, GL_RGB, GL_FLOAT, nullptr);
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        break;

    case CUBE_MAP_PREFILTER:
        for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, nullptr);
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); // be sure to set minification filter to mip_linear
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

    initBuffers();
    loadFromFiles(faceFilePaths, format, wrapS, wrapT, wrapR, minFilter, magFilter);
}

void CubeMap::initBuffers() {
    std::vector<CubeMapVertex> skyboxVertices = {
        // back face
        { {-1.0f, -1.0f, -1.0f, } },
        { { 1.0f,  1.0f, -1.0f, } },
        { { 1.0f, -1.0f, -1.0f, } },
        { { 1.0f,  1.0f, -1.0f, } },
        { {-1.0f, -1.0f, -1.0f, } },
        { {-1.0f,  1.0f, -1.0f, } },
        // front face
        { {-1.0f, -1.0f,  1.0f, } },
        { { 1.0f, -1.0f,  1.0f, } },
        { { 1.0f,  1.0f,  1.0f, } },
        { { 1.0f,  1.0f,  1.0f, } },
        { {-1.0f,  1.0f,  1.0f, } },
        { {-1.0f, -1.0f,  1.0f, } },
        // left face
        { {-1.0f,  1.0f,  1.0f, } },
        { {-1.0f,  1.0f, -1.0f, } },
        { {-1.0f, -1.0f, -1.0f, } },
        { {-1.0f, -1.0f, -1.0f, } },
        { {-1.0f, -1.0f,  1.0f, } },
        { {-1.0f,  1.0f,  1.0f, } },
        // right face
        { { 1.0f,  1.0f,  1.0f, } },
        { { 1.0f, -1.0f, -1.0f, } },
        { { 1.0f,  1.0f, -1.0f, } },
        { { 1.0f, -1.0f, -1.0f, } },
        { { 1.0f,  1.0f,  1.0f, } },
        { { 1.0f, -1.0f,  1.0f, } },
        // bottom face
        { {-1.0f, -1.0f, -1.0f, } },
        { { 1.0f, -1.0f, -1.0f, } },
        { { 1.0f, -1.0f,  1.0f, } },
        { { 1.0f, -1.0f,  1.0f, } },
        { {-1.0f, -1.0f,  1.0f, } },
        { {-1.0f, -1.0f, -1.0f, } },
        // top face
        { {-1.0f,  1.0f, -1.0f, } },
        { { 1.0f,  1.0f , 1.0f, } },
        { { 1.0f,  1.0f, -1.0f, } },
        { { 1.0f,  1.0f,  1.0f, } },
        { {-1.0f,  1.0f, -1.0f, } },
        { {-1.0f,  1.0f,  1.0f, } }
    };

    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);

    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);

    glBufferData(GL_ARRAY_BUFFER, skyboxVertices.size() * sizeof(CubeMapVertex), skyboxVertices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(CubeMapVertex), (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void CubeMap::loadFromFiles(std::vector<std::string> faceFilePaths,
            GLenum format,
            GLint wrapS, GLint wrapT, GLint wrapR,
            GLint minFilter, GLint magFilter) {
    glGenTextures(1, &ID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, ID);

    int textureWidth, textureHeight, nrChannels;
    this->width = textureWidth;
    this->height = textureHeight;
    for (unsigned int i = 0; i < faceFilePaths.size(); i++) {
        void* data = stbi_load(faceFilePaths[i].c_str(), &textureWidth, &textureHeight, &nrChannels, 0);
        if (data) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, format, textureWidth, textureHeight, 0, format, GL_UNSIGNED_BYTE, data);
            stbi_image_free(data);
        }
        else {
            throw std::runtime_error("Cubemap tex failed to load at path: " + faceFilePaths[i]);
        }
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, wrapS);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, wrapT);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, wrapR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, magFilter);
}

void CubeMap::loadFromEquirectTexture(Shader &equirectToCubeMapShader, Texture &equirectTexture) {
    equirectToCubeMapShader.bind();

    equirectToCubeMapShader.setInt("equirectangularMap", 0);
    equirectToCubeMapShader.setMat4("projection", captureProjection);

    equirectTexture.bind(0);
    glViewport(0, 0, width, height);

    for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
        equirectToCubeMapShader.setMat4("view", captureViews[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, ID, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        drawCube();
    }

    equirectTexture.unbind();

    equirectToCubeMapShader.unbind();
}

void CubeMap::convolve(Shader &convolutionShader, CubeMap &envCubeMap) {
    convolutionShader.bind();

    convolutionShader.setInt("environmentMap", 0);
    convolutionShader.setMat4("projection", captureProjection);

    envCubeMap.bind(0);
    glViewport(0, 0, width, height);

    for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
        convolutionShader.setMat4("view", captureViews[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, ID, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        drawCube();
    }

    convolutionShader.unbind();
}

void CubeMap::prefilter(Shader &prefilterShader, CubeMap &envCubeMap, Renderbuffer &captureRBO) {
    prefilterShader.bind();

    prefilterShader.setInt("environmentMap", 0);
    prefilterShader.setMat4("projection", captureProjection);

    envCubeMap.bind(0);

    for (int mip = 0; mip < maxMipLevels; mip++) {
        unsigned int mipWidth = static_cast<unsigned int>(width * std::pow(0.5f, mip));
        unsigned int mipHeight = static_cast<unsigned int>(height * std::pow(0.5f, mip));

        captureRBO.bind();
        captureRBO.resize(mipWidth, mipHeight);
        glViewport(0, 0, mipWidth, mipHeight);

        float roughness = (float)mip / (float)(maxMipLevels - 1);
        prefilterShader.setFloat("roughness", roughness);

        for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
            prefilterShader.setMat4("view", captureViews[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, ID, mip);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            drawCube();
        }
    }

    prefilterShader.unbind();
}

void CubeMap::draw(Shader &shader, Camera &camera) {
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_FALSE);

    shader.bind();

    glm::mat4 view = glm::mat4(glm::mat3(camera.getViewMatrix()));
    shader.setMat4("view", view);
    shader.setMat4("projection", camera.getProjectionMatrix());

    bind();
    drawCube();
    unbind();

    shader.unbind();

    // restore depth func
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);
}

void CubeMap::drawCube() {
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
}
