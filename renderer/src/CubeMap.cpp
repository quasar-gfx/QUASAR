#include <CubeMap.h>
#include <Cameras/PerspectiveCamera.h>
#include <Cameras/VRCamera.h>
#include <Utils/FileIO.h>

using namespace quasar;

const glm::mat4 CubeMap::captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
const glm::mat4 CubeMap::captureViews[] = {
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
};

CubeMap::CubeMap()
    : vertexBuffer({
        .target = GL_ARRAY_BUFFER,
        .dataSize = sizeof(CubeMapVertex),
    })
{
    target = GL_TEXTURE_CUBE_MAP;
}

CubeMap::CubeMap(const CubeMapCreateParams& params)
    : type(params.type)
    , wrapR(params.wrapR)
    , vertexBuffer({
        .target = GL_ARRAY_BUFFER,
        .dataSize = sizeof(CubeMapVertex),
    })
    , Texture({
        .width = params.width,
        .height = params.height,
        .format = params.format,
        .wrapS = params.wrapS,
        .wrapT = params.wrapT,
        .minFilter = params.minFilter,
        .magFilter = params.magFilter,
    })
{
    target = GL_TEXTURE_CUBE_MAP;
    if (params.rightFaceTexturePath != "" && params.leftFaceTexturePath != "" &&
        params.topFaceTexturePath != "" && params.bottomFaceTexturePath != "" &&
        params.frontFaceTexturePath != "" && params.backFaceTexturePath != "") {
        initBuffers();
        std::vector<std::string> faceTexturePaths = {
            params.rightFaceTexturePath,
            params.leftFaceTexturePath,
            params.topFaceTexturePath,
            params.bottomFaceTexturePath,
            params.frontFaceTexturePath,
            params.backFaceTexturePath
        };
        loadFromFiles(faceTexturePaths, params.format, params.wrapS, params.wrapT, params.wrapR, params.minFilter, params.magFilter);
    }
    else {
        init(params.width, params.height, params.type);
    }
}

CubeMap::~CubeMap() {
    glDeleteBuffers(1, &vertexArrayBuffer);
    glDeleteTextures(1, &ID);
}

void CubeMap::init(uint width, uint height, CubeMapType type) {
    initBuffers();

    glGenTextures(1, &ID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, ID);

    switch (type) {
    case CubeMapType::STANDARD:
        for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        break;

    case CubeMapType::SHADOW:
        for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
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

    case CubeMapType::HDR:
        for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, nullptr);
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        break;

    case CubeMapType::PREFILTER:
        for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_HALF_FLOAT, nullptr);
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

void CubeMap::initBuffers() {
    std::vector<CubeMapVertex> skyboxVertices = {
        { {-1.0f, 1.0f, -1.0f} },
        { {-1.0f, -1.0f, -1.0f} },
        { {1.0f, -1.0f, -1.0f} },
        { {1.0f, -1.0f, -1.0f} },
        { {1.0f, 1.0f, -1.0f} },
        { {-1.0f, 1.0f, -1.0f} },

        { {-1.0f, -1.0f, 1.0f} },
        { {-1.0f, -1.0f, -1.0f} },
        { {-1.0f, 1.0f, -1.0f} },
        { {-1.0f, 1.0f, -1.0f} },
        { {-1.0f, 1.0f, 1.0f} },
        { {-1.0f, -1.0f, 1.0f} },

        { {1.0f, -1.0f, -1.0f} },
        { {1.0f, -1.0f, 1.0f} },
        { {1.0f, 1.0f, 1.0f} },
        { {1.0f, 1.0f, 1.0f} },
        { {1.0f, 1.0f, -1.0f} },
        { {1.0f, -1.0f, -1.0f} },

        { {-1.0f, -1.0f, 1.0f} },
        { {-1.0f, 1.0f, 1.0f} },
        { {1.0f, 1.0f, 1.0f} },
        { {1.0f, 1.0f, 1.0f} },
        { {1.0f, -1.0f, 1.0f} },
        { {-1.0f, -1.0f, 1.0f} },

        { {-1.0f, 1.0f, -1.0f} },
        { {1.0f, 1.0f, -1.0f} },
        { {1.0f, 1.0f, 1.0f} },
        { {1.0f, 1.0f, 1.0f} },
        { {-1.0f, 1.0f, 1.0f} },
        { {-1.0f, 1.0f, -1.0f} },

        { {-1.0f, -1.0f, -1.0f} },
        { {-1.0f, -1.0f, 1.0f} },
        { {1.0f, -1.0f, -1.0f} },
        { {1.0f, -1.0f, -1.0f} },
        { {-1.0f, -1.0f, 1.0f} },
        { {1.0f, -1.0f, 1.0f} },
    };
    vertexBuffer.bind();
    vertexBuffer.setData(skyboxVertices.size(), skyboxVertices.data());

    glGenVertexArrays(1, &vertexArrayBuffer);
    glBindVertexArray(vertexArrayBuffer);

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
    for (uint i = 0; i < faceFilePaths.size(); i++) {
        void* data = FileIO::loadImage(faceFilePaths[i], &textureWidth, &textureHeight, &nrChannels, 0);
        if (data) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, format, textureWidth, textureHeight, 0, format, GL_UNSIGNED_BYTE, data);
            FileIO::freeImage(data);
        }
        else {
            throw std::runtime_error("Cubemap tex failed to load at path: " + faceFilePaths[i]);
        }
    }

    width = textureWidth;
    height = textureHeight;

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, wrapS);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, wrapT);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, wrapR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, magFilter);
}

void CubeMap::loadFromEquirectTexture(const Shader& equirectToCubeMapShader, const Texture& equirectTexture) const {
    equirectToCubeMapShader.bind();

    equirectToCubeMapShader.setTexture("equirectangularMap", equirectTexture, 0);
    equirectToCubeMapShader.setMat4("projection", captureProjection);

    glViewport(0, 0, width, height);

    for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
        equirectToCubeMapShader.setMat4("view", captureViews[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, ID, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        drawCube();
    }

    equirectTexture.unbind();
}

void CubeMap::convolve(const Shader& convolutionShader, const CubeMap& envCubeMap) const {
    convolutionShader.bind();

    convolutionShader.setTexture("environmentMap", envCubeMap, 0);
    convolutionShader.setMat4("projection", captureProjection);

    glViewport(0, 0, width, height);

    for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
        convolutionShader.setMat4("view", captureViews[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, ID, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        drawCube();
    }
}

void CubeMap::prefilter(const Shader& prefilterShader, const CubeMap& envCubeMap, Renderbuffer& captureRBO) const {
    prefilterShader.bind();

    prefilterShader.setTexture("environmentMap", envCubeMap, 0);
    prefilterShader.setMat4("projection", captureProjection);

    for (int mip = 0; mip < maxMipLevels; mip++) {
        uint mipWidth = static_cast<uint>(width * std::pow(0.5f, mip));
        uint mipHeight = static_cast<uint>(height * std::pow(0.5f, mip));

        captureRBO.bind();
        captureRBO.resize(mipWidth, mipHeight);
        glViewport(0, 0, mipWidth, mipHeight);

        float roughness = (float)mip / static_cast<float>(maxMipLevels - 1);
        prefilterShader.setFloat("roughness", roughness);

        for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
            prefilterShader.setMat4("view", captureViews[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, ID, mip);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            drawCube();
        }
    }
}

RenderStats CubeMap::draw(const Shader& shader, const Camera& camera) const {
    RenderStats stats;

    shader.bind();

    glm::mat4 view;
    glm::mat4 projection;
    if (camera.isVR()) {
        auto& vrCamera = static_cast<const VRCamera&>(camera);
        view = glm::mat4(glm::mat3(vrCamera.left.getViewMatrix()));
        projection = vrCamera.getProjectionMatrix();
    }
    else {
        auto& monoCamera = static_cast<const PerspectiveCamera&>(camera);
        view = monoCamera.getViewMatrix();
        projection = monoCamera.getProjectionMatrix();
    }

    shader.setUint("DrawID", ID);

    // Remove translation from the view matrix
    view = glm::mat4(glm::mat3(view));
    shader.setMat4("view", view);
    shader.setMat4("projection", projection);

    bind();
    stats += drawCube();
    unbind();

    stats.drawCalls = 1;

    return stats;
}

RenderStats CubeMap::drawCube() const {
    RenderStats stats;
    stats.trianglesDrawn = 12;

    glBindVertexArray(vertexArrayBuffer);
    glDrawArrays(GL_TRIANGLES, 0, stats.trianglesDrawn * 3);
    glBindVertexArray(0);

    return stats;
}
