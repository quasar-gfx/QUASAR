#include <QuadsGenerator.h>
#include <Utils/TimeUtils.h>

#define THREADS_PER_LOCALGROUP 16

#define MAX_PROXY_SIZE 1024

QuadsGenerator::QuadsGenerator(const glm::uvec2 &remoteWindowSize)
        : remoteWindowSize(remoteWindowSize)
        , depthBufferSize(2u * remoteWindowSize)
        , maxQuads(remoteWindowSize.x * remoteWindowSize.y)
        , genQuadMapShader({
            .computeCodePath = "shaders/genQuadMap.comp",
            .defines = {
                "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
            }
        })
        , simplifyQuadMapShader({
            .computeCodePath = "shaders/simplifyQuadMap.comp",
            .defines = {
                "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
            }
        })
        , fillOutputQuadsShader({
            .computeCodePath = "shaders/fillOutputQuads.comp",
            .defines = {
                "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
            }
        })
        , sizesBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, 1, nullptr)
        , outputQuadBuffers(maxQuads)
        , depthOffsetsBuffer({
            .width = depthBufferSize.x,
            .height = depthBufferSize.y,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_HALF_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST
        }) {
    // make sure maxProxySize is a power of 2
    glm::uvec2 maxProxySize = remoteWindowSize;
    maxProxySize.x = 1 << static_cast<int>(glm::ceil(glm::log2(static_cast<float>(maxProxySize.x))));
    maxProxySize.y = 1 << static_cast<int>(glm::ceil(glm::log2(static_cast<float>(maxProxySize.y))));
    maxProxySize = glm::min(maxProxySize, glm::uvec2(MAX_PROXY_SIZE));

    numQuadMaps = glm::log2(static_cast<float>(glm::min(maxProxySize.x, maxProxySize.y)));

    // create quad buffers
    quadBuffers.reserve(numQuadMaps);
    quadMapSizes.reserve(numQuadMaps);

    glm::uvec2 currQuadMapSize = remoteWindowSize;
    for (int i = 0; i < numQuadMaps; i++) {
        quadBuffers.emplace_back(currQuadMapSize.x * currQuadMapSize.y);

        quadMapSizes.emplace_back(currQuadMapSize);
        currQuadMapSize = glm::max(currQuadMapSize / 2u, glm::uvec2(1u));
    }

    // set stuff that won't change
    genQuadMapShader.bind();
    genQuadMapShader.setVec2("remoteWindowSize", remoteWindowSize);
    genQuadMapShader.setVec2("depthBufferSize", depthBufferSize);
    genQuadMapShader.setVec2("quadMapSize", quadMapSizes[0]);
    genQuadMapShader.setImageTexture(0, depthOffsetsBuffer, 0, GL_FALSE, 0, GL_READ_WRITE, depthOffsetsBuffer.internalFormat);

    simplifyQuadMapShader.bind();
    simplifyQuadMapShader.setVec2("remoteWindowSize", remoteWindowSize);

    fillOutputQuadsShader.bind();
    fillOutputQuadsShader.setVec2("remoteWindowSize", remoteWindowSize);
}

QuadsGenerator::BufferSizes QuadsGenerator::getBufferSizes() {
    BufferSizes bufferSizes;

    sizesBuffer.bind();
    sizesBuffer.getData(&bufferSizes);
    return bufferSizes;
}

void QuadsGenerator::generateInitialQuadMap(const GeometryBuffer& gBuffer, const PerspectiveCamera &remoteCamera) {
    /*
    ============================
    SECOND PASS: Generate quads from G-Buffer
    ============================
    */
    genQuadMapShader.startTiming();

    genQuadMapShader.bind();
    {
        genQuadMapShader.setTexture(gBuffer.normalsBuffer, 0);
        genQuadMapShader.setTexture(gBuffer.depthStencilBuffer, 1);
    }
    {
        genQuadMapShader.setMat4("view", remoteCamera.getViewMatrix());
        genQuadMapShader.setMat4("projection", remoteCamera.getProjectionMatrix());
        genQuadMapShader.setMat4("viewInverse", remoteCamera.getViewMatrixInverse());
        genQuadMapShader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
        genQuadMapShader.setFloat("near", remoteCamera.getNear());
        genQuadMapShader.setFloat("far", remoteCamera.getFar());
    }
    {
        genQuadMapShader.setBool("doOrientationCorrection", doOrientationCorrection);
        genQuadMapShader.setFloat("distanceThreshold", distanceThreshold);
        genQuadMapShader.setFloat("angleThreshold", glm::radians(angleThreshold));
        genQuadMapShader.setFloat("flatThreshold", flatThreshold * 0.01f);
    }
    {
        genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, sizesBuffer);

        genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, quadBuffers[0].normalSphericalsBuffer);
        genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, quadBuffers[0].depthsBuffer);
        genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, quadBuffers[0].offsetSizeFlattenedsBuffer);
    }
    genQuadMapShader.dispatch((remoteWindowSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                              (remoteWindowSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    genQuadMapShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    genQuadMapShader.endTiming();
    stats.timeToGenerateQuadsMs = genQuadMapShader.getElapsedTime();
}

void QuadsGenerator::simplifyQuadMaps(const PerspectiveCamera &remoteCamera) {
    /*
    ============================
    THIRD PASS: Simplify quad map
    ============================
    */
    simplifyQuadMapShader.startTiming();

    simplifyQuadMapShader.bind();
    {
        simplifyQuadMapShader.setMat4("view", remoteCamera.getViewMatrix());
        simplifyQuadMapShader.setMat4("projection", remoteCamera.getProjectionMatrix());
        simplifyQuadMapShader.setMat4("viewInverse", remoteCamera.getViewMatrixInverse());
        simplifyQuadMapShader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
        simplifyQuadMapShader.setFloat("near", remoteCamera.getNear());
        simplifyQuadMapShader.setFloat("far", remoteCamera.getFar());
    }
    {
        simplifyQuadMapShader.setVec2("depthBufferSize", depthBufferSize);
    }
    {
        simplifyQuadMapShader.setFloat("flatThreshold", flatThreshold * 0.01f);
        simplifyQuadMapShader.setFloat("proxySimilarityThreshold", proxySimilarityThreshold);
    }
    {
        simplifyQuadMapShader.setImageTexture(0, depthOffsetsBuffer, 0, GL_FALSE, 0, GL_READ_WRITE, depthOffsetsBuffer.internalFormat);
    }
    for (int i = 1; i < numQuadMaps; i++) {
        auto& prevQuadMapSize = quadMapSizes[i-1];
        auto& prevQuadBuffers = quadBuffers[i-1];

        auto& currQuadMapSize = quadMapSizes[i];
        auto& currQuadBuffers = quadBuffers[i];

        {
            simplifyQuadMapShader.setVec2("inputQuadMapSize", prevQuadMapSize);
            simplifyQuadMapShader.setVec2("outputQuadMapSize", currQuadMapSize);
        }
        {
            simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, prevQuadBuffers.normalSphericalsBuffer);
            simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, prevQuadBuffers.depthsBuffer);
            simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, prevQuadBuffers.offsetSizeFlattenedsBuffer);

            simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, currQuadBuffers.normalSphericalsBuffer);
            simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, currQuadBuffers.depthsBuffer);
            simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, currQuadBuffers.offsetSizeFlattenedsBuffer);
        }
        simplifyQuadMapShader.dispatch((currQuadMapSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                       (currQuadMapSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
        simplifyQuadMapShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
    simplifyQuadMapShader.memoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    simplifyQuadMapShader.endTiming();
    stats.timeToSimplifyQuadsMs = simplifyQuadMapShader.getElapsedTime();
}

void QuadsGenerator::fillOutputQuads() {
    /*
    ============================
    FOURTH PASS: Fill output quads buffer
    ============================
    */
    fillOutputQuadsShader.startTiming();

    fillOutputQuadsShader.bind();
    for (int i = 0; i < numQuadMaps; i++) {
        auto& currQuadBuffers = quadBuffers[i];
        auto& currQuadMapSize = quadMapSizes[i];

        {
            fillOutputQuadsShader.setVec2("quadMapSize", currQuadMapSize);
        }
        {
            fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, sizesBuffer);

            fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, currQuadBuffers.normalSphericalsBuffer);
            fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, currQuadBuffers.depthsBuffer);
            fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, currQuadBuffers.offsetSizeFlattenedsBuffer);

            fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, outputQuadBuffers.normalSphericalsBuffer);
            fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, outputQuadBuffers.depthsBuffer);
            fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, outputQuadBuffers.offsetSizeFlattenedsBuffer);
        }
        fillOutputQuadsShader.dispatch((currQuadMapSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                       (currQuadMapSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    }
    fillOutputQuadsShader.memoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    fillOutputQuadsShader.endTiming();
    stats.timeToFillOutputQuadsMs = fillOutputQuadsShader.getElapsedTime();
}

QuadsGenerator::BufferSizes QuadsGenerator::createProxiesFromGBuffer(const GeometryBuffer& gBuffer, const PerspectiveCamera &remoteCamera) {
    generateInitialQuadMap(gBuffer, remoteCamera);
    simplifyQuadMaps(remoteCamera);
    fillOutputQuads();

    QuadsGenerator::BufferSizes bufferSizes = getBufferSizes();
    unsigned int numProxies = bufferSizes.numProxies;
    outputQuadBuffers.resize(numProxies);

    return bufferSizes;
}

#ifdef GL_CORE
unsigned int QuadsGenerator::saveProxiesToFile(const std::string &filename) {
    auto bufferSizes = getBufferSizes();
    unsigned int numProxies = bufferSizes.numProxies;
    outputQuadBuffers.resize(numProxies);
    return outputQuadBuffers.saveProxiesToFile(filename);
}
#endif
