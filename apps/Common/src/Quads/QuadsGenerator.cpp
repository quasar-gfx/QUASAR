#include <Quads/QuadsGenerator.h>
#include <Utils/TimeUtils.h>

#define THREADS_PER_LOCALGROUP 2

#define MAX_PROXY_SIZE 1024

QuadsGenerator::QuadsGenerator(glm::uvec2 &remoteWindowSize)
        : remoteWindowSize(remoteWindowSize)
        , depthBufferSize(2u * remoteWindowSize) // 4 offsets per pixel
        , maxProxies(remoteWindowSize.x * remoteWindowSize.y)
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
        , sizesBuffer(GL_SHADER_STORAGE_BUFFER, 1, sizeof(BufferSizes), nullptr, GL_DYNAMIC_COPY)
        , depthOffsets(depthBufferSize)
        , outputQuadBuffers(maxProxies) {
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
}

QuadsGenerator::BufferSizes QuadsGenerator::getBufferSizes() {
    BufferSizes bufferSizes;

    sizesBuffer.bind();
    sizesBuffer.getData(&bufferSizes);
    return bufferSizes;
}

void QuadsGenerator::generateInitialQuadMap(
        const GBuffer& gBuffer,
        const GBuffer &gBufferHighRes,
        const glm::vec2 &gBufferSize,
        const PerspectiveCamera &remoteCamera
    ) {
    /*
    ============================
    FIRST PASS: Generate quads from G-Buffer
    ============================
    */
    genQuadMapShader.startTiming();

    int closestQuadMapIdx = 0;
    for (int i = 0; i < numQuadMaps; i++) {
        if (gBufferSize.x <= quadMapSizes[i].x && gBufferSize.y <= quadMapSizes[i].y) {
            closestQuadMapIdx = i;
        }
    }

    genQuadMapShader.bind();
    {
        genQuadMapShader.setVec2("gBufferSize", gBufferSize);
        genQuadMapShader.setVec2("gBufferHighResSize", glm::vec2(gBufferHighRes.width, gBufferHighRes.height));
        genQuadMapShader.setVec2("depthBufferSize", depthOffsets.size);
        genQuadMapShader.setVec2("quadMapSize", quadMapSizes[closestQuadMapIdx]);
    }
    {
        genQuadMapShader.setTexture(gBuffer.normalsBuffer, 0);
        genQuadMapShader.setTexture(gBuffer.depthStencilBuffer, 1);
        genQuadMapShader.setTexture(gBufferHighRes.normalsBuffer, 2);
        genQuadMapShader.setTexture(gBufferHighRes.depthStencilBuffer, 3);
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
        genQuadMapShader.setBool("expandEdges", expandEdges);
        genQuadMapShader.setBool("correctOrientation", correctOrientation);
        genQuadMapShader.setFloat("depthThreshold", depthThreshold);
        genQuadMapShader.setFloat("angleThreshold", glm::radians(angleThreshold));
        genQuadMapShader.setFloat("flatThreshold", flatThreshold);
    }
    {
        genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, sizesBuffer);

        genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, quadBuffers[closestQuadMapIdx].normalSphericalsBuffer);
        genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, quadBuffers[closestQuadMapIdx].depthsBuffer);
        genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, quadBuffers[closestQuadMapIdx].offsetSizeFlattenedsBuffer);

        genQuadMapShader.setImageTexture(0, depthOffsets.buffer, 0, GL_FALSE, 0, GL_READ_WRITE, depthOffsets.buffer.internalFormat);
        genQuadMapShader.setImageTexture(1, gBuffer.colorBuffer, 0, GL_FALSE, 0, GL_READ_WRITE, gBuffer.colorBuffer.internalFormat);
        genQuadMapShader.setImageTexture(2, gBufferHighRes.colorBuffer, 0, GL_FALSE, 0, GL_READ_WRITE, gBufferHighRes.colorBuffer.internalFormat);
    }
    genQuadMapShader.dispatch((gBufferSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                              (gBufferSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    genQuadMapShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    genQuadMapShader.endTiming();
    stats.timeToGenerateQuadsMs = genQuadMapShader.getElapsedTime();
}

void QuadsGenerator::simplifyQuadMaps(const PerspectiveCamera &remoteCamera, const glm::vec2 &gBufferSize) {
    /*
    ============================
    SECOND PASS: Simplify quad map
    ============================
    */
    simplifyQuadMapShader.startTiming();

    int closestQuadMapIdx = 0;
    for (int i = 1; i < numQuadMaps; i++) {
        if (gBufferSize.x <= quadMapSizes[i].x && gBufferSize.y <= quadMapSizes[i].y) {
            closestQuadMapIdx = i;
        }
    }

    simplifyQuadMapShader.bind();
    {
        simplifyQuadMapShader.setVec2("gBufferSize", gBufferSize);
        simplifyQuadMapShader.setVec2("depthBufferSize", depthOffsets.size);
    }
    {
        simplifyQuadMapShader.setMat4("view", remoteCamera.getViewMatrix());
        simplifyQuadMapShader.setMat4("projection", remoteCamera.getProjectionMatrix());
        simplifyQuadMapShader.setMat4("viewInverse", remoteCamera.getViewMatrixInverse());
        simplifyQuadMapShader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
        simplifyQuadMapShader.setFloat("near", remoteCamera.getNear());
        simplifyQuadMapShader.setFloat("far", remoteCamera.getFar());
    }
    {
        simplifyQuadMapShader.setFloat("flatThreshold", flatThreshold);
        simplifyQuadMapShader.setFloat("proxySimilarityThreshold", proxySimilarityThreshold);
    }
    {
        simplifyQuadMapShader.setImageTexture(0, depthOffsets.buffer, 0, GL_FALSE, 0, GL_READ_WRITE, depthOffsets.buffer.internalFormat);
    }
    for (int i = closestQuadMapIdx + 1; i < numQuadMaps; i++) {
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

void QuadsGenerator::fillOutputQuads(const glm::vec2 &gBufferSize) {
    /*
    ============================
    THIRD PASS: Fill output quads buffer
    ============================
    */
    fillOutputQuadsShader.startTiming();

    int closestQuadMapIdx = 0;
    for (int i = 1; i < numQuadMaps; i++) {
        if (gBufferSize.x <= quadMapSizes[i].x && gBufferSize.y <= quadMapSizes[i].y) {
            closestQuadMapIdx = i;
        }
    }

    fillOutputQuadsShader.bind();
    {
        fillOutputQuadsShader.setVec2("gBufferSize", gBufferSize);
    }
    for (int i = closestQuadMapIdx; i < numQuadMaps; i++) {
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

QuadsGenerator::BufferSizes QuadsGenerator::createProxiesFromGBuffer(
        const GBuffer& gBuffer,
        const GBuffer &gBufferHighRes,
        const PerspectiveCamera &remoteCamera
    ) {
    const glm::vec2 gBufferSize = glm::vec2(gBuffer.width, gBuffer.height);

    generateInitialQuadMap(gBuffer, gBufferHighRes, gBufferSize, remoteCamera);
    simplifyQuadMaps(remoteCamera, gBufferSize);
    fillOutputQuads(gBufferSize);

    QuadsGenerator::BufferSizes bufferSizes = getBufferSizes();
    unsigned int numProxies = bufferSizes.numProxies;
    outputQuadBuffers.resize(numProxies);

    return bufferSizes;
}

#ifdef GL_CORE
unsigned int QuadsGenerator::saveQuadsToMemory(std::vector<char> &compressedData, bool compress) {
    auto bufferSizes = getBufferSizes();
    unsigned int numProxies = bufferSizes.numProxies;
    outputQuadBuffers.resize(numProxies);
    return outputQuadBuffers.saveToMemory(compressedData, compress);
}

unsigned int QuadsGenerator::saveDepthOffsetsToMemory(std::vector<char> &compressedData) {
#if !defined(__APPLE__) && !defined(__ANDROID__)
    return depthOffsets.saveToMemory(compressedData);
#else
    return 0;
#endif
}

unsigned int QuadsGenerator::saveToFile(const std::string &filename) {
    auto bufferSizes = getBufferSizes();
    unsigned int numProxies = bufferSizes.numProxies;
    outputQuadBuffers.resize(numProxies);
    return outputQuadBuffers.saveToFile(filename);
}

unsigned int QuadsGenerator::saveDepthOffsetsToFile(const std::string &filename) {
#if !defined(__APPLE__) && !defined(__ANDROID__)
    return depthOffsets.saveToFile(filename);
#else
    return 0;
#endif
}
#endif
