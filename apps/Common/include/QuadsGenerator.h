#ifndef QUADS_GENERATOR_H
#define QUADS_GENERATOR_H

#include <Cameras/PerspectiveCamera.h>
#include <Shaders/ComputeShader.h>
#include <RenderTargets/GBuffer.h>
#include <Utils/TimeUtils.h>

#define THREADS_PER_LOCALGROUP 16

#define VERTICES_IN_A_QUAD 4
#define NUM_SUB_QUADS 4

class QuadsGenerator {
public:
    struct BufferSizes {
        unsigned int numVertices;
        unsigned int numIndices;
        unsigned int numProxies;
        unsigned int numDepthOffsets;
    };

    struct Stats {
        double timeToGenerateQuadsMs = -1.0f;
        double timeToSimplifyQuadsMs = -1.0f;
        double timeToFillOutputQuadsMs = -1.0f;
    } stats;

    glm::uvec2 remoteWindowSize;
    glm::uvec2 depthBufferSize;
    glm::uvec2 maxProxySize;

    unsigned int numQuadMaps;
    unsigned int maxQuads;

    bool doOrientationCorrection = true;
    float distanceThreshold = 0.5f;
    float angleThreshold = 85.0f;
    float flatThreshold = 1.0f;
    float proxySimilarityThreshold = 0.25f;

    std::vector<glm::uvec2> quadMapSizes;

    Buffer<unsigned int> outputNormalSphericalsBuffer;
    Buffer<float> outputDepthsBuffer;
    Buffer<glm::vec2> outputUVsBuffer;
    Buffer<unsigned int> outputOffsetSizeFlattenedsBuffer;

    Texture depthOffsetsBuffer;

    QuadsGenerator(const glm::uvec2 &remoteWindowSize)
            : remoteWindowSize(remoteWindowSize)
            , depthBufferSize(4u * remoteWindowSize)
            , maxQuads(remoteWindowSize.x * remoteWindowSize.y * NUM_SUB_QUADS)
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
            , sizesBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, 1, &bufferSizes)
            , outputNormalSphericalsBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, maxQuads, nullptr)
            , outputDepthsBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, maxQuads, nullptr)
            , outputUVsBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, maxQuads, nullptr)
            , outputOffsetSizeFlattenedsBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, maxQuads, nullptr)
            , depthOffsetsBuffer({
                .width = depthBufferSize.x,
                .height = depthBufferSize.y,
                .internalFormat = GL_R16F,
                .format = GL_RED,
                .type = GL_FLOAT,
                .wrapS = GL_REPEAT,
                .wrapT = GL_REPEAT,
                .minFilter = GL_NEAREST,
                .magFilter = GL_NEAREST
            }) {
        // make sure maxProxySize is a power of 2
        maxProxySize = remoteWindowSize;
        maxProxySize.x = 1 << static_cast<int>(glm::ceil(glm::log2(static_cast<float>(maxProxySize.x))));
        maxProxySize.y = 1 << static_cast<int>(glm::ceil(glm::log2(static_cast<float>(maxProxySize.y))));
        numQuadMaps = glm::log2(static_cast<float>(glm::min(maxProxySize.x, maxProxySize.y))) + 1;

        initializeBuffers();

        // set stuff that won't change
        genQuadMapShader.bind();
        genQuadMapShader.setVec2("remoteWindowSize", remoteWindowSize);
        genQuadMapShader.setVec2("depthBufferSize", depthBufferSize);
        genQuadMapShader.setVec2("quadMapSize", quadMapSizes[0]);
        genQuadMapShader.setImageTexture(0, depthOffsetsBuffer, 0, GL_FALSE, 0, GL_READ_WRITE, depthOffsetsBuffer.internalFormat);

        simplifyQuadMapShader.bind();
        simplifyQuadMapShader.setVec2("remoteWindowSize", remoteWindowSize);
    }

    void generateInitialQuadMap(const GeometryBuffer& gBuffer, const PerspectiveCamera &remoteCamera) {
        /*
        ============================
        SECOND PASS: Generate quads from G-Buffer
        ============================
        */
        double startTime = timeutils::getTimeMicros();

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
            genQuadMapShader.setFloat("flatThreshold", flatThreshold * 1e-2f);
        }
        {
            genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, sizesBuffer);

            genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, normalSphericalsBuffers[0]);
            genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, depthsBuffers[0]);
            genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, uvsBuffers[0]);
            genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, offsetSizeFlattenedsBuffers[0]);
        }
        genQuadMapShader.dispatch((remoteWindowSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                  (remoteWindowSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
        genQuadMapShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        stats.timeToGenerateQuadsMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    }

    void simplifyQuadMaps(const PerspectiveCamera &remoteCamera) {
        /*
        ============================
        THIRD PASS: Simplify quad map
        ============================
        */
        double startTime = timeutils::getTimeMicros();

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
            simplifyQuadMapShader.setFloat("flatThreshold", flatThreshold * 1e-2f);
            simplifyQuadMapShader.setFloat("proxySimilarityThreshold", proxySimilarityThreshold);
        }
        {
            simplifyQuadMapShader.setImageTexture(0, depthOffsetsBuffer, 0, GL_FALSE, 0, GL_READ_WRITE, depthOffsetsBuffer.internalFormat);
        }
        for (int i = 1; i < numQuadMaps; i++) {
            auto& prevQuadMapSize = quadMapSizes[i-1];
            auto& prevNormalSphericalBuffer = normalSphericalsBuffers[i-1];
            auto& prevDepthsBuffer = depthsBuffers[i-1];
            auto& prevUVsBuffer = uvsBuffers[i-1];
            auto& prevOffsetsBuffer = offsetSizeFlattenedsBuffers[i-1];

            auto& currQuadMapSize = quadMapSizes[i];
            auto& currNormalSphericalBuffer = normalSphericalsBuffers[i];
            auto& currDepthsBuffer = depthsBuffers[i];
            auto& currUVsBuffer = uvsBuffers[i];
            auto& currOffsetsBuffer = offsetSizeFlattenedsBuffers[i];

            {
                simplifyQuadMapShader.setVec2("inputQuadMapSize", prevQuadMapSize);
                simplifyQuadMapShader.setVec2("outputQuadMapSize", currQuadMapSize);
            }
            {
                simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, prevNormalSphericalBuffer);
                simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, prevDepthsBuffer);
                simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, prevUVsBuffer);
                simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, prevOffsetsBuffer);

                simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, currNormalSphericalBuffer);
                simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, currDepthsBuffer);
                simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, currUVsBuffer);
                simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 7, currOffsetsBuffer);
            }
            simplifyQuadMapShader.dispatch((currQuadMapSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                           (currQuadMapSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
            simplifyQuadMapShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
        simplifyQuadMapShader.memoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        stats.timeToSimplifyQuadsMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    }

    void fillOutputQuads() {
        /*
        ============================
        FOURTH PASS: Fill output quads buffer
        ============================
        */
        double startTime = timeutils::getTimeMicros();

        fillOutputQuadsShader.bind();
        for (int i = 0; i < numQuadMaps; i++) {
            auto& currNormalSphericalBuffer = normalSphericalsBuffers[i];
            auto& currDepthsBuffer = depthsBuffers[i];
            auto& currUVsBuffer = uvsBuffers[i];
            auto& currOffsetsBuffer = offsetSizeFlattenedsBuffers[i];

            auto& currQuadMapSize = quadMapSizes[i];

            {
                fillOutputQuadsShader.setVec2("quadMapSize", currQuadMapSize);
            }
            {
                fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, sizesBuffer);

                fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, currNormalSphericalBuffer);
                fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, currDepthsBuffer);
                fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, currUVsBuffer);
                fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, currOffsetsBuffer);

                fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, outputNormalSphericalsBuffer);
                fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, outputDepthsBuffer);
                fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 7, outputUVsBuffer);
                fillOutputQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 8, outputOffsetSizeFlattenedsBuffer);
            }
            fillOutputQuadsShader.dispatch((currQuadMapSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                           (currQuadMapSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
        }
        fillOutputQuadsShader.memoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

        stats.timeToFillOutputQuadsMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    }

    void createProxiesFromGBuffer(const GeometryBuffer& gBuffer, const PerspectiveCamera &remoteCamera){
        generateInitialQuadMap(gBuffer, remoteCamera);
        simplifyQuadMaps(remoteCamera);
        fillOutputQuads();
    }

    unsigned int getProxies(char* proxiesData) {
        auto bufferSizes = getBufferSizes();

        unsigned int offset = 0;
        unsigned int numProxies = bufferSizes.numProxies;

        // write number of proxies
        memcpy(proxiesData, &numProxies, sizeof(unsigned int));
        offset += sizeof(unsigned int);

        // write normals
        std::vector<unsigned int> normalSphericals(numProxies);
        outputNormalSphericalsBuffer.bind();
        outputNormalSphericalsBuffer.getSubData(0, numProxies, normalSphericals.data());
        memcpy(proxiesData + offset, normalSphericals.data(), numProxies * sizeof(unsigned int));
        offset += numProxies * sizeof(unsigned int);

        // write depths
        std::vector<float> depths(numProxies);
        outputDepthsBuffer.bind();
        outputDepthsBuffer.getSubData(0, numProxies, depths.data());
        memcpy(proxiesData + offset, depths.data(), numProxies * sizeof(float));
        offset += numProxies * sizeof(float);

        // write uvs
        std::vector<glm::vec2> uvs(numProxies);
        outputUVsBuffer.bind();
        outputUVsBuffer.getSubData(0, numProxies, uvs.data());
        memcpy(proxiesData + offset, uvs.data(), numProxies * sizeof(glm::vec2));
        offset += numProxies * sizeof(glm::vec2);

        // write offsetSizeFlatteneds
        std::vector<unsigned int> offsetSizeFlatteneds(numProxies);
        outputOffsetSizeFlattenedsBuffer.bind();
        outputOffsetSizeFlattenedsBuffer.getSubData(0, numProxies, offsetSizeFlatteneds.data());
        memcpy(proxiesData + offset, offsetSizeFlatteneds.data(), bufferSizes.numProxies * sizeof(unsigned int));
        offset += numProxies * sizeof(unsigned int);

        return offset;
    }

    unsigned int saveProxies(const std::string &filename) {
        char* data = new char[sizeof(unsigned int) +
                                maxQuads * (sizeof(unsigned int) + sizeof(float) + sizeof(glm::vec2) + sizeof(unsigned int))];
        unsigned int payloadSize = getProxies(data);

        std::ofstream quadsFile(filename, std::ios::binary);
        quadsFile.write(data, payloadSize);
        quadsFile.close();

        delete[] data;
        return payloadSize;
    }

    Buffer<BufferSizes>& getSizesBuffer() {
        return sizesBuffer;
    }

    BufferSizes getBufferSizes() {
        sizesBuffer.bind();
        sizesBuffer.getSubData(0, 1, &bufferSizes);
        return bufferSizes;
    }

private:
    BufferSizes bufferSizes = { 0 };
    Buffer<BufferSizes> sizesBuffer;

    std::vector<Buffer<unsigned int>> normalSphericalsBuffers;
    std::vector<Buffer<float>> depthsBuffers;
    std::vector<Buffer<glm::vec2>> uvsBuffers;
    std::vector<Buffer<unsigned int>> offsetSizeFlattenedsBuffers;

    ComputeShader genQuadMapShader;
    ComputeShader simplifyQuadMapShader;
    ComputeShader fillOutputQuadsShader;

    void initializeBuffers() {
        quadMapSizes.reserve(numQuadMaps);

        normalSphericalsBuffers.reserve(numQuadMaps);
        depthsBuffers.reserve(numQuadMaps);
        uvsBuffers.reserve(numQuadMaps);
        offsetSizeFlattenedsBuffers.reserve(numQuadMaps);

        glm::vec2 currQuadMapSize = maxProxySize;
        for (int i = 0; i < numQuadMaps; i++) {
            normalSphericalsBuffers.emplace_back(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, currQuadMapSize.x * currQuadMapSize.y, nullptr);
            depthsBuffers.emplace_back(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, currQuadMapSize.x * currQuadMapSize.y, nullptr);
            uvsBuffers.emplace_back(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, currQuadMapSize.x * currQuadMapSize.y, nullptr);
            offsetSizeFlattenedsBuffers.emplace_back(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, currQuadMapSize.x * currQuadMapSize.y, nullptr);

            quadMapSizes[i] = currQuadMapSize;
            currQuadMapSize /= 2;
        }
    }
};

#endif // QUADS_GENERATOR_H
