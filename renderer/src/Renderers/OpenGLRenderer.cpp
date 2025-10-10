#include <algorithm>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Primitives/Sphere.h>
#include <Renderers/OpenGLRenderer.h>
#include <Materials/UnlitMaterial.h>
#include <Materials/LitMaterial.h>
#include <Primitives/Mesh.h>

using namespace quasar;

#ifndef PLATFORM_APPLE
void glDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const void *userParam) {
    std::cout << "OpenGL Debug message (" << id << "): " << message << std::endl;

    switch (source) {
    case GL_DEBUG_SOURCE_API:
        std::cout << "Source: API";
        break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
        std::cout << "Source: Window System";
        break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER:
        std::cout << "Source: Shader Compiler";
        break;
    case GL_DEBUG_SOURCE_THIRD_PARTY:
        std::cout << "Source: Third Party";
        break;
    case GL_DEBUG_SOURCE_APPLICATION:
        std::cout << "Source: Application";
        break;
    case GL_DEBUG_SOURCE_OTHER:
        std::cout << "Source: Other";
        break;
    }
    std::cout << std::endl;

    switch (type) {
    case GL_DEBUG_TYPE_ERROR:
        std::cout << "Type: Error";
        break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        std::cout << "Type: Deprecated Behaviour";
        break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        std::cout << "Type: Undefined Behaviour";
        break;
    case GL_DEBUG_TYPE_PORTABILITY:
        std::cout << "Type: Portability";
        break;
    case GL_DEBUG_TYPE_PERFORMANCE:
        std::cout << "Type: Performance";
        break;
    case GL_DEBUG_TYPE_MARKER:
        std::cout << "Type: Marker";
        break;
    case GL_DEBUG_TYPE_PUSH_GROUP:
        std::cout << "Type: Push Group";
        break;
    case GL_DEBUG_TYPE_POP_GROUP:
        std::cout << "Type: Pop Group";
        break;
    case GL_DEBUG_TYPE_OTHER:
        std::cout << "Type: Other";
        break;
    }
    std::cout << std::endl;

    switch (severity) {
    case GL_DEBUG_SEVERITY_HIGH:
        std::cout << "Severity: high";
        break;
    case GL_DEBUG_SEVERITY_MEDIUM:
        std::cout << "Severity: medium";
        break;
    case GL_DEBUG_SEVERITY_LOW:
        std::cout << "Severity: low";
        break;
    case GL_DEBUG_SEVERITY_NOTIFICATION:
        std::cout << "Severity: notification";
        break;
    }
    std::cout << std::endl;
    std::cout << std::endl;

    if (type == GL_DEBUG_TYPE_ERROR)
        exit(EXIT_FAILURE);
}
#endif

OpenGLRenderer::OpenGLRenderer(const Config& config)
    : width(config.width), height(config.height)
    , windowWidth(config.width), windowHeight(config.height)
    , sortTransparent(config.sortTransparent)
{
#ifdef GL_CORE
    // Enable setting vertex size for point clouds
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
#endif

#ifndef PLATFORM_APPLE
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(glDebugCallback, nullptr);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_FALSE);
    glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_ERROR, GL_DONT_CARE, 0, nullptr, GL_TRUE);
#endif

    createResources();

    setGraphicsPipeline(config.pipeline);
    pipeline.apply();
}

void OpenGLRenderer::resize(uint width, uint height) {
    this->width = width;
    this->height = height;
}

void OpenGLRenderer::setWindowSize(uint width, uint height) {
    windowWidth = width;
    windowHeight = height;
}

void OpenGLRenderer::createResources() {
    skyboxShader = std::make_shared<Shader>(ShaderDataCreateParams({
        .vertexCodeData = SHADER_BUILTIN_SKYBOX_VERT,
        .vertexCodeSize = SHADER_BUILTIN_SKYBOX_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_SKYBOX_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_SKYBOX_FRAG_len,
    }));

    pointLightsUBO = std::make_shared<Buffer>(BufferCreateParams({
        .target = GL_UNIFORM_BUFFER,
        .dataSize = sizeof(Scene::GPUPointLightBlock),
        .numElems = 1,
        .usage = GL_DYNAMIC_DRAW,
    }));
    outputFsQuad = std::make_shared<FullScreenQuad>();
}

RenderStats OpenGLRenderer::updateDirLightShadow(Scene& scene, const Camera& camera) {
    RenderStats stats;
    if (scene.directionalLight == nullptr) {
        return stats;
    }

    auto& shadowMapRT = scene.directionalLight->shadowMapRenderTarget;

    shadowMapRT.bind();
    shadowMapRT.setViewport(0, 0, shadowMapRT.width, shadowMapRT.height);
    glClear(GL_DEPTH_BUFFER_BIT);

    for (auto* child : scene.children) {
        stats += drawNodeImmediate(scene, camera, child, glm::mat4(1.0f), false, &scene.directionalLight->shadowMapMaterial);
    }

    shadowMapRT.unbind();

    return stats;
}

RenderStats OpenGLRenderer::updatePointLightShadows(Scene& scene, const Camera& camera) {
    RenderStats stats;

    for (int i = 0; i < scene.pointLights.size(); i++) {
        auto& pointLight = scene.pointLights[i];
        if (pointLight->intensity == 0)
            continue;

        auto& shadowMapRT = pointLight->shadowMapRenderTarget;

        shadowMapRT.bind();
        shadowMapRT.setViewport(0, 0, shadowMapRT.width, shadowMapRT.height);
        glClear(GL_DEPTH_BUFFER_BIT);

        pointLight->shadowMapMaterial.bind();
        pointLight->shadowMapMaterial.shader->setVec3("lightPos", pointLight->position);
        pointLight->shadowMapMaterial.shader->setFloat("farPlane", pointLight->shadowFar);

        glm::mat4 shadowProj = pointLight->shadowProjectionMat;
        for (int face = 0; face < NUM_CUBEMAP_FACES; face++) {
            pointLight->shadowMapMaterial.shader->setMat4("shadowMatrices[" + std::to_string(face) + "]", shadowProj * pointLight->lookAtPerFace[face]);
        }

        for (auto* child : scene.children) {
            stats += drawNodeImmediate(scene, camera, child, glm::mat4(1.0f), *pointLight, &pointLight->shadowMapMaterial);
        }

        shadowMapRT.unbind();
    }

    return stats;
}

void OpenGLRenderer::fillRenderLists(Scene& scene, const Camera& camera) {
    renderLists.clear();
    for (auto* child : scene.children) {
        gatherNodes(scene, camera, child, glm::mat4(1.0f), true);
    }
}

RenderStats OpenGLRenderer::drawSceneImpl(Scene& scene, const Camera& camera, uint32_t clearMask) {
    RenderStats stats;
    if (sortTransparent) {
        fillRenderLists(scene, camera);
        stats += drawOpaque(scene, camera);
        stats += drawTransparent(scene, camera);
    }
    else {
        for (auto* child : scene.children) {
            stats += drawNodeImmediate(scene, camera, child, glm::mat4(1.0f), true);
        }
    }
    return stats;
}

RenderStats OpenGLRenderer::drawScene(Scene& scene, const Camera& camera, uint32_t clearMask) {
    beginRendering();
    if (clearMask != 0) {
        glClearColor(scene.backgroundColor.x, scene.backgroundColor.y, scene.backgroundColor.z, scene.backgroundColor.w);
        glClear(clearMask);
    }

    RenderStats stats = drawSceneImpl(scene, camera, clearMask);
    endRendering();
    return stats;
}

RenderStats OpenGLRenderer::drawLightsImpl(Scene& scene, const Camera& camera) {
    // Don't clear color or depth bit here, since we want this to draw over

    RenderStats stats;
    for (auto& pointLight : scene.pointLights) {
        // Only draw if debug is set
        if (pointLight->debug) {
            auto material = std::make_shared<UnlitMaterial>(UnlitMaterial({ .baseColor = glm::vec4(pointLight->color, 1.0) }));
            Sphere light({
                .material = material.get(),
            }, 32, 32);
            Node nodeLight(&light);
            nodeLight.setPosition(pointLight->position);
            nodeLight.setScale(glm::vec3(0.1));

            Sphere radius({
                .material = material.get(),
            }, 32, 32);
            Node nodeRadius(&radius);
            nodeRadius.wireframe = true;
#ifdef GL_CORE
            nodeRadius.primitiveType = GL_LINES;
#endif
            nodeRadius.setPosition(pointLight->position);
            nodeRadius.setScale(glm::vec3(pointLight->getLightRadius()));

            stats += drawNodeImmediate(scene, camera, &nodeLight, glm::mat4(1.0f), false);
            stats += drawNodeImmediate(scene, camera, &nodeRadius, glm::mat4(1.0f), false);
        }
    }

    return stats;
}

RenderStats OpenGLRenderer::drawLights(Scene& scene, const Camera& camera) {
    beginRendering();
    RenderStats stats = drawLightsImpl(scene, camera);
    endRendering();
    return stats;
}

RenderStats OpenGLRenderer::drawSkyBoxImpl(Scene& scene, const Camera& camera, uint32_t clearMask) {
    RenderStats stats;

    if (scene.envCubeMap == nullptr) {
        return stats;
    }

    // Disable writing to the depth buffer
    pipeline.depthState.depthFunc = GL_LEQUAL;
    pipeline.writeMaskState.depth = false;
    pipeline.apply();

    skyboxShader->bind();
    skyboxShader->setTexture("environmentMap", *scene.envCubeMap, 0);
    stats = scene.envCubeMap->draw(*skyboxShader, camera);

    // Restore depth state
    pipeline.depthState.depthFunc = GL_LESS;
    pipeline.writeMaskState.depth = true;
    pipeline.apply();

    return stats;
}

RenderStats OpenGLRenderer::drawSkyBox(Scene& scene, const Camera& camera, uint32_t clearMask) {
    beginRendering();
    if (clearMask != 0) {
        glClearColor(scene.backgroundColor.x, scene.backgroundColor.y, scene.backgroundColor.z, scene.backgroundColor.w);
        glClear(clearMask);
    }

    RenderStats stats = drawSkyBoxImpl(scene, camera, clearMask);
    endRendering();
    return stats;
}

RenderStats OpenGLRenderer::drawObjectsNoLighting(Scene& scene, const Camera& camera, uint32_t clearMask) {
    pipeline.apply();

    RenderStats stats;

    // Draw all objects in the scene
    stats += drawScene(scene, camera, clearMask);

    // Draw skybox
    stats += drawSkyBox(scene, camera);

    return stats;
}

RenderStats OpenGLRenderer::drawObjects(Scene& scene, const Camera& camera, uint32_t clearMask) {
    pipeline.apply();

    RenderStats stats;

    // Update shadows
    updateDirLightShadow(scene, camera);
    updatePointLightShadows(scene, camera);

    // Draw all objects in the scene
    stats += drawScene(scene, camera, clearMask);

    // Draw lights for debugging
    stats += drawLights(scene, camera);

    // Draw skybox
    stats += drawSkyBox(scene, camera);

    return stats;
}

void OpenGLRenderer::gatherNodes(Scene& scene, const Camera& camera, const Node* node, const glm::mat4& parentTransform,
                                 bool frustumCull, const Material* overrideMaterial, const Texture* prevIDMap) {
    const glm::mat4 model = parentTransform * node->getTransformParentFromLocal() * node->getTransformAnimation();
    const Material* materialToUse = overrideMaterial != nullptr ? overrideMaterial : node->overrideMaterial;

    if (node->visible) {
        if (!node->entities.empty()) {
            bool isTransparent = (materialToUse != nullptr) && materialToUse->isTransparent();
            for (auto* entity : node->entities) {
                isTransparent = isTransparent || (entity->getMaterial() != nullptr && entity->getMaterial()->isTransparent());
            }

            RenderItem item{ node, model, materialToUse, frustumCull };
            if (!isTransparent) {
                renderLists.opaque.push_back(item);
            }
            else {
                renderLists.transparent.push_back(item);
            }
        }

        for (auto* child : node->children) {
            gatherNodes(scene, camera, child, model, frustumCull, materialToUse, prevIDMap);
        }
    }
}

RenderStats OpenGLRenderer::drawNode(Scene& scene, const Camera& camera, const Node* node, const glm::mat4& model,
                                     bool frustumCull, const Material* overrideMaterial, const Texture* prevIDMap) {
    RenderStats stats;

    if (node->visible) {
        for (auto* entity : node->entities) {
            entity->bindMaterial(scene, *pointLightsUBO, overrideMaterial, prevIDMap);

#ifdef GL_CORE
            // Set polygon mode to wireframe if needed
            if (node->wireframe || node->primitiveType == GL_LINES) {
                glEnable(GL_POLYGON_OFFSET_LINE); // To avoid z-fighting
                glPolygonOffset(-1.0, -1.0); // Adjust depth
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                glLineWidth(node->wireframeLineWidth);
            }
            if (node->primitiveType == GL_POINTS) {
                glEnable(GL_POLYGON_OFFSET_POINT); // To avoid z-fighting
                glPolygonOffset(-1.0, -1.0); // Adjust depth
                glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
                glPointSize(node->pointSize);
            }
#else
            if (node->wireframe || node->primitiveType == GL_LINES) {
                glLineWidth(node->wireframeLineWidth);
                glDepthRangef(0.0f, 0.999f);
            }
#endif

        stats += entity->draw(node->primitiveType, camera, model, frustumCull && node->frustumCulled, overrideMaterial);

#ifdef GL_CORE
            // Restore polygon mode
            if (node->wireframe || node->primitiveType == GL_LINES) {
                glDisable(GL_POLYGON_OFFSET_LINE);
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            }
            if (node->primitiveType == GL_POINTS) {
                glDisable(GL_POLYGON_OFFSET_POINT);
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            }
#else
            if (node->wireframe || node->primitiveType == GL_LINES) {
                glDepthRangef(0.0f, 1.0f);
            }
#endif
        }
    }

    return stats;
}

RenderStats OpenGLRenderer::drawNodeImmediate(Scene& scene, const Camera& camera, const Node* node, const glm::mat4& parentTransform,
                                              bool frustumCull, const Material* overrideMaterial, const Texture* prevIDMap) {
    const glm::mat4 model = parentTransform * node->getTransformParentFromLocal() * node->getTransformAnimation();
    const Material* materialToUse = overrideMaterial != nullptr ? overrideMaterial : node->overrideMaterial;

    RenderStats stats;

    if (node->visible) {
        stats += drawNode(scene, camera, node, model, frustumCull && node->frustumCulled, materialToUse, prevIDMap);

        for (auto* child : node->children) {
            stats += drawNodeImmediate(scene, camera, child, model, frustumCull, materialToUse, prevIDMap);
        }
    }

    return stats;
}

RenderStats OpenGLRenderer::drawNodeImmediate(Scene& scene, const Camera& camera, const Node* node, const glm::mat4& parentTransform,
                                              const PointLight& pointLight, const Material* overrideMaterial) {
    const glm::mat4& model = parentTransform * node->getTransformParentFromLocal() * node->getTransformAnimation();

    RenderStats stats;

    if (node->visible) {
        for (auto* entity : node->entities) {
            // Don't have to bind to scene and camera here, since we are only drawing shadows
            stats += entity->draw(node->primitiveType, camera, model, pointLight.boundingSphere, overrideMaterial);
        }

        for (auto* child : node->children) {
            stats += drawNodeImmediate(scene, camera, child, model, pointLight, overrideMaterial);
        }
    }

    return stats;
}

RenderStats OpenGLRenderer::drawItem(Scene& scene, const Camera& camera, const RenderItem& item) {
    RenderStats stats;

    const Node* node = item.node;
    const glm::mat4& model = item.model;
    const Material* materialToUse = item.materialOverride != nullptr ? item.materialOverride : node->overrideMaterial;
    stats += drawNode(scene, camera, node, model, item.frustumCull, materialToUse, nullptr);

    return stats;
};

RenderStats OpenGLRenderer::drawOpaque(Scene& scene, const Camera& camera) {
    pipeline.apply();

    RenderStats stats;

    // Draw nodes normally (no sorting needed)
    for (const auto& item : renderLists.opaque) {
        stats += drawItem(scene, camera, item);
    }

    return stats;
}

RenderStats OpenGLRenderer::drawTransparent(Scene& scene, const Camera& camera) {
    pipeline.apply();

    RenderStats stats;

    // Sort back-to-front by AABB distance to camera
    std::vector<RenderItem>& sorted = renderLists.transparent;
    std::sort(sorted.begin(), sorted.end(), [&](const RenderItem& a, const RenderItem& b) {
        // Get AABB centers in world space for comparison
        glm::vec3 aabbCenterA = glm::vec3(0.0f);
        glm::vec3 aabbCenterB = glm::vec3(0.0f);

        // Calculate AABB center for item A
        if (!a.node->entities.empty()) {
            // Transform AABB center to world space
            glm::vec3 localCenter = a.node->entities[0]->aabb.getCenter();
            aabbCenterA = glm::vec3(a.model * glm::vec4(localCenter, 1.0f));
        }
        else {
            // Fallback to node world position
            aabbCenterA = glm::vec3(a.model[3]);
        }

        // Calculate AABB center for item B
        if (!b.node->entities.empty()) {
            // Transform AABB center to world space
            glm::vec3 localCenter = b.node->entities[0]->aabb.getCenter();
            aabbCenterB = glm::vec3(b.model * glm::vec4(localCenter, 1.0f));
        }
        else {
            // Fallback to node world position
            aabbCenterB = glm::vec3(b.model[3]);
        }

        // Compare squared distance to camera
        glm::vec3 daVec = camera.getPosition() - aabbCenterA;
        glm::vec3 dbVec = camera.getPosition() - aabbCenterB;
        float da = glm::dot(daVec, daVec);
        float db = glm::dot(dbVec, dbVec);
        return da > db; // Sort back-to-front (farther objects first)
    });

    for (const auto& item : sorted) {
        stats += drawItem(scene, camera, item);
    }

    return stats;
}

RenderStats OpenGLRenderer::drawToScreen(const Shader& screenShader, const RenderTargetBase* overrideRenderTarget) {
    pipeline.apply();

    if (overrideRenderTarget != nullptr) {
        overrideRenderTarget->bind();
    }
    else {
        // Screen buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, windowWidth, windowHeight);
    }

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    screenShader.bind();
    RenderStats stats = outputFsQuad->draw();

    if (overrideRenderTarget != nullptr) {
        overrideRenderTarget->unbind();
    }

    return stats;
}

RenderStats OpenGLRenderer::drawToRenderTarget(const Shader& screenShader, const RenderTargetBase& renderTarget) {
    return drawToScreen(screenShader, &renderTarget);
}
