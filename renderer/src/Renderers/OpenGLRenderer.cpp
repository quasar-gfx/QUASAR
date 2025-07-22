#include <Renderers/OpenGLRenderer.h>
#include <Primitives/Sphere.h>
#include <Materials/UnlitMaterial.h>

using namespace quasar;

#ifndef __APPLE__
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
    , skyboxShader({
        .vertexCodeData = SHADER_BUILTIN_SKYBOX_VERT,
        .vertexCodeSize = SHADER_BUILTIN_SKYBOX_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_SKYBOX_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_SKYBOX_FRAG_len
    })
    , outputFsQuad()
{
#ifdef GL_CORE
    // Enable setting vertex size for point clouds
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
#endif

#ifndef __APPLE__
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(glDebugCallback, nullptr);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_FALSE);
    glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_ERROR, GL_DONT_CARE, 0, nullptr, GL_TRUE);
#endif

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

RenderStats OpenGLRenderer::updateDirLightShadow(Scene& scene, const Camera& camera) {
    RenderStats stats;
    if (scene.directionalLight == nullptr) {
        return stats;
    }

    auto& shadowMapRT = scene.directionalLight->shadowMapRenderTarget;

    shadowMapRT.bind();
    shadowMapRT.setViewport(0, 0, shadowMapRT.width, shadowMapRT.height);
    glClear(GL_DEPTH_BUFFER_BIT);

    for (auto& child : scene.rootNode.children) {
        stats += drawNode(scene, camera, child, glm::mat4(1.0f), false, &scene.directionalLight->shadowMapMaterial);
    }

    shadowMapRT.unbind();

    return stats;
}

RenderStats OpenGLRenderer::updatePointLightShadows(Scene& scene, const Camera& camera) {
    RenderStats stats;

    for (int i = 0; i < scene.pointLights.size(); i++) {
        auto pointLight = scene.pointLights[i];
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

        for (auto& child : scene.rootNode.children) {
            stats += drawNode(scene, camera, child, glm::mat4(1.0f), pointLight, &pointLight->shadowMapMaterial);
        }

        shadowMapRT.unbind();
    }

    return stats;
}

RenderStats OpenGLRenderer::drawSceneImpl(Scene& scene, const Camera& camera, uint32_t clearMask) {
    if (clearMask != 0) {
        glClearColor(scene.backgroundColor.x, scene.backgroundColor.y, scene.backgroundColor.z, scene.backgroundColor.w);
        glClear(clearMask);
    }

    RenderStats stats;
    for (auto& child : scene.rootNode.children) {
        stats += drawNode(scene, camera, child, glm::mat4(1.0f), true);
    }

    return stats;
}

RenderStats OpenGLRenderer::drawScene(Scene& scene, const Camera& camera, uint32_t clearMask) {
    beginRendering();
    RenderStats stats = drawSceneImpl(scene, camera, clearMask);
    endRendering();
    return stats;
}

RenderStats OpenGLRenderer::drawLightsImpl(Scene& scene, const Camera& camera) {
    // Dont clear color or depth bit here, since we want this to draw over

    RenderStats stats;
    for (auto& pointLight : scene.pointLights) {
        // Only draw if debug is set
        if (pointLight->debug) {
            auto material = new UnlitMaterial({ .baseColor = glm::vec4(pointLight->color, 1.0) });
            Sphere light = Sphere({
                .material = material
            }, 32, 32);
            Node nodeLight = Node(&light);
            nodeLight.setPosition(pointLight->position);
            nodeLight.setScale(glm::vec3(0.1));
            stats += drawNode(scene, camera, &nodeLight, glm::mat4(1.0f), false);

            Sphere radius = Sphere({
                .material = material,
            }, 32, 32);
            Node nodeRadius = Node(&radius);
            nodeRadius.wireframe = true;
            nodeRadius.setPosition(pointLight->position);
            nodeRadius.setScale(glm::vec3(pointLight->getLightRadius()));
            stats += drawNode(scene, camera, &nodeRadius, glm::mat4(1.0f), false);
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

RenderStats OpenGLRenderer::drawSkyBoxImpl(Scene& scene, const Camera& camera) {
    // Dont clear color or depth bit here, since we want this to draw over

    RenderStats stats;

    if (scene.envCubeMap == nullptr) {
        return stats;
    }

    auto& skybox = *scene.envCubeMap;

    skyboxShader.bind();
    skyboxShader.setTexture("environmentMap", skybox, 0);

    // Disable writing to the depth buffer
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_FALSE);

    if (scene.envCubeMap != nullptr) {
        stats = scene.envCubeMap->draw(skyboxShader, camera);
    }

    // Restore depth func
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);

    return stats;
}

RenderStats OpenGLRenderer::drawSkyBox(Scene& scene, const Camera& camera) {
    beginRendering();
    RenderStats stats = drawSkyBoxImpl(scene, camera);
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

RenderStats OpenGLRenderer::drawNode(Scene& scene, const Camera& camera, Node* node, const glm::mat4& parentTransform,
                                     bool frustumCull, const Material* overrideMaterial, const Texture* prevIDMap) {
    const glm::mat4& model = parentTransform * node->getTransformParentFromLocal() * node->getTransformAnimation();

    auto* materialToUse = overrideMaterial != nullptr ? overrideMaterial : node->overrideMaterial;

    RenderStats stats;
    if (node->entity != nullptr) {
        if (node->visible) {
            node->entity->bindMaterial(scene, model, materialToUse, prevIDMap);
            bool doFrustumCull = frustumCull && node->frustumCulled;

#ifdef GL_CORE
            // Set polygon mode to wireframe if needed
            if (node->wireframe || node->primativeType == GL_LINES) {
                glEnable(GL_POLYGON_OFFSET_LINE); // to avoid z-fighting
                glPolygonOffset(-1.0, -1.0); // adjust depth
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                glLineWidth(node->wireframeLineWidth);
            }
            if (node->primativeType == GL_POINTS) {
                glEnable(GL_POLYGON_OFFSET_POINT); // to avoid z-fighting
                glPolygonOffset(-1.0, -1.0); // adjust depth
                glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
                glPointSize(node->pointSize);
            }
#else
            if (node->primativeType == GL_LINES) {
                glLineWidth(node->wireframeLineWidth);
                glDepthRangef(0.0f, 0.999f);
            }
#endif

            stats += node->entity->draw(node->primativeType, camera, model, doFrustumCull, materialToUse);

#ifdef GL_CORE
            // Restore polygon mode
            if (node->wireframe) {
                glDisable(GL_POLYGON_OFFSET_LINE);
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            }
            if (node->primativeType == GL_POINTS) {
                glDisable(GL_POLYGON_OFFSET_POINT);
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            }
#else
            if (node->primativeType == GL_LINES) {
                glDepthRangef(0.0f, 1.0f);
            }
#endif
        }
    }

    for (auto& child : node->children) {
        stats += drawNode(scene, camera, child, model, materialToUse);
    }

    return stats;
}

RenderStats OpenGLRenderer::drawNode(Scene& scene, const Camera& camera, Node* node, const glm::mat4& parentTransform,
                                     const PointLight* pointLight, const Material* overrideMaterial) {
    const glm::mat4& model = parentTransform * node->getTransformParentFromLocal() * node->getTransformAnimation();

    RenderStats stats;
    if (node->entity != nullptr) {
        if (node->visible) {
            // Don't have to bind to scene and camera here, since we are only drawing shadows
            stats += node->entity->draw(node->primativeType, camera, model, pointLight->boundingSphere, overrideMaterial);
        }
    }

    for (auto& child : node->children) {
        stats += drawNode(scene, camera, child, model, pointLight, overrideMaterial);
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

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    screenShader.bind();
    RenderStats stats = outputFsQuad.draw();

    if (overrideRenderTarget != nullptr) {
        overrideRenderTarget->unbind();
    }

    return stats;
}

RenderStats OpenGLRenderer::drawToRenderTarget(const Shader& screenShader, const RenderTargetBase& renderTarget) {
    return drawToScreen(screenShader, &renderTarget);
}
