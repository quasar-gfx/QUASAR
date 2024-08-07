#include <OpenGLRenderer.h>
#include <Primatives/Sphere.h>
#include <Materials/UnlitMaterial.h>

#ifndef __APPLE__
void APIENTRY glDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const void *userParam) {
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

OpenGLRenderer::OpenGLRenderer(unsigned int width, unsigned int height)
        : width(width), height(height)
        , gBuffer({ .width = width, .height = height })
        , skyboxShader({
            .vertexCodeData = SHADER_SKYBOX_VERT,
            .vertexCodeSize = SHADER_SKYBOX_VERT_len,
            .fragmentCodeData = SHADER_SKYBOX_FRAG,
            .fragmentCodeSize = SHADER_SKYBOX_FRAG_len
        })
        , outputFsQuad() {
    // enable setting vertex size for point clouds
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

#ifndef __APPLE__
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(glDebugCallback, nullptr);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_FALSE);
    glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_ERROR, GL_DONT_CARE, 0, nullptr, GL_TRUE);
#endif
}

RenderStats OpenGLRenderer::updateDirLightShadow(const Scene &scene, const Camera &camera) {
    RenderStats stats;
    if (scene.directionalLight == nullptr) {
        return stats;
    }

    scene.directionalLight->shadowMapRenderTarget.bind();
    glClear(GL_DEPTH_BUFFER_BIT);

    for (auto& child : scene.children) {
        stats += drawNode(scene, camera, child, glm::mat4(1.0f), false, &scene.directionalLight->shadowMapMaterial);
    }

    scene.directionalLight->shadowMapRenderTarget.unbind();

    return stats;
}

RenderStats OpenGLRenderer::updatePointLightShadows(const Scene &scene, const Camera &camera) {
    RenderStats stats;

    for (int i = 0; i < scene.pointLights.size(); i++) {
        auto pointLight = scene.pointLights[i];
        if (pointLight->intensity == 0)
            continue;

        pointLight->shadowMapRenderTarget.bind();
        glClear(GL_DEPTH_BUFFER_BIT);

        pointLight->shadowMapMaterial.bind();
        pointLight->shadowMapMaterial.shader->setVec3("lightPos", pointLight->position);
        pointLight->shadowMapMaterial.shader->setFloat("farPlane", pointLight->shadowFar);

        glm::mat4 shadowProj = pointLight->shadowProjectionMat;
        for (int face = 0; face < NUM_CUBEMAP_FACES; face++) {
            pointLight->shadowMapMaterial.shader->setMat4("shadowMatrices[" + std::to_string(face) + "]", shadowProj * pointLight->lookAtPerFace[face]);
        }
        pointLight->shadowMapMaterial.unbind();

        for (auto& child : scene.children) {
            stats += drawNode(scene, camera, child, glm::mat4(1.0f), pointLight, &pointLight->shadowMapMaterial);
        }

        pointLight->shadowMapRenderTarget.unbind();
    }

    return stats;
}

RenderStats OpenGLRenderer::drawSkyBox(const Scene &scene, const Camera &camera) {
    RenderStats stats;

    if (scene.envCubeMap == nullptr) {
        return stats;
    }

    gBuffer.bind();
    // dont clear color or depth bit here, since we want this to draw over

    skyboxShader.bind();
    skyboxShader.setInt("environmentMap", 0);
    skyboxShader.setMat4("view", camera.getViewMatrix());
    skyboxShader.setMat4("projection", camera.getProjectionMatrix());

    if (scene.envCubeMap != nullptr) {
        stats = scene.envCubeMap->draw(skyboxShader, camera);
    }

    skyboxShader.unbind();

    gBuffer.unbind();

    return stats;
}

RenderStats OpenGLRenderer::drawObjects(const Scene &scene, const Camera &camera) {
    pipeline.apply();

    RenderStats stats;

    // update shadows
    updateDirLightShadow(scene, camera);
    updatePointLightShadows(scene, camera);

    // bind to gBuffer and draw scene as we normally would to color texture
    gBuffer.bind();
    glClearColor(scene.backgroundColor.x, scene.backgroundColor.y, scene.backgroundColor.z, scene.backgroundColor.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // draw point lights, if debug is set
    for (auto& pointLight : scene.pointLights) {
        if (pointLight->debug) {
            auto material = new UnlitMaterial({ .baseColor = glm::vec4(pointLight->color, 1.0) });
            Sphere light = Sphere({
                .material = material,
                .wireframe = false
            }, 32, 32);
            Node nodeLight = Node(&light);
            nodeLight.setPosition(pointLight->position);
            nodeLight.setScale(glm::vec3(0.1));
            stats += drawNode(scene, camera, &nodeLight, glm::mat4(1.0f), false);

            Sphere radius = Sphere({
                .material = material,
                .wireframe = true
            }, 32, 32);
            Node nodeRadius = Node(&radius);
            nodeRadius.setPosition(pointLight->position);
            nodeRadius.setScale(glm::vec3(pointLight->getLightRadius()));
            stats += drawNode(scene, camera, &nodeRadius, glm::mat4(1.0f), false);
        }
    }

    // draw scene
    for (auto& child : scene.children) {
        stats += drawNode(scene, camera, child, glm::mat4(1.0f));
    }

    // draw skybox
    stats += drawSkyBox(scene, camera);

    // now bind back to default gBuffer and draw a quad plane with the attached gBuffer color texture
    gBuffer.unbind();

    return stats;
}

RenderStats OpenGLRenderer::drawNode(const Scene &scene, const Camera &camera, Node* node, const glm::mat4 &parentTransform,
                                     bool frustumCull, const Material* overrideMaterial) {
    const glm::mat4 &model = parentTransform * node->getTransformParentFromLocal();

    RenderStats stats;
    if (node->entity != nullptr) {
        if (node->visible) {
            node->entity->bindMaterial(scene, model, overrideMaterial);
            bool doFrustumCull = frustumCull && node->frustumCulled;
            stats += node->entity->draw(camera, model, doFrustumCull, overrideMaterial);
        }
    }

    for (auto& child : node->children) {
        stats += drawNode(scene, camera, child, model, overrideMaterial);
    }

    return stats;
}

RenderStats OpenGLRenderer::drawNode(const Scene &scene, const Camera &camera, Node* node, const glm::mat4 &parentTransform,
                                     const PointLight* pointLight, const Material* overrideMaterial) {
    const glm::mat4 &model = parentTransform * node->getTransformParentFromLocal();

    RenderStats stats;
    if (node->entity != nullptr) {
        if (node->visible) {
            // don't have to bind to scene and camera here, since we are only drawing shadows
            stats += node->entity->draw(camera, model, pointLight->boundingSphere, overrideMaterial);
        }
    }

    for (auto& child : node->children) {
        stats += drawNode(scene, camera, child, model, pointLight, overrideMaterial);
    }

    return stats;
}

RenderStats OpenGLRenderer::drawToScreen(const Shader &screenShader, const RenderTarget* overrideRenderTarget) {
    RenderStats stats;

    if (overrideRenderTarget != nullptr) {
        overrideRenderTarget->bind();
        glViewport(0, 0, overrideRenderTarget->width, overrideRenderTarget->height);
    }
    else {
        // screen buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, width, height);
    }

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    screenShader.bind();

    // set gbuffer texture uniforms
    screenShader.setTexture("screenColor", gBuffer.colorBuffer, 0);
    screenShader.setTexture("screenDepth", gBuffer.depthBuffer, 1);
    screenShader.setTexture("screenPositions", gBuffer.positionBuffer, 2);
    screenShader.setTexture("screenNormals", gBuffer.normalsBuffer, 3);
    screenShader.setTexture("idBuffer", gBuffer.idBuffer, 4);

    stats += outputFsQuad.draw();

    screenShader.unbind();

    if (overrideRenderTarget != nullptr) {
        overrideRenderTarget->unbind();
    }

    return stats;
}

RenderStats OpenGLRenderer::drawToRenderTarget(const Shader &screenShader, const RenderTarget &renderTarget) {
    return drawToScreen(screenShader, &renderTarget);
}

void OpenGLRenderer::resize(unsigned int width, unsigned int height) {
    this->width = width;
    this->height = height;

    glViewport(0, 0, width, height);
    gBuffer.resize(width, height);
    gBuffer.resize(width, height);
}
