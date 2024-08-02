#include <OpenGLRenderer.h>
#include <Materials/UnlitMaterial.h>

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
    // enable msaa for screen buffer
    glEnable(GL_MULTISAMPLE);

    // enable seamless cube map sampling for lower mip levels in the pre-filter map
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    // enable alpha blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // enable setting vertex size for point clouds
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
}

RenderStats OpenGLRenderer::updateDirLightShadow(Scene &scene, Camera &camera) {
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

RenderStats OpenGLRenderer::updatePointLightShadows(Scene &scene, Camera &camera) {
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

RenderStats OpenGLRenderer::drawSkyBox(Scene &scene, Camera &camera) {
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

RenderStats OpenGLRenderer::drawObjects(Scene &scene, Camera &camera) {
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

RenderStats OpenGLRenderer::drawNode(Scene &scene, Camera &camera, Node* node, const glm::mat4 &parentTransform,
                                     bool frustumCull, const Material* overrideMaterial) {
    const glm::mat4 &model = parentTransform * node->getTransformParentFromLocal();

    RenderStats stats;
    if (node->entity != nullptr) {
        if (node->visible) {
            node->entity->bindMaterial(scene, camera, model, overrideMaterial);
            bool doFrustumCull = frustumCull && node->frustumCulled;
            stats += node->entity->draw(scene, camera, model, doFrustumCull, overrideMaterial);
        }
    }

    for (auto& child : node->children) {
        stats += drawNode(scene, camera, child, model, overrideMaterial);
    }

    return stats;
}

RenderStats OpenGLRenderer::drawNode(Scene &scene, Camera &camera, Node* node, const glm::mat4 &parentTransform,
                                     const PointLight* pointLight, const Material* overrideMaterial) {
    const glm::mat4 &model = parentTransform * node->getTransformParentFromLocal();

    RenderStats stats;
    if (node->entity != nullptr) {
        if (node->visible) {
            // don't have to bind to scene and camera here, since we are only drawing shadows
            stats += node->entity->draw(scene, camera, model, pointLight->boundingSphere, overrideMaterial);
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

    screenShader.setTexture("screenPositions", gBuffer.positionBuffer, 0);
    screenShader.setTexture("screenNormals", gBuffer.normalsBuffer, 1);
    screenShader.setTexture("idBuffer", gBuffer.idBuffer, 2);
    screenShader.setTexture("screenColor", gBuffer.colorBuffer, 3);
    screenShader.setTexture("screenDepth", gBuffer.depthBuffer, 4);

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
