#include <OpenGLRenderer.h>

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

void OpenGLRenderer::updateDirLightShadow(Scene &scene, Camera &camera) {
    if (scene.directionalLight == nullptr) {
        return;
    }

    scene.directionalLight->shadowMapRenderTarget.bind();
    glClear(GL_DEPTH_BUFFER_BIT);

    for (auto& child : scene.children) {
        drawNode(scene, camera, child, glm::mat4(1.0f), &scene.directionalLight->shadowMapMaterial);
    }

    scene.directionalLight->shadowMapRenderTarget.unbind();
}

void OpenGLRenderer::updatePointLightShadows(Scene &scene, Camera &camera) {
    for (int i = 0; i < scene.pointLights.size(); i++) {
        auto pointLight = scene.pointLights[i];

        pointLight->shadowMapRenderTarget.bind();
        glClear(GL_DEPTH_BUFFER_BIT);

        pointLight->shadowMapMaterial.bind();
        pointLight->shadowMapMaterial.shader->setVec3("lightPos", pointLight->position);
        pointLight->shadowMapMaterial.shader->setFloat("farPlane", pointLight->zFar);

        glm::mat4 shadowProj = pointLight->shadowProjectionMat;
        for (int face = 0; face < NUM_CUBEMAP_FACES; face++) {
            pointLight->shadowMapMaterial.shader->setMat4("shadowMatrices[" + std::to_string(face) + "]", shadowProj * pointLight->lookAtPerFace[face]);
        }
        pointLight->shadowMapMaterial.unbind();

        for (auto& child : scene.children) {
            drawNode(scene, camera, child, glm::mat4(1.0f), &pointLight->shadowMapMaterial);
        }

        pointLight->shadowMapRenderTarget.unbind();
    }
}

unsigned int OpenGLRenderer::drawSkyBox(Scene &scene, Camera &camera) {
    unsigned int trianglesDrawn = 0;
    if (scene.envCubeMap == nullptr) {
        return trianglesDrawn;
    }

    gBuffer.bind();
    // dont clear color or depth bit here, since we want this to draw over

    skyboxShader.bind();
    skyboxShader.setInt("environmentMap", 0);
    skyboxShader.setMat4("view", camera.getViewMatrix());
    skyboxShader.setMat4("projection", camera.getProjectionMatrix());

    if (scene.envCubeMap != nullptr) {
        trianglesDrawn = scene.envCubeMap->draw(skyboxShader, camera);
    }

    skyboxShader.unbind();

    gBuffer.unbind();

    return trianglesDrawn;
}

unsigned int OpenGLRenderer::drawObjects(Scene &scene, Camera &camera) {
    // update shadows
    updateDirLightShadow(scene, camera);
    updatePointLightShadows(scene, camera);

    // bind to gBuffer and draw scene as we normally would to color texture
    gBuffer.bind();
    glClearColor(scene.backgroundColor.x, scene.backgroundColor.y, scene.backgroundColor.z, scene.backgroundColor.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    unsigned int trianglesDrawn = 0;
    for (auto& child : scene.children) {
        trianglesDrawn += drawNode(scene, camera, child, glm::mat4(1.0f));
    }

    trianglesDrawn += drawSkyBox(scene, camera);

    // now bind back to default gBuffer and draw a quad plane with the attached gBuffer color texture
    gBuffer.unbind();

    return trianglesDrawn;
}

unsigned int OpenGLRenderer::drawNode(Scene &scene, Camera &camera, Node* node, glm::mat4 parentTransform, Material* overrideMaterial) {
    glm::mat4 model = parentTransform * node->getTransformParentFromLocal();

    unsigned int trianglesDrawn = 0;
    if (node->entity != nullptr) {
        node->entity->bindSceneAndCamera(scene, camera, model, overrideMaterial);
        trianglesDrawn += node->entity->draw(overrideMaterial);
    }

    for (auto& child : node->children) {
        trianglesDrawn += drawNode(scene, camera, child, model, overrideMaterial);
    }

    return trianglesDrawn;
}

void OpenGLRenderer::drawToScreen(Shader &screenShader, RenderTarget* overrideRenderTarget) {
    if (overrideRenderTarget != nullptr) {
        overrideRenderTarget->bind();
    }
    else {
        // screen buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    glViewport(0, 0, width, height);

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    screenShader.bind();
    screenShader.setInt("screenPositions", 0);
    gBuffer.positionBuffer.bind(0);
    screenShader.setInt("screenNormals", 1);
    gBuffer.normalsBuffer.bind(1);
    screenShader.setInt("idBuffer", 2);
    gBuffer.idBuffer.bind(2);
    screenShader.setInt("screenColor", 3);
    gBuffer.colorBuffer.bind(3);
    screenShader.setInt("screenDepth", 4);
    gBuffer.depthBuffer.bind(4);

    outputFsQuad.draw();

    gBuffer.positionBuffer.unbind();
    gBuffer.normalsBuffer.unbind();
    gBuffer.idBuffer.unbind();
    gBuffer.colorBuffer.unbind();
    gBuffer.depthBuffer.unbind();
    screenShader.unbind();

    if (overrideRenderTarget != nullptr) {
        overrideRenderTarget->unbind();
    }
}

void OpenGLRenderer::drawToRenderTarget(Shader &screenShader, RenderTarget &renderTarget) {
    drawToScreen(screenShader, &renderTarget);
}

void OpenGLRenderer::resize(unsigned int width, unsigned int height) {
    this->width = width;
    this->height = height;

    glViewport(0, 0, width, height);
    gBuffer.resize(width, height);
    gBuffer.resize(width, height);
}
