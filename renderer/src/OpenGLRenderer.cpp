#include <OpenGLRenderer.h>

void OpenGLRenderer::init(unsigned int width, unsigned int height) {
    this->width = width;
    this->height = height;

    // enable depth testing
    glEnable(GL_DEPTH_TEST);

    // enable msaa
    glEnable(GL_MULTISAMPLE);

    // enable seamless cube map sampling for lower mip levels in the pre-filter map
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    // enable backface culling
    // glEnable(GL_CULL_FACE);
    // glFrontFace(GL_CCW);

    // enable aplha blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    outputFsQuad.init();
    gBuffer.createBuffers(width, height);
}

void OpenGLRenderer::updateDirLightShadowMap(Shader &dirLightShadowsShader, Scene* scene, Camera* camera) {
    if (scene->directionalLight == nullptr) {
        return;
    }

    scene->directionalLight->dirLightShadowMapFBO.bind();
    glClear(GL_DEPTH_BUFFER_BIT);

    dirLightShadowsShader.bind();
    dirLightShadowsShader.setMat4("lightSpaceMatrix", scene->directionalLight->lightSpaceMatrix);

    for (auto child : scene->children) {
        drawNode(dirLightShadowsShader, child, glm::mat4(1.0f));
    }

    dirLightShadowsShader.unbind();
    scene->directionalLight->dirLightShadowMapFBO.unbind();
}

void OpenGLRenderer::updatePointLightShadowMaps(Shader &pointLightShadowsShader, Scene* scene, Camera* camera) {
    for (int i = 0; i < scene->pointLights.size(); i++) {
        scene->pointLights[i]->pointLightShadowMapFBO.bind();
        glClear(GL_DEPTH_BUFFER_BIT);

        pointLightShadowsShader.bind();
        pointLightShadowsShader.setVec3("lightPos", scene->pointLights[i]->position);
        pointLightShadowsShader.setFloat("farPlane", scene->pointLights[i]->zFar);

        glm::mat4 shadowProj = scene->pointLights[i]->shadowProjectionMat;
        for (int face = 0; face < NUM_CUBEMAP_FACES; face++) {
            pointLightShadowsShader.setMat4("shadowMatrices[" + std::to_string(face) + "]", shadowProj * scene->pointLights[i]->lookAtPerFace[face]);
        }

        for (auto child : scene->children) {
            drawNode(pointLightShadowsShader, child, glm::mat4(1.0f));
        }

        pointLightShadowsShader.unbind();
        scene->pointLights[i]->pointLightShadowMapFBO.unbind();
    }
}

void OpenGLRenderer::drawSkyBox(Shader &backgroundShader, Scene* scene, Camera* camera) {
    gBuffer.bind();
    // dont clear color or depth bit here, since we want this to draw over

    backgroundShader.bind();
    backgroundShader.setInt("environmentMap", 0);
    backgroundShader.setMat4("view", camera->getViewMatrix());
    backgroundShader.setMat4("projection", camera->getProjectionMatrix());

    if (scene->ambientLight != nullptr) {
        scene->ambientLight->draw(backgroundShader);
    }

    if (scene->envCubeMap != nullptr) {
        scene->envCubeMap->draw(backgroundShader, camera);
    }

    backgroundShader.unbind();

    gBuffer.unbind();
}

void OpenGLRenderer::drawObjects(Shader &shader, Scene* scene, Camera* camera) {
    // bind to gBuffer and draw scene as we normally would to color texture
    gBuffer.bind();
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    shader.bind();

    shader.setMat4("view", camera->getViewMatrix());
    shader.setMat4("projection", camera->getProjectionMatrix());
    shader.setVec3("camPos", camera->position);
    scene->bindPBREnvMap(shader);

    if (scene->ambientLight != nullptr) {
        scene->ambientLight->draw(shader);
    }

    int texIdx = Scene::numTextures;
    if (scene->directionalLight != nullptr) {
        shader.setMat4("lightSpaceMatrix", scene->directionalLight->lightSpaceMatrix);
        shader.setInt("dirLightShadowMap", Mesh::numTextures + texIdx);
        scene->directionalLight->dirLightShadowMapFBO.depthBuffer.bind(Mesh::numTextures + texIdx);

        scene->directionalLight->draw(shader);
    }
    texIdx++;

    for (int i = 0; i < scene->pointLights.size(); i++) {
        shader.setFloat("farPlane", scene->pointLights[i]->zFar);
        shader.setInt("pointLightShadowMaps[" + std::to_string(i) + "]", Mesh::numTextures + texIdx);
        scene->pointLights[i]->pointLightShadowMapFBO.depthCubeMap.bind(Mesh::numTextures + texIdx);
        texIdx++;

        scene->pointLights[i]->draw(shader, i);
    }

    for (auto child : scene->children) {
        drawNode(shader, child, glm::mat4(1.0f));
    }

    shader.unbind();

    // now bind back to default gBuffer and draw a quad plane with the attached gBuffer color texture
    gBuffer.unbind();
}

void OpenGLRenderer::drawNode(Shader &shader, Node* node, glm::mat4 parentTransform) {
    glm::mat4 model = parentTransform * node->getTransformParentFromLocal();

    if (node->entity != nullptr) {
        shader.setMat4("model", model);
        shader.setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(model))));
        node->entity->draw(shader);
    }

    for (auto& child : node->children) {
        drawNode(shader, child, model);
    }
}

void OpenGLRenderer::drawToScreen(Shader &screenShader, unsigned int screenWidth, unsigned int screenHeight) {
    glViewport(0, 0, screenWidth, screenHeight);

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    screenShader.bind();
    screenShader.setInt("screenPositions", 0);
    gBuffer.positionBuffer.bind(0);
    screenShader.setInt("screenNormals", 1);
    gBuffer.normalsBuffer.bind(1);
    screenShader.setInt("screenColor", 2);
    gBuffer.colorBuffer.bind(2);
    screenShader.setInt("screenDepth", 3);
    gBuffer.depthBuffer.bind(3);

    outputFsQuad.draw();

    gBuffer.positionBuffer.unbind();
    gBuffer.normalsBuffer.unbind();
    gBuffer.colorBuffer.unbind();
    gBuffer.depthBuffer.unbind();
    screenShader.unbind();
}
