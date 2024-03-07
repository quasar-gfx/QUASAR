#include <OpenGLRenderer.h>

void OpenGLRenderer::init() {
    // enable msaa
    glEnable(GL_MULTISAMPLE);

    // enable seamless cube map sampling for lower mip levels in the pre-filter map
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    // enable backface culling
    // glEnable(GL_CULL_FACE);
    // glFrontFace(GL_CCW);
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
}

void OpenGLRenderer::draw(Shader &shader, Scene* scene, Camera* camera) {
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
        shader.setInt("dirLightDepthMap", Mesh::numTextures + texIdx);
        scene->directionalLight->dirLightShadowMapFBO.depthBuffer.bind(Mesh::numTextures + texIdx);

        scene->directionalLight->draw(shader);
    }
    texIdx++;

    for (int i = 0; i < scene->pointLights.size(); i++) {
        shader.setFloat("farPlane", scene->pointLights[i]->zFar);
        shader.setInt("pointLightDepthMaps[" + std::to_string(i) + "]", Mesh::numTextures + texIdx);
        scene->pointLights[i]->pointLightShadowMapFBO.depthCubeMap.bind(Mesh::numTextures + texIdx);
        texIdx++;

        scene->pointLights[i]->draw(shader, i);
    }

    for (auto child : scene->children) {
        drawNode(shader, child, glm::mat4(1.0f));
    }

    shader.unbind();
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
