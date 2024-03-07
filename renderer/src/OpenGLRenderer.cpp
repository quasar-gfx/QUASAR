#include <OpenGLRenderer.h>

void OpenGLRenderer::init() {
    // enable backface culling
    // glEnable(GL_CULL_FACE);
    // glFrontFace(GL_CCW);

    // enable seamless cube map sampling for lower mip levels in the pre-filter map
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    // enable msaa
    glEnable(GL_MULTISAMPLE);
}

void OpenGLRenderer::drawDirLightShadows(Shader &shader, Scene* scene, Camera* camera) {
    if (scene->directionalLight == nullptr) {
        return;
    }

    scene->directionalLight->dirLightShadowMapFBO.bind();
    glViewport(0, 0, 2048, 2048);
    glClear(GL_DEPTH_BUFFER_BIT);

    shader.bind();

    float left = scene->directionalLight->orthoBoxSize;
    float right = -left;
    float top = left;
    float bottom = -top;
    scene->directionalLight->shadowProjectionMat = glm::ortho(left, right, bottom, top, scene->directionalLight->zNear, scene->directionalLight->zFar);
    scene->directionalLight->lightView = glm::lookAt(
                                    -scene->directionalLight->direction,
                                    glm::vec3(0.0f, 0.0f, 0.0f),
                                    glm::vec3(0.0f, 1.0f, 0.0f));

    scene->directionalLight->lightSpaceMatrix = scene->directionalLight->shadowProjectionMat * scene->directionalLight->lightView;
    shader.setMat4("lightSpaceMatrix", scene->directionalLight->lightSpaceMatrix);

    for (auto child : scene->children) {
        drawNode(shader, child, glm::mat4(1.0f));
    }

    shader.unbind();
    scene->directionalLight->dirLightShadowMapFBO.unbind();
}

void OpenGLRenderer::drawPointLightShadows(Shader &shader, Scene* scene, Camera* camera) {
    for (int i = 0; i < scene->pointLights.size(); i++) {
        scene->pointLights[i]->pointLightShadowMapFBO.bind();
        glViewport(0, 0, 2048, 2048);
        glClear(GL_DEPTH_BUFFER_BIT);

        shader.bind();

        shader.setVec3("lightPos", scene->pointLights[i]->position);
        shader.setFloat("far_plane", scene->pointLights[i]->zFar);

        scene->pointLights[i]->shadowProjectionMat = glm::perspective(glm::radians(90.0f), 1.0f, scene->pointLights[i]->zNear, scene->pointLights[i]->zFar);
        scene->pointLights[i]->lookAtPerFace[0] = glm::lookAt(scene->pointLights[i]->position, scene->pointLights[i]->position + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
        scene->pointLights[i]->lookAtPerFace[1] = glm::lookAt(scene->pointLights[i]->position, scene->pointLights[i]->position + glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
        scene->pointLights[i]->lookAtPerFace[2] = glm::lookAt(scene->pointLights[i]->position, scene->pointLights[i]->position + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        scene->pointLights[i]->lookAtPerFace[3] = glm::lookAt(scene->pointLights[i]->position, scene->pointLights[i]->position + glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
        scene->pointLights[i]->lookAtPerFace[4] = glm::lookAt(scene->pointLights[i]->position, scene->pointLights[i]->position + glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f));
        scene->pointLights[i]->lookAtPerFace[5] = glm::lookAt(scene->pointLights[i]->position, scene->pointLights[i]->position + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f));

        glm::mat4 shadowProj = scene->pointLights[i]->shadowProjectionMat;
        for (int face = 0; face < NUM_CUBEMAP_FACES; face++) {
            shader.setMat4("shadowMatrices[" + std::to_string(face) + "]", shadowProj * scene->pointLights[i]->lookAtPerFace[face]);
        }

        for (auto child : scene->children) {
            drawNode(shader, child, glm::mat4(1.0f));
        }

        shader.unbind();
        scene->pointLights[i]->pointLightShadowMapFBO.unbind();
    }
}

void OpenGLRenderer::drawSkyBox(Shader &shader, Scene* scene, Camera* camera) {
    shader.bind();
    shader.setInt("environmentMap", 0);
    shader.setMat4("view", camera->getViewMatrix());
    shader.setMat4("projection", camera->getProjectionMatrix());

    if (scene->ambientLight != nullptr) {
        scene->ambientLight->draw(shader);
    }

    if (scene->envCubeMap != nullptr) {
        scene->envCubeMap->draw(shader, camera);
    }

    shader.unbind();
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

    int texIdx = 3;
    if (scene->directionalLight != nullptr) {
        shader.setMat4("lightSpaceMatrix", scene->directionalLight->lightSpaceMatrix);
        shader.setInt("dirLightDepthMap", Mesh::numTextures + texIdx);
        scene->directionalLight->dirLightShadowMapFBO.depthBuffer.bind(Mesh::numTextures + texIdx);

        scene->directionalLight->draw(shader);
    }
    texIdx++;

    for (int i = 0; i < scene->pointLights.size(); i++) {
        shader.setFloat("far_plane", scene->pointLights[i]->zFar);
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
