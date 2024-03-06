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

void OpenGLRenderer::drawSkyBox(Shader &shader, Scene* scene, Camera* camera) {
    shader.bind();
    shader.setInt("environmentMap", 0);
    shader.setMat4("view", camera->getViewMatrix());
    shader.setMat4("projection", camera->getProjectionMatrix());

    if (scene->ambientLight != nullptr) {
        scene->ambientLight->draw(shader);
    }

    if (scene->skyBox != nullptr) {
        scene->skyBox->draw(shader, camera);
    }

    shader.unbind();
}

void OpenGLRenderer::draw(Shader &shader, Scene* scene, Camera* camera) {
    shader.bind();

    shader.setMat4("view", camera->getViewMatrix());
    shader.setMat4("projection", camera->getProjectionMatrix());
    shader.setVec3("camPos", camera->position);

    if (scene->ambientLight != nullptr) {
        scene->ambientLight->draw(shader);
    }

    if (scene->directionalLight != nullptr) {
        scene->directionalLight->draw(shader);
    }

    for (int i = 0; i < scene->pointLights.size(); i++) {
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
