#include <OpenGLRenderer.h>

void OpenGLRenderer::init() {
    // enable backface culling
    glEnable(GL_CULL_FACE);
    glFrontFace(GL_CCW);

    // enable seamless cube map sampling for lower mip levels in the pre-filter map
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    // enable alpha blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // enable msaa
    glEnable(GL_MULTISAMPLE);
}

void OpenGLRenderer::drawSkyBox(Shader &shader, Scene* scene, Camera* camera) {
    shader.bind();
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
    shader.setVec3("viewPos", camera->position);

    if (scene->skyBox != nullptr) {
        glActiveTexture(GL_TEXTURE0 + 4);
        glBindTexture(GL_TEXTURE_CUBE_MAP, scene->skyBox->ID);
        shader.setInt("skybox", 4);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    }

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
        node->entity->draw(shader);
    }

    for (auto& child : node->children) {
        drawNode(shader, child, model);
    }
}
