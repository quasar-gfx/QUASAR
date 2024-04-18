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

    // enable alpha blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // background skybox shader
    ShaderCreateParams skyboxShaderParams = {
        .vertexData = SHADER_SKYBOX_VERT,
        .vertexDataSize = SHADER_SKYBOX_VERT_len,
        .fragmentData = SHADER_SKYBOX_FRAG,
        .fragmentDataSize = SHADER_SKYBOX_FRAG_len
    };
    skyboxShader = std::make_shared<Shader>(skyboxShaderParams);

    outputFsQuad.init();
    gBuffer.createBuffers(width, height);
}

void OpenGLRenderer::updateDirLightShadow(Scene &scene, Camera &camera) {
    if (scene.directionalLight == nullptr) {
        return;
    }

    scene.directionalLight->shadowMapFramebuffer.bind();
    glClear(GL_DEPTH_BUFFER_BIT);

    scene.directionalLight->shadowMapMaterial.bind();
    scene.directionalLight->shadowMapMaterial.shader->setMat4("lightSpaceMatrix", scene.directionalLight->lightSpaceMatrix);

    for (auto& child : scene.children) {
        drawNode(scene, camera, child, glm::mat4(1.0f), &scene.directionalLight->shadowMapMaterial);
    }

    scene.directionalLight->shadowMapMaterial.unbind();
    scene.directionalLight->shadowMapFramebuffer.unbind();
}

void OpenGLRenderer::updatePointLightShadows(Scene &scene, Camera &camera) {
    for (int i = 0; i < scene.pointLights.size(); i++) {
        auto pointLight = scene.pointLights[i];

        pointLight->shadowMapFramebuffer.bind();
        glClear(GL_DEPTH_BUFFER_BIT);

        pointLight->shadowMapMaterial.bind();
        pointLight->shadowMapMaterial.shader->setVec3("lightPos", pointLight->position);
        pointLight->shadowMapMaterial.shader->setFloat("farPlane", pointLight->zFar);

        glm::mat4 shadowProj = pointLight->shadowProjectionMat;
        for (int face = 0; face < NUM_CUBEMAP_FACES; face++) {
            pointLight->shadowMapMaterial.shader->setMat4("shadowMatrices[" + std::to_string(face) + "]", shadowProj * pointLight->lookAtPerFace[face]);
        }

        for (auto& child : scene.children) {
            drawNode(scene, camera, child, glm::mat4(1.0f), &pointLight->shadowMapMaterial);
        }

        pointLight->shadowMapMaterial.unbind();
        pointLight->shadowMapFramebuffer.unbind();
    }
}

void OpenGLRenderer::drawSkyBox(Scene &scene, Camera &camera) {
    if (scene.envCubeMap == nullptr) {
        return;
    }

    gBuffer.bind();
    // dont clear color or depth bit here, since we want this to draw over

    bool isHDR = (scene.envCubeMap->type == CUBE_MAP_HDR);

    skyboxShader->bind();
    skyboxShader->setInt("environmentMap", 0);
    skyboxShader->setMat4("view", camera.getViewMatrix());
    skyboxShader->setMat4("projection", camera.getProjectionMatrix());
    skyboxShader->setBool("isHDR", isHDR);

    if (scene.envCubeMap != nullptr) {
        scene.envCubeMap->draw(*skyboxShader, camera);
    }

    skyboxShader->unbind();

    gBuffer.unbind();
}

void OpenGLRenderer::drawObjects(Scene &scene, Camera &camera) {
    // update shadows
    updateDirLightShadow(scene, camera);
    updatePointLightShadows(scene, camera);

    // bind to gBuffer and draw scene as we normally would to color texture
    gBuffer.bind();
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    for (auto& child : scene.children) {
        drawNode(scene, camera, child, glm::mat4(1.0f));
    }

    drawSkyBox(scene, camera);

    // now bind back to default gBuffer and draw a quad plane with the attached gBuffer color texture
    gBuffer.unbind();
}

void OpenGLRenderer::drawNode(Scene &scene, Camera &camera, Node* node, glm::mat4 parentTransform, Material* overrideMaterial) {
    glm::mat4 model = parentTransform * node->getTransformParentFromLocal();

    if (node->entity != nullptr) {
        node->entity->bindSceneAndCamera(scene, camera, model, overrideMaterial);
        node->entity->draw(overrideMaterial);
    }

    for (auto& child : node->children) {
        drawNode(scene, camera, child, model, overrideMaterial);
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
