#ifndef OPENGL_RENDERER_H
#define OPENGL_RENDERER_H

#include <vector>

#include <Utils/FileIO.h>
#include <Utils/TimeUtils.h>
#include <Texture.h>
#include <CubeMap.h>
#include <Scene.h>
#include <Cameras/Camera.h>
#include <Shaders/Shader.h>
#include <Shaders/ComputeShader.h>
#include <Materials/LitMaterial.h>
#include <Materials/UnlitMaterial.h>
#include <Lights/Lights.h>
#include <OpenGLAppConfig.h>

namespace quasar {

class OpenGLRenderer {
public:
    uint width, height;
    uint windowWidth, windowHeight;

    GraphicsPipeline pipeline;

    OpenGLRenderer(const Config& config);
    ~OpenGLRenderer() = default;

    void setGraphicsPipeline(const GraphicsPipeline& pipeline) { this->pipeline = pipeline; }

    virtual void setScreenShaderUniforms(const Shader& screenShader) {};

    virtual void resize(uint width, uint height);
    virtual void setWindowSize(uint width, uint height);

    virtual void beginRendering() {}
    virtual void endRendering() {}

    RenderStats updateDirLightShadow(Scene& scene, const Camera& camera);
    RenderStats updatePointLightShadows(Scene& scene, const Camera& camera);

    virtual RenderStats drawSkyBox(Scene& scene, const Camera& camera, uint32_t clearMask = 0);
    virtual RenderStats drawScene(Scene& scene, const Camera& camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    virtual RenderStats drawLights(Scene& scene, const Camera& camera);
    virtual RenderStats drawObjects(Scene& scene, const Camera& camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    virtual RenderStats drawObjectsNoLighting(Scene& scene, const Camera& camera, uint32_t clearMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    virtual RenderStats drawToScreen(const Shader& screenShader, const RenderTargetBase* overrideRenderTarget = nullptr);
    virtual RenderStats drawToRenderTarget(const Shader& screenShader, const RenderTargetBase& renderTarget);

protected:
    struct RenderItem {
        const Node* node = nullptr;
        glm::mat4 model{1.0f};
        const Material* materialOverride = nullptr;
        bool frustumCull = true;
    };

    struct RenderList {
        std::vector<RenderItem> opaque;
        std::vector<RenderItem> transparent;

        void clear() {
            opaque.clear();
            transparent.clear();
        }
        bool empty() const { return opaque.empty() && transparent.empty(); }
    };

    Shader skyboxShader;
    FullScreenQuad outputFsQuad;

    Buffer pointLightsUBO;

    RenderStats drawSkyBoxImpl(Scene& scene, const Camera& camera, uint32_t clearMask);
    RenderStats drawSceneImpl(Scene& scene, const Camera& camera, uint32_t clearMask);
    RenderStats drawLightsImpl(Scene& scene, const Camera& camera);

    virtual void gatherNodes(Scene& scene, const Camera& camera, const Node* node, const glm::mat4& parentTransform, RenderList& renderList,
                             bool frustumCull, const Material* overrideMaterial = nullptr, const Texture* prevIDMap = nullptr);
    virtual RenderStats drawOpaqueFromList(Scene& scene, const Camera& camera, RenderList& renderList);
    virtual RenderStats drawTransparentFromList(Scene& scene, const Camera& camera, RenderList& renderList);
    virtual RenderStats drawItem(Scene& scene, const Camera& camera, const RenderItem& item);

    virtual RenderStats drawNodeImmediate(Scene& scene, const Camera& camera, const Node* node, const glm::mat4& parentTransform,
                                          bool frustumCull, const Material* overrideMaterial = nullptr, const Texture* prevIDMap = nullptr);
    virtual RenderStats drawNodeImmediate(Scene& scene, const Camera& camera, const Node* node, const glm::mat4& parentTransform,
                                          const PointLight& pointLight, const Material* overrideMaterial = nullptr);
    virtual RenderStats drawNode(Scene& scene, const Camera& camera, const Node* node, const glm::mat4& model,
                                 bool frustumCull, const Material* overrideMaterial = nullptr, const Texture* prevIDMap = nullptr);
};

} // namespace quasar

#endif // OPENGL_RENDERER_H
