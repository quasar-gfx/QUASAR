#ifndef POST_PROCESSING_EFFECT_H
#define POST_PROCESSING_EFFECT_H

#include <Renderers/OpenGLRenderer.h>
#include <Cameras/PerspectiveCamera.h>

namespace quasar {

class PostProcessingEffect {
public:
    virtual RenderStats drawToScreen(OpenGLRenderer& renderer) = 0;
    virtual RenderStats drawToRenderTarget(OpenGLRenderer& renderer, RenderTargetBase &rt) = 0;
};

} // namespace quasar

#endif // POST_PROCESSING_EFFECT_H
