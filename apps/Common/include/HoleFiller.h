#ifndef HOLE_FILLER_H
#define HOLE_FILLER_H

#include <PostProcessing/PostProcessingEffect.h>

namespace quasar {

class HoleFiller : public PostProcessingEffect {
public:
    HoleFiller();

    void enableTonemapping(bool enable);
    void setDepthThreshold(float depthThreshold);

    RenderStats drawToScreen(OpenGLRenderer& renderer) override;

    RenderStats drawToRenderTarget(OpenGLRenderer& renderer, RenderTargetBase& rt) override;

private:
    Shader shader;
};

} // namespace quasar

#endif // HOLE_FILLER_H
