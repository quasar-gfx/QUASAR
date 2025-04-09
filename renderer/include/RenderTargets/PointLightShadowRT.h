#ifndef POINT_LIGHT_SHADOW_BUFFER_H
#define POINT_LIGHT_SHADOW_BUFFER_H

#include <RenderTargets/RenderTargetBase.h>

namespace quasar {

class PointLightShadowRT : public RenderTargetBase {
public:
    CubeMap depthCubeMap;

    PointLightShadowRT(const RenderTargetCreateParams &params)
            : RenderTargetBase(params)
            ,  depthCubeMap({
                .width = params.width,
                .height = params.height,
                .type = CubeMapType::SHADOW
            }) {
        framebuffer.bind();
        framebuffer.attachCubeMap(depthCubeMap, GL_DEPTH_ATTACHMENT);

#ifdef GL_CORE
        glDrawBuffer(GL_NONE);
#endif
        glReadBuffer(GL_NONE);

        if (!framebuffer.checkStatus()) {
            throw std::runtime_error("PointLightShadowRT Framebuffer is not complete!");
        }

        glClear(GL_DEPTH_BUFFER_BIT);

        framebuffer.unbind();
    }
};

} // namespace quasar

#endif // POINT_LIGHT_SHADOW_BUFFER_H
