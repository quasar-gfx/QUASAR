#ifndef POINT_LIGHT_SHADOW_BUFFER_H
#define POINT_LIGHT_SHADOW_BUFFER_H

#include <RenderTargets/RenderTargetBase.h>

class PointLightShadowRT : public RenderTargetBase {
public:
    CubeMap depthCubeMap;

    explicit PointLightShadowRT() = default;

    void init(const RenderTargetCreateParams &params) override {
        width = params.width;
        height = params.height;

        depthCubeMap.init(params.width, params.height, CubeMapType::SHADOW);

        framebuffer.init();
        framebuffer.bind();
        framebuffer.attachCubeMap(depthCubeMap, GL_DEPTH_ATTACHMENT);

        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);

        if (!framebuffer.checkStatus()) {
            throw std::runtime_error("PointLightShadowRT Framebuffer is not complete!");
        }

        glClear(GL_DEPTH_BUFFER_BIT);

        framebuffer.unbind();
    }
};

#endif // POINT_LIGHT_SHADOW_BUFFER_H
