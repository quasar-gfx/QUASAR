#ifndef GRAPHICS_PIPELINE_H
#define GRAPHICS_PIPELINE_H

#include <Utils/Platform.h>

struct DepthState {
    bool depthTestEnabled = true;
    GLenum depthFunc = GL_LESS; // GL_NEVER, GL_LESS, GL_EQUAL, GL_LEQUAL, GL_GREATER, GL_NOTEQUAL, GL_GEQUAL, GL_ALWAYS
    GLfloat clearDepth = 1.0f;
};

struct StencilState {
    bool stencilTestEnabled = false;
    GLenum stencilFunc = GL_ALWAYS; // GL_NEVER, GL_LESS, GL_LEQUAL, GL_GREATER, GL_GEQUAL, GL_EQUAL, GL_NOTEQUAL, GL_ALWAYS
    GLint stencilRef = 0;
    GLuint stencilMask = 0xFF;
    GLenum stencilFail = GL_KEEP; // GL_KEEP, GL_ZERO, GL_REPLACE, GL_INCR, GL_INCR_WRAP, GL_DECR, GL_DECR_WRAP, GL_INVERT
    GLenum stencilPassDepthFail = GL_KEEP;
    GLenum stencilPassDepthPass = GL_KEEP;
    GLuint writeStencilMask = 0xFF;

    void enableRenderingIntoStencilBuffer() {
        stencilTestEnabled = true;

        stencilFunc = GL_ALWAYS;
        stencilRef = 1;
        stencilMask = 0xFF;

        writeStencilMask = 0xFF;

        stencilFail = GL_KEEP;
        stencilPassDepthFail = GL_KEEP;
        stencilPassDepthPass = GL_REPLACE;
    };

    void enableRenderingUsingStencilBufferAsMask() {
        stencilFunc = GL_EQUAL;
        stencilRef = 0;
        stencilMask = 0xFF;

        writeStencilMask = 0x00;
    };

    void restoreStencilState() {
        stencilTestEnabled = false;

        stencilFunc = GL_ALWAYS;
        stencilRef = 0;
        stencilMask = 0xFF;

        writeStencilMask = 0xFF;

        stencilFail = GL_KEEP;
        stencilPassDepthFail = GL_KEEP;
        stencilPassDepthPass = GL_KEEP;
    };
};

struct BlendState {
    bool blendEnabled = true;
    GLenum srcFactor = GL_SRC_ALPHA;
    GLenum dstFactor = GL_ONE_MINUS_SRC_ALPHA;
    GLenum blendEquation = GL_FUNC_ADD; // GL_FUNC_ADD, GL_FUNC_SUBTRACT, GL_FUNC_REVERSE_SUBTRACT, GL_MIN, GL_MAX
    GLfloat blendColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

struct MultiSampleState {
    bool multiSampleEnabled = true;
    unsigned int numSamples = 4;
    bool sampleShadingEnabled = false;
    GLfloat minSampleShading = 1.0f;
    bool alphaToCoverageEnabled = false;
    bool alphaToOneEnabled = false;
};

struct RasterState {
    bool cullFaceEnabled = true;
    GLenum cullFaceMode = GL_BACK; // GL_FRONT, GL_BACK, GL_FRONT_AND_BACK
    GLenum frontFace = GL_CCW; // GL_CW, GL_CCW
    bool scissorTestEnabled = false;
    bool polygonOffsetEnabled = false;
    GLfloat polygonOffsetUnits = 0.0f;
    bool sRGB = true;
};

struct GraphicsPipeline {
    DepthState depthState;
    StencilState stencilState;
    BlendState blendState;
    MultiSampleState multiSampleState;
    RasterState rasterState;

    void apply() {
        // Multisample Configuration
        if (multiSampleState.multiSampleEnabled) {
#ifdef GL_CORE
            glEnable(GL_MULTISAMPLE);
#endif
            if (multiSampleState.sampleShadingEnabled) {
                glEnable(GL_SAMPLE_SHADING);
                glMinSampleShading(multiSampleState.minSampleShading);
            }
            else {
                glDisable(GL_SAMPLE_SHADING);
            }
            if (multiSampleState.alphaToCoverageEnabled) {
                glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);
            }
            else {
                glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
            }
#ifdef GL_CORE
            if (multiSampleState.alphaToOneEnabled) {
                glEnable(GL_SAMPLE_ALPHA_TO_ONE);
            }
            else {
                glDisable(GL_SAMPLE_ALPHA_TO_ONE);
            }
#endif
        }
        else {
#ifdef GL_CORE
            glDisable(GL_MULTISAMPLE);
#endif
        }

        // Depth Configuration
        if (depthState.depthTestEnabled) {
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(depthState.depthFunc);
#ifdef GL_CORE
            glClearDepth(depthState.clearDepth);
#else
            glClearDepthf(depthState.clearDepth);
#endif
        }
        else {
            glDisable(GL_DEPTH_TEST);
        }

        // Stencil Configuration
        if (stencilState.stencilTestEnabled) {
            glEnable(GL_STENCIL_TEST);
            glStencilFunc(stencilState.stencilFunc,
                          stencilState.stencilRef,
                          stencilState.stencilMask);
            glStencilOp(stencilState.stencilFail,
                        stencilState.stencilPassDepthFail,
                        stencilState.stencilPassDepthPass);
            glStencilMask(stencilState.writeStencilMask);
        }
        else {
            glDisable(GL_STENCIL_TEST);
        }

        // Blend Configuration
        if (blendState.blendEnabled) {
            glEnable(GL_BLEND);
            glBlendFunc(blendState.srcFactor, blendState.dstFactor);
            glBlendEquation(blendState.blendEquation);
            glBlendColor(blendState.blendColor[0],
                         blendState.blendColor[1],
                         blendState.blendColor[2],
                         blendState.blendColor[3]);
        }
        else {
            glDisable(GL_BLEND);
        }

        // Cull Face Configuration
        if (rasterState.cullFaceEnabled) {
            glEnable(GL_CULL_FACE);
            glCullFace(rasterState.cullFaceMode);
            glFrontFace(rasterState.frontFace);
        }
        else {
            glDisable(GL_CULL_FACE);
        }

        // Scissor Test Configuration
        if (rasterState.scissorTestEnabled) {
            glEnable(GL_SCISSOR_TEST);
        }
        else {
            glDisable(GL_SCISSOR_TEST);
        }

        // Polygon Offset Configuration
        if (rasterState.polygonOffsetEnabled) {
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(rasterState.polygonOffsetUnits, rasterState.polygonOffsetUnits);
        }
        else {
            glDisable(GL_POLYGON_OFFSET_FILL);
        }

#ifdef GL_CORE
        // sRGB Framebuffer Configuration
        if (rasterState.sRGB) {
            glEnable(GL_FRAMEBUFFER_SRGB);
        }
        else {
            glDisable(GL_FRAMEBUFFER_SRGB);
        }
#endif
    }
};

#endif // GRAPHICS_PIPELINE_H
