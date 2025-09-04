#ifndef SSAO_H
#define SSAO_H

#include <random>

#include <Shaders/Shader.h>
#include <RenderTargets/RenderTarget.h>

#include <PostProcessing/PostProcessingEffect.h>

namespace quasar {

class SSAO: public PostProcessingEffect {
public:
    SSAO(glm::uvec2& windowSize, PerspectiveCamera& camera, uint seed = 42)
        : windowSize(windowSize)
        , camera(camera)
        , generator(seed)
        , randomFloats(0.0, 1.0)
        , ssaoShader({
            .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
            .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_SSAO_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_SSAO_FRAG_len,
        })
        , ssaoBlurShader({
            .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
            .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_SSAO_BLUR_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_SSAO_BLUR_FRAG_len,
        })
        , ssaoFinalShader({
            .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
            .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_SSAO_FINAL_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_SSAO_FINAL_FRAG_len,
        })
        , ssaoRenderTarget({
            .width = windowSize.x,
            .height = windowSize.y,
            .internalFormat = GL_RED,
            .format = GL_RED,
            .type = GL_FLOAT,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
        })
        , ssaoBlurRenderTarget({
            .width = windowSize.x,
            .height = windowSize.y,
            .internalFormat = GL_RED,
            .format = GL_RED,
            .type = GL_FLOAT,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
        })
    {
        for (uint i = 0; i < 64; i++) {
            glm::vec3 sample(randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0, randomFloats(generator));
            sample = glm::normalize(sample);
            sample *= randomFloats(generator);
            float scale = float(i) / 64.0f;

            scale = lerp(0.1f, 1.0f, scale * scale);
            sample *= scale;
            ssaoKernel.push_back(sample);
        }

        for (uint i = 0; i < 16; i++) {
            glm::vec3 noise(randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0, 0.0f);
            ssaoNoise.push_back(noise);
        }
        noiseTexture = new Texture({
            .width = 4,
            .height = 4,
            .internalFormat = GL_RGBA32F,
            .format = GL_RGBA,
            .type = GL_FLOAT,
            .wrapS = GL_REPEAT,
            .wrapT = GL_REPEAT,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
            .data = reinterpret_cast<unsigned char*>(ssaoNoise.data()),
        });
    }

    void setExposure(float exposure) {
        ssaoFinalShader.bind();
        ssaoFinalShader.setFloat("exposure", exposure);
    }

    RenderStats drawToScreen(OpenGLRenderer& renderer) override {
        RenderStats stats;

        // Render ssao
        ssaoShader.bind();
        ssaoShader.setMat4("view", camera.getViewMatrix());
        ssaoShader.setMat4("projection", camera.getProjectionMatrix());
        for (uint i = 0; i < 64; i++) {
            ssaoShader.setVec3("samples[" + std::to_string(i) + "]", ssaoKernel[i]);
        }
        ssaoShader.setTexture("noiseTexture", *noiseTexture, 5);
        stats += renderer.drawToRenderTarget(ssaoShader, ssaoRenderTarget);

        // Blur ssao
        ssaoBlurShader.bind();
        ssaoBlurShader.setTexture("ssaoInput", ssaoRenderTarget.colorTexture, 5);
        stats += renderer.drawToRenderTarget(ssaoBlurShader, ssaoBlurRenderTarget);

        // Final ssao
        ssaoFinalShader.bind();
        ssaoFinalShader.setTexture("ssao", ssaoBlurRenderTarget.colorTexture, 5);
        stats += renderer.drawToScreen(ssaoFinalShader);

        return stats;
    }

    RenderStats drawToRenderTarget(OpenGLRenderer& renderer, RenderTargetBase& rt) override {
        RenderStats stats;

        // Render ssao
        ssaoShader.bind();
        ssaoShader.setMat4("view", camera.getViewMatrix());
        ssaoShader.setMat4("projection", camera.getProjectionMatrix());
        for (uint i = 0; i < 64; i++) {
            ssaoShader.setVec3("samples[" + std::to_string(i) + "]", ssaoKernel[i]);
        }
        ssaoShader.setTexture("noiseTexture", *noiseTexture, 5);
        renderer.setScreenShaderUniforms(ssaoShader);
        stats += renderer.drawToRenderTarget(ssaoShader, ssaoRenderTarget);

        // Blur ssao
        ssaoBlurShader.bind();
        ssaoBlurShader.setTexture("ssaoInput", ssaoRenderTarget.colorTexture, 5);
        renderer.setScreenShaderUniforms(ssaoBlurShader);
        stats += renderer.drawToRenderTarget(ssaoBlurShader, ssaoBlurRenderTarget);

        // Final ssao
        ssaoFinalShader.bind();
        ssaoFinalShader.setTexture("ssao", ssaoBlurRenderTarget.colorTexture, 5);
        renderer.setScreenShaderUniforms(ssaoFinalShader);
        stats += renderer.drawToRenderTarget(ssaoFinalShader, rt);

        return stats;
    }

private:
    glm::uvec2& windowSize;
    PerspectiveCamera& camera;

    Shader ssaoShader;
    Shader ssaoBlurShader;
    Shader ssaoFinalShader;

    RenderTarget ssaoBlurRenderTarget;
    RenderTarget ssaoRenderTarget;

    const Texture* noiseTexture;

    std::uniform_real_distribution<GLfloat> randomFloats;
    std::default_random_engine generator;
    std::vector<glm::vec3> ssaoKernel;

    std::vector<glm::vec3> ssaoNoise;

    float lerp(float a, float b, float f) {
        return a + f * (b - a);
    }
};

} // namespace quasar

#endif // SSAO_H
