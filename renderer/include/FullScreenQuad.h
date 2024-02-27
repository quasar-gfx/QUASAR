#ifndef FULL_SCREEN_QUAD_H
#define FULL_SCREEN_QUAD_H

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>

#include <VertexBuffer.h>

class FullScreenQuad {
public:
    void draw() {
        // disable depth test so screen-space quad isn't discarded due to depth test.
        glDisable(GL_DEPTH_TEST);

        quadVBO->bind();
        glDrawArrays(GL_TRIANGLES, 0, 6);
        quadVBO->unbind();

        // re-enable depth test
        glEnable(GL_DEPTH_TEST);
    }

    static FullScreenQuad* create() {
        return new FullScreenQuad();
    }

private:
    struct FSQuadVertex {
        glm::vec2 position;
        glm::vec2 texCoords;
    };

    VertexBuffer* quadVBO;
    FullScreenQuad() {
        // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
        std::vector<FSQuadVertex> quadVertices = {
            {{-1.0f,  1.0f}, {0.0f, 1.0f}},
            {{-1.0f, -1.0f}, {0.0f, 0.0f}},
            {{ 1.0f, -1.0f}, {1.0f, 0.0f}},

            {{-1.0f,  1.0f}, {0.0f, 1.0f}},
            {{ 1.0f, -1.0f}, {1.0f, 0.0f}},
            {{ 1.0f,  1.0f}, {1.0f, 1.0f}}
        };

        quadVBO = new VertexBuffer(quadVertices.data(), quadVertices.size() * sizeof(FSQuadVertex));
        quadVBO->bind();
        quadVBO->addAttribute(0, 2, GL_FALSE, sizeof(FSQuadVertex), 0);
        quadVBO->addAttribute(1, 2, GL_FALSE, sizeof(FSQuadVertex), offsetof(FSQuadVertex, texCoords));
    }

    ~FullScreenQuad() { }
};

#endif // FULL_SCREEN_QUAD_H
