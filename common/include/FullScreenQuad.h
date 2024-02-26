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
        quadVB->bind();
        glDrawArrays(GL_TRIANGLES, 0, 6);
        quadVB->unbind();
    }

    static FullScreenQuad* create() {
        return new FullScreenQuad();
    }

private:
    struct FSQuadVertex {
        glm::vec2 position;
        glm::vec2 texCoords;
    };

    VertexBuffer* quadVB;
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

        quadVB = new VertexBuffer(quadVertices.data(), quadVertices.size() * sizeof(FSQuadVertex));
        quadVB->bind();
        quadVB->addAttribute(0, 2, GL_FALSE, 4 * sizeof(float), 0);
        quadVB->addAttribute(1, 2, GL_FALSE, 4 * sizeof(float), 2 * sizeof(float));
    }

    ~FullScreenQuad() { }
};

#endif // FULL_SCREEN_QUAD_H
