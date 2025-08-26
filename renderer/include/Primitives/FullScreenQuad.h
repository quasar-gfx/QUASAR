#ifndef FULL_SCREEN_QUAD_H
#define FULL_SCREEN_QUAD_H

#include <vector>

#include <Buffer.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace quasar {

class FullScreenQuad {
public:
    Buffer vertexBuffer;

    FullScreenQuad()
        : vertexBuffer({
            .target = GL_ARRAY_BUFFER,
            .dataSize = sizeof(FSQuadVertex),
            .usage = GL_STATIC_DRAW,
        })
    {
        // Vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
        std::vector<FSQuadVertex> quadVertices = {
            // Bottom triangle
            { {-1.0f,  1.0f}, {0.0f, 1.0f} },
            { { 1.0f, -1.0f}, {1.0f, 0.0f} },
            { { 1.0f,  1.0f}, {1.0f, 1.0f} },

            // Top triangle
            { {-1.0f,  1.0f}, {0.0f, 1.0f} },
            { {-1.0f, -1.0f}, {0.0f, 0.0f} },
            { { 1.0f, -1.0f}, {1.0f, 0.0f} }
        };
        vertexBuffer.bind();
        vertexBuffer.setData(quadVertices.size(), quadVertices.data());

        setArrayBufferAttributes(FSQuadVertex::getVertexInputAttributes(), sizeof(FSQuadVertex));
    }
    ~FullScreenQuad() {
        glDeleteVertexArrays(1, &vertexArrayBuffer);
    }

    void setArrayBufferAttributes(const VertexInputAttributes& attributes, uint vertexSize) {
        glGenVertexArrays(1, &vertexArrayBuffer);
        glBindVertexArray(vertexArrayBuffer);

        if (attributes.size() == 0) {
            spdlog::warn("No vertex attributes provided!");
        }
        for (const auto& attribute : attributes) {
            glEnableVertexAttribArray(attribute.index);
            glVertexAttribPointer(attribute.index, attribute.size, attribute.type, attribute.normalized, vertexSize, (void*)attribute.pointer);
        }

        glBindVertexArray(0);
    }

    RenderStats draw() {
        RenderStats stats;
        stats.trianglesDrawn = 2;

        // Disable depth test so screen-space quad isn't discarded due to depth test.
        glDisable(GL_DEPTH_TEST);

        glBindVertexArray(vertexArrayBuffer);
        glDrawArrays(GL_TRIANGLES, 0, stats.trianglesDrawn * 3);
        glBindVertexArray(0);

        // Reenable depth test
        glEnable(GL_DEPTH_TEST);

        stats.drawCalls = 1;

        return stats;
    }

private:
    struct FSQuadVertex {
        glm::vec2 position;
        glm::vec2 texCoord;

        static const VertexInputAttributes getVertexInputAttributes() {
            return {
                {0, 2, GL_FLOAT, GL_FALSE, offsetof(FSQuadVertex, position)},
                {1, 2, GL_FLOAT, GL_FALSE, offsetof(FSQuadVertex, texCoord)},
            };
        }
    };

    GLuint vertexArrayBuffer;
};

} // namespace quasar

#endif // FULL_SCREEN_QUAD_H
