#ifndef FULL_SCREEN_QUAD_H
#define FULL_SCREEN_QUAD_H

#include <vector>

#include <Buffer.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class FullScreenQuad {
private:
    struct FSQuadVertex {
        glm::vec2 position;
        glm::vec2 texCoords;
    };

public:
    Buffer vertexBuffer;

    FullScreenQuad() : vertexBuffer(GL_ARRAY_BUFFER, sizeof(FSQuadVertex), GL_STATIC_DRAW) {
        // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
        std::vector<FSQuadVertex> quadVertices = {
            // bottom triangle
            { {-1.0f,  1.0f}, {0.0f, 1.0f} },
            { { 1.0f, -1.0f}, {1.0f, 0.0f} },
            { { 1.0f,  1.0f}, {1.0f, 1.0f} },

            // top triangle
            { {-1.0f,  1.0f}, {0.0f, 1.0f} },
            { {-1.0f, -1.0f}, {0.0f, 0.0f} },
            { { 1.0f, -1.0f}, {1.0f, 0.0f} }
        };

        glGenVertexArrays(1, &vertexArrayBuffer);

        glBindVertexArray(vertexArrayBuffer);

        vertexBuffer.bind();
        vertexBuffer.setData(quadVertices.size(), quadVertices.data());

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(FSQuadVertex), (void*)offsetof(FSQuadVertex, position));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(FSQuadVertex), (void*)offsetof(FSQuadVertex, texCoords));

        glBindVertexArray(0);
    }

    ~FullScreenQuad() {
        glDeleteVertexArrays(1, &vertexArrayBuffer);
    };

    RenderStats draw() {
        RenderStats stats;
        stats.trianglesDrawn = 2;

        // disable depth test so screen-space quad isn't discarded due to depth test.
        glDisable(GL_DEPTH_TEST);

        glBindVertexArray(vertexArrayBuffer);
        glDrawArrays(GL_TRIANGLES, 0, stats.trianglesDrawn * 3);
        glBindVertexArray(0);

        // reenable depth test
        glEnable(GL_DEPTH_TEST);

        stats.drawCalls = 1;

        return stats;
    }

private:
    GLuint vertexArrayBuffer;
};

#endif // FULL_SCREEN_QUAD_H
