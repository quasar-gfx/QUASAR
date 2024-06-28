#ifndef FULL_SCREEN_QUAD_H
#define FULL_SCREEN_QUAD_H

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>

class FullScreenQuad {
public:
    explicit FullScreenQuad() {
        // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
        std::vector<FSQuadVertex> quadVertices = {
            { {-1.0f, 1.0f, 0.0f}, {0.0f, 1.0f} },
            { {-1.0f, -1.0f, 0.0f}, {0.0f, 0.0f} },
            { {1.0f, -1.0f, 0.0f}, {1.0f, 0.0f} },

            { {-1.0f, 1.0f, 0.0f}, {0.0f, 1.0f} },
            { {1.0f, -1.0f, 0.0f}, {1.0f, 0.0f} },
            { {1.0f, 1.0f, 0.0f}, {1.0f, 1.0f} }
        };

        glGenVertexArrays(1, &vertexArrayBuffer);
        glGenBuffers(1, &vertexBuffer);

        glBindVertexArray(vertexArrayBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);

        glBufferData(GL_ARRAY_BUFFER, quadVertices.size() * sizeof(FSQuadVertex), quadVertices.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(FSQuadVertex), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(FSQuadVertex), (void*)offsetof(FSQuadVertex, texCoords));

        glBindVertexArray(0);
    }

    ~FullScreenQuad() {
        cleanup();
    };

    void draw() {
        // disable depth test so screen-space quad isn't discarded due to depth test.
        glDisable(GL_DEPTH_TEST);

        glBindVertexArray(vertexArrayBuffer);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);

        // reenable depth test
        glEnable(GL_DEPTH_TEST);
    }

    void cleanup() {
        glDeleteVertexArrays(1, &vertexArrayBuffer);
        glDeleteBuffers(1, &vertexBuffer);
    }

private:
    struct FSQuadVertex {
        glm::vec3 position;
        glm::vec2 texCoords;
    };

    GLuint vertexArrayBuffer;
    GLuint vertexBuffer;
};

#endif // FULL_SCREEN_QUAD_H
