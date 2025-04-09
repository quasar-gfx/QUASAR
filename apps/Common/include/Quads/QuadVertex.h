#ifndef QUADS_VERTEX_H
#define QUADS_VERTEX_H

namespace quasar {

struct QuadVertex {
    alignas(16) glm::vec3 position;
    alignas(16) glm::vec3 texCoords3D;

    static const VertexInputAttributes getVertexInputAttributes() {
        return {
            {0, 3, GL_FLOAT, GL_FALSE, offsetof(QuadVertex, position)},
            {1, 3, GL_FLOAT, GL_FALSE, offsetof(QuadVertex, texCoords3D)},
        };
    }
};

} // namespace quasar

#endif // QUADS_VERTEX_H
