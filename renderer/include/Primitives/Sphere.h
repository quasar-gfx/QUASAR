#ifndef SPHERE_H
#define SPHERE_H

#include <Primitives/Mesh.h>

namespace quasar {

class Sphere : public Mesh {
public:
    float radius = 1.0f;

    Sphere(const MeshDataCreateParams &params, uint xSegments = 64, uint ySegments = 64, float radius = 1.0f) : Mesh(params), radius(radius) {
        std::vector<Vertex> vertices;
        std::vector<uint> indices;

        for (int i = 0; i <= xSegments; i++) {
            float phi = M_PI * static_cast<float>(i) / xSegments;
            float cosPhi = cos(phi);
            float sinPhi = sin(phi);

            for (int j = 0; j <= ySegments; j++) {
                float theta = 2 * M_PI * static_cast<float>(j) / ySegments;
                float cosTheta = cos(theta);
                float sinTheta = sin(theta);

                Vertex vertex;
                vertex.position[0] = radius * sinPhi * cosTheta;
                vertex.position[1] = radius * cosPhi;
                vertex.position[2] = radius * sinPhi * sinTheta;

                vertex.normal[0] = sinPhi * cosTheta;
                vertex.normal[1] = cosPhi;
                vertex.normal[2] = sinPhi * sinTheta;

                vertex.texCoords[0] = static_cast<float>(j) / ySegments;
                vertex.texCoords[1] = static_cast<float>(i) / xSegments;

                // Compute tangent (approximation)
                vertex.tangent[0] = -radius * sinTheta;
                vertex.tangent[1] = 0;
                vertex.tangent[2] = radius * cosTheta;

                // Compute bitangent (approximation)
                vertex.bitangent[0] = radius * cosPhi * cosTheta;
                vertex.bitangent[1] = -radius * sinPhi;
                vertex.bitangent[2] = radius * cosPhi * sinTheta;

                vertices.push_back(vertex);
            }
        }

        for (int i = 0; i < xSegments; i++) {
            for (int j = 0; j < ySegments; j++) {
                int first = (i * (ySegments + 1)) + j;
                int second = first + ySegments + 1;

                // Reversed order of indices
                indices.push_back(first);
                indices.push_back(first + 1);
                indices.push_back(second);

                indices.push_back(second);
                indices.push_back(first + 1);
                indices.push_back(second + 1);
            }
        }

        setBuffers(vertices.data(), vertices.size(), indices.data(), indices.size());
    }
};

} // namespace quasar

#endif // SPHERE_H
