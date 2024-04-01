#ifndef SPHERE_H
#define SPHERE_H

#include <Mesh.h>

class Sphere : public Mesh {
public:
    explicit Sphere(Material &material, unsigned int xSegments = 64, unsigned int ySegments = 64) : Mesh() {
        float radius = 1.0f;
        const float PI = 3.14159265359f;

        for (int i = 0; i < xSegments; ++i) {
            for (int j = 0; j < ySegments; ++j) {
                int first = (i * (ySegments + 1)) + j;
                int second = first + ySegments + 1;

                this->indices.push_back(first);
                this->indices.push_back(second);
                this->indices.push_back(first + 1);

                this->indices.push_back(second);
                this->indices.push_back(second + 1);
                this->indices.push_back(first + 1);
            }
        }

        for (int i = 0; i <= xSegments; ++i) {
            float phi = M_PI * static_cast<float>(i) / xSegments;
            float cosPhi = cos(phi);
            float sinPhi = sin(phi);

            for (int j = 0; j <= ySegments; ++j) {
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

                this->vertices.push_back(vertex);
            }
        }
        this->material = material;

        init();
    }

    ~Sphere() {
        cleanup();
    }
};

#endif // SPHERE_H
