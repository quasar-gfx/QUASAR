#ifndef PLANE_H
#define PLANE_H

#include <Primatives/Mesh.h>

class Plane : public Mesh {
public:
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    Plane(const MeshCreateParams &params, bool twoSided = true) : Mesh(params) {
        if (twoSided) {
            vertices = {
                // Top face (facing up)
                { {-1.0f, 0.0f,  1.0f}, { 0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f} },
                { {-1.0f, 0.0f, -1.0f}, { 0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f} },
                { { 1.0f, 0.0f, -1.0f}, { 0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f} },
                { { 1.0f, 0.0f,  1.0f}, { 0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f} },

                // Bottom face (facing down)
                { {-1.0f, 0.0f,  1.0f}, { 0.0f, -1.0f, 0.0}, {1.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f} },
                { { 1.0f, 0.0f,  1.0f}, { 0.0f, -1.0f, 0.0}, {0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f} },
                { { 1.0f, 0.0f, -1.0f}, { 0.0f, -1.0f, 0.0}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f} },
                { {-1.0f, 0.0f, -1.0f}, { 0.0f, -1.0f, 0.0}, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f} }
            };

            indices = {
                // Front face
                0, 1, 2,
                2, 3, 0,

                // Back face
                4, 5, 6,
                6, 7, 4,
            };
            setBuffers(vertices, indices);
        }
        else {
            vertices = {
                { {-1.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f} }, // Bottom Left
                { { 1.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f} }, // Bottom Right
                { { 1.0f, 0.0f,  1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f} }, // Top Right
                { { 1.0f, 0.0f,  1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f} }, // Top Right
                { {-1.0f, 0.0f,  1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f} }, // Top Left
                { {-1.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f} }  // Bottom Left
            };
            setBuffers(vertices);
        }
    }
};

#endif // PLANE_H
