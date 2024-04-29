#ifndef PLANE_H
#define PLANE_H

#include <Primatives/Mesh.h>

class Plane : public Mesh {
public:
    explicit Plane(Material* material) : Mesh() {
        this->vertices = {
            {{ 1.0f, -1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {2.0f, 2.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
            {{-1.0f, -1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 2.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
            {{-1.0f, -1.0f,  1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},

            {{ 1.0f, -1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {2.0f, 2.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
            {{-1.0f, -1.0f,  1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
            {{ 1.0f, -1.0f,  1.0f}, {0.0f, 1.0f, 0.0f}, {2.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}}
        };
        this->material = material;

        init();
    }

    ~Plane() {
        cleanup();
    }
};

#endif // PLANE_H
