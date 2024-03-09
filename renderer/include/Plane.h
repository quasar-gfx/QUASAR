#ifndef PLANE_H
#define PLANE_H

#include <Mesh.h>

class Plane : public Mesh {
public:
    Plane(std::vector<TextureID> &textures, float shininess = 1.0f) : Mesh() {
        this->vertices = {
            {{ 1.0f, -1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {2.0f, 2.0f}, {1.0f, 0.0f, 0.0f}},
            {{-1.0f, -1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 2.0f}, {1.0f, 0.0f, 0.0f}},
            {{-1.0f, -1.0f,  1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},

            {{ 1.0f, -1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {2.0f, 2.0f}, {1.0f, 0.0f, 0.0f}},
            {{-1.0f, -1.0f,  1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},
            {{ 1.0f, -1.0f,  1.0f}, {0.0f, 1.0f, 0.0f}, {2.0f, 0.0f}, {1.0f, 0.0f, 0.0f}}
        };
        this->textures = textures;
        this->shininess = shininess;

        init();
    }

    ~Plane() {
        cleanup();
    }
};

#endif // PLANE_H
