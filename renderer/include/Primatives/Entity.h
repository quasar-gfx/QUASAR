#ifndef ENTITY_H
#define ENTITY_H

#include <vector>

#include <Materials/Material.h>
#include <Culling/AABB.h>
#include <Culling/BoundingSphere.h>

class Node;
class Scene;
class Camera;

enum class EntityType {
    EMPTY,
    MESH,
    MODEL,
    FULL_SCREEN_QUAD
};

struct RenderStats {
    unsigned int trianglesDrawn = 0;
    unsigned int drawCalls = 0;

    RenderStats operator+ (const RenderStats &other) {
        RenderStats stats;
        stats.trianglesDrawn = trianglesDrawn + other.trianglesDrawn;
        stats.drawCalls = drawCalls + other.drawCalls;
        return stats;
    }
    RenderStats operator+= (const RenderStats &other) {
        trianglesDrawn += other.trianglesDrawn;
        drawCalls += other.drawCalls;
        return *this;
    }

    void reset() {
        trianglesDrawn = 0;
        drawCalls = 0;
    }
};

class Entity {
public:
    Node* parentNode = nullptr;

    AABB aabb;

    Entity() : ID(nextID++), aabb() {}

    int getID() const { return ID; }

    virtual void bindMaterial(const Scene &scene, const glm::mat4 &model,
                              const Material* overrideMaterial = nullptr,
                              const Texture* prevDepthMap = nullptr) = 0;

    virtual RenderStats draw(const Camera &camera, const glm::mat4 &model,
                             bool frustumCull = true,
                             const Material* overrideMaterial = nullptr) = 0;
    virtual RenderStats draw(const Camera &camera, const glm::mat4 &model,
                             const BoundingSphere &boundingSphere,
                             const Material* overrideMaterial = nullptr) = 0;

    virtual EntityType getType() const { return EntityType::EMPTY; }

private:
    unsigned int ID;

    static unsigned int nextID;
};

#endif // ENTITY_H
