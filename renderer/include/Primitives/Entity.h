#ifndef ENTITY_H
#define ENTITY_H

#include <vector>

#include <Materials/Material.h>
#include <Culling/AABB.h>
#include <Culling/BoundingSphere.h>

namespace quasar {

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
    uint trianglesDrawn = 0;
    uint drawCalls = 0;

    RenderStats operator+ (const RenderStats& other) {
        RenderStats stats;
        stats.trianglesDrawn = trianglesDrawn + other.trianglesDrawn;
        stats.drawCalls = drawCalls + other.drawCalls;
        return stats;
    }
    RenderStats operator+= (const RenderStats& other) {
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

    Entity()
        : ID(nextID++)
        , aabb()
    {}

    int getID() const { return ID; }

    virtual void bindMaterial(Scene& scene, const glm::mat4& model,
                              const Material* overrideMaterial = nullptr,
                              const Texture* prevIDMap = nullptr) = 0;

    virtual RenderStats draw(GLenum primativeType, const Camera& camera, const glm::mat4& model,
                             bool frustumCull = true,
                             const Material* overrideMaterial = nullptr) = 0;
    virtual RenderStats draw(GLenum primativeType, const Camera& camera, const glm::mat4& model,
                             const BoundingSphere& boundingSphere,
                             const Material* overrideMaterial = nullptr) = 0;

    virtual void updateAnimations(float dt) {}

    virtual EntityType getType() const { return EntityType::EMPTY; }

protected:
    uint32_t ID;

    static uint32_t nextID;
};

} // namespace quasar

#endif // ENTITY_H
