#ifndef ENTITY_H
#define ENTITY_H

#include <vector>

#include <Buffer.h>
#include <Materials/Material.h>
#include <Culling/AABB.h>
#include <Culling/BoundingSphere.h>

namespace quasar {

class Node;
class Scene;
class Camera;

struct RenderStats {
    size_t trianglesDrawn = 0;
    size_t drawCalls = 0;

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
    AABB aabb;

    Entity() : ID(nextID++) {}
    Entity(const Material* material)
        : ID(nextID++)
        , material(material)
    {}

    int getID() const { return ID; }
    const Material* getMaterial() { return material; }

    virtual void bindMaterial(Scene& scene, Buffer& pointLightsUBO,
                              const Material* overrideMaterial = nullptr,
                              const Texture* prevIDMap = nullptr) = 0;

    virtual RenderStats draw(GLenum primitiveType, const Camera& camera, const glm::mat4& model,
                             bool frustumCull = true,
                             const Material* overrideMaterial = nullptr) = 0;
    virtual RenderStats draw(GLenum primitiveType, const Camera& camera, const glm::mat4& model,
                             const BoundingSphere& boundingSphere,
                             const Material* overrideMaterial = nullptr) = 0;

protected:
    uint32_t ID;
    static uint32_t nextID;

    const Material* material;
};

} // namespace quasar

#endif // ENTITY_H
