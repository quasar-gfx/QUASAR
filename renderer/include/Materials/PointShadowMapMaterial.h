#ifndef POINT_SHADOW_MAP_MATERIAL_H
#define POINT_SHADOW_MAP_MATERIAL_H

#include <Materials/Material.h>

class PointShadowMapMaterial : public Material {
public:
    PointShadowMapMaterial();

    void bind() override;
    void unbind() override;

    void cleanup() override;
};

#endif // POINT_SHADOW_MAP_MATERIAL_H
