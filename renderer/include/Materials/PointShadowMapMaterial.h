#ifndef POINT_SHADOW_MAP_MATERIAL_H
#define POINT_SHADOW_MAP_MATERIAL_H

#include <Materials/ShadowMapMaterial.h>

class PointShadowMapMaterial : public ShadowMapMaterial {
public:
    PointShadowMapMaterial();

    void bind() override;
};

#endif // POINT_SHADOW_MAP_MATERIAL_H
