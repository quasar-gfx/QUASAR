#ifndef SHADOW_MAP_MATERIAL_H
#define SHADOW_MAP_MATERIAL_H

#include <Materials/Material.h>

class ShadowMapMaterial : public Material {
public:
    ShadowMapMaterial() = default;
    ~ShadowMapMaterial() = default;

    unsigned int getTextureCount() const override { return 0; }
};

#endif // SHADOW_MAP_MATERIAL_H
