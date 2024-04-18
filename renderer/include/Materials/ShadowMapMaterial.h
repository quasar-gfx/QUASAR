#ifndef SHADOW_MAP_MATERIAL_H
#define SHADOW_MAP_MATERIAL_H

#include <Materials/Material.h>

class ShadowMapMaterial : public Material {
public:
    ShadowMapMaterial() = default;

    void bind() override;
};

#endif // SHADOW_MAP_MATERIAL_H
