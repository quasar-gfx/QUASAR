#ifndef SHADOW_MAP_MATERIAL_H
#define SHADOW_MAP_MATERIAL_H

#include <Materials/Material.h>

class ShadowMapMaterial : public Material {
public:
    explicit ShadowMapMaterial() = default;

    void bind() override;

    unsigned int getTextureCount() override { return 0; }
};

#endif // SHADOW_MAP_MATERIAL_H
