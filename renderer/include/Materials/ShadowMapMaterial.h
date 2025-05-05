#ifndef SHADOW_MAP_MATERIAL_H
#define SHADOW_MAP_MATERIAL_H

#include <Materials/Material.h>

namespace quasar {

class ShadowMapMaterial : public Material {
public:
    ShadowMapMaterial() = default;
    ~ShadowMapMaterial() = default;

    uint getTextureCount() const override { return 0; }
};

} // namespace quasar

#endif // SHADOW_MAP_MATERIAL_H
