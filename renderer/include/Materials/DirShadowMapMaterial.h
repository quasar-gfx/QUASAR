#ifndef DIR_SHADOW_MAP_MATERIAL_H
#define DIR_SHADOW_MAP_MATERIAL_H

#include <Materials/ShadowMapMaterial.h>

class DirShadowMapMaterial : public ShadowMapMaterial {
public:
    explicit DirShadowMapMaterial();

    void bind() override;
};

#endif // DIR_SHADOW_MAP_MATERIAL_H
