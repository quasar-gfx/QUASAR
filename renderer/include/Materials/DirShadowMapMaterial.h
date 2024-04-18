#ifndef DIR_SHADOW_MAP_MATERIAL_H
#define DIR_SHADOW_MAP_MATERIAL_H

#include <Materials/Material.h>

class DirShadowMapMaterial : public Material {
public:
    DirShadowMapMaterial();

    void bind() override;
    void unbind() override;

    void cleanup() override;
};

#endif // DIR_SHADOW_MAP_MATERIAL_H
