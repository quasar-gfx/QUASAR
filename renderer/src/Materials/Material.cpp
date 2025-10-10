#include <Materials/Material.h>

using namespace quasar;

uint32_t Material::nextID = 0;

Material::Material()
    : ID(nextID++)
    , name("Material" + std::to_string(ID))
{}

Material::Material(Material::AlphaMode alphaMode)
    : ID(nextID++)
    , name("Material" + std::to_string(ID))
    , alphaMode(alphaMode)
{}

Material::Material(const std::string& name, Material::AlphaMode alphaMode)
    : ID(nextID++)
    , name(name)
    , alphaMode(alphaMode)
{}
