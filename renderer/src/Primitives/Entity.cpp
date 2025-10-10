#include <Primitives/Entity.h>

using namespace quasar;

uint32_t Entity::nextID = 0;

Entity::Entity()
    : ID(nextID++)
    , name("Entity" + std::to_string(ID))
{}

Entity::Entity(const std::string& name)
    : ID(nextID++)
    , name(name)
{}

Entity::Entity(const Material* material)
    : ID(nextID++)
    , material(material)
    , name("Entity" + std::to_string(ID))
{}

Entity::Entity(const std::string& name, const Material* material)
    : ID(nextID++)
    , name(name)
    , material(material)
{}

