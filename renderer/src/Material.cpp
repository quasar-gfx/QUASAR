#include <Material.h>

void Material::bind(Shader &shader) {
    shader.setFloat("shininess", shininess);

    std::string name;
    for (int i = 0; i < textures.size(); i++) {
        glActiveTexture(GL_TEXTURE0 + i);
        switch(i) {
        case 0:
            name = "albedoMap";
            break;
        case 1:
            name = "specularMap";
            break;
        case 2:
            name = "normalMap";
            break;
        case 3:
            name = "metallicMap";
            break;
        case 4:
            name = "roughnessMap";
            break;
        case 5:
            name = "aoMap";
            break;
        default:
            break;
        }

        shader.setInt(name, i);
        glBindTexture(GL_TEXTURE_2D, textures[i]);
    }
}

void Material::unbind() {
    for (int i = 0; i < textures.size(); i++) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

void Material::cleanup() {
    for (auto &textureID : textures) {
        if (textureID == 0) continue;
        glDeleteTextures(1, &textureID);
    }
}
