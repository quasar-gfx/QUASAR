#include <Mesh.h>

void Mesh::draw(Shader &shader) {
    shader.setFloat("material.shininess", shininess);

    for (int i = 0; i < textures.size(); i++) {
        std::string name;

        TextureType type = textures[i]->type;
        if (type == TEXTURE_DIFFUSE) {
            name = "material.diffuse";
        }
        else if (type == TEXTURE_SPECULAR) {
            name = "material.specular";
        }
        else if (type == TEXTURE_NORMAL) {
            name = "material.normal";
        }
        else if (type == TEXTURE_HEIGHT) {
            name = "material.height";
        }

        shader.setInt(name.c_str(), i);
        textures[i]->bind(i);
    }

    vbo.bind();
    if (indices.size() > 0) {
        ibo.bind();
        glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);
        ibo.unbind();
    }
    else {
        glDrawArrays(GL_TRIANGLES, 0, static_cast<unsigned int>(vertices.size()));
    }
    vbo.unbind();

    for (int i = 0; i < textures.size(); i++) {
        std::string name;

        TextureType type = textures[i]->type;
        if (type == TEXTURE_DIFFUSE) {
            name = "material.diffuse";
        }
        else if (type == TEXTURE_SPECULAR) {
            name = "material.specular";
        }
        else if (type == TEXTURE_NORMAL) {
            name = "material.normal";
        }
        else if (type == TEXTURE_HEIGHT) {
            name = "material.height";
        }

        textures[i]->unbind(i);
    }
}
