#include <Mesh.h>

void Mesh::draw(Shader &shader) {
    int diffuseIdx  = 1;
    int specularIdx = 1;
    int normalIdx   = 1;
    int heightIdx   = 1;

    for (int i = 0; i < textures.size(); i++) {
        std::string name;

        TextureType type = textures[i]->type;
        if (type == TEXTURE_DIFFUSE) {
            name = "texture_diffuse" + std::to_string(diffuseIdx++);
        }
        else if (type == TEXTURE_SPECULAR) {
            name = "texture_specular" + std::to_string(specularIdx++);
        }
        else if (type == TEXTURE_NORMAL) {
            name = "texture_normal" + std::to_string(normalIdx++);
        }
        else if (type == TEXTURE_HEIGHT) {
            name = "texture_height" + std::to_string(heightIdx++);
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

    if (textures.size() > 0) {
        textures[textures.size()-1]->unbind();
    }
}
