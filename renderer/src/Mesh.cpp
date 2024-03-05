#include <Mesh.h>

void Mesh::init()  {
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(ATTRIBUTE_POSITION);
    glVertexAttribPointer(ATTRIBUTE_POSITION,   3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

    glEnableVertexAttribArray(ATTRIBUTE_NORMAL);
    glVertexAttribPointer(ATTRIBUTE_NORMAL,     3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));

    glEnableVertexAttribArray(ATTRIBUTE_TEX_COORDS);
    glVertexAttribPointer(ATTRIBUTE_TEX_COORDS, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));

    glEnableVertexAttribArray(ATTRIBUTE_TANGENT);
    glVertexAttribPointer(ATTRIBUTE_TANGENT,    3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, tangent));

    glBindVertexArray(0);
}

void Mesh::draw(Shader &shader) {
    shader.setFloat("material.shininess", shininess);

    for (int i = 0; i < textures.size(); i++) {
        if (textures[i] == 0) continue;

        std::string name;

        if (i == TEXTURE_DIFFUSE) {
            name = "material.diffuse";
        }
        else if (i == TEXTURE_SPECULAR) {
            name = "material.specular";
        }
        else if (i == TEXTURE_NORMAL) {
            name = "material.normal";
        }
        else if (i == TEXTURE_HEIGHT) {
            name = "material.height";
        }

        glActiveTexture(GL_TEXTURE0 + i);
        shader.setInt(name.c_str(), i);
        glBindTexture(GL_TEXTURE_2D, textures[i]);
    }

    glBindVertexArray(VAO);
    if (indices.size() > 0) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    }
    else {
        glDrawArrays(GL_TRIANGLES, 0, static_cast<unsigned int>(vertices.size()));
    }
    glBindVertexArray(0);

    for (int i = 0; i < textures.size(); i++) {
        if (textures[i] == 0) continue;

        std::string name;

        if (i == TEXTURE_DIFFUSE) {
            name = "material.diffuse";
        }
        else if (i == TEXTURE_SPECULAR) {
            name = "material.specular";
        }
        else if (i == TEXTURE_NORMAL) {
            name = "material.normal";
        }
        else if (i == TEXTURE_HEIGHT) {
            name = "material.height";
        }

        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}
