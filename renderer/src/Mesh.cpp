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
    shader.setFloat("shininess", shininess);

    std::string name;
    for (int i = 0; i < numTextures; i++) {
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
        if (i < textures.size()) {
            glBindTexture(GL_TEXTURE_2D, textures[i]);
        }
        else {
            glBindTexture(GL_TEXTURE_2D, 0);
        }
    }

    if (wireframe) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }

    GLenum primativeType = drawAsPointCloud ? GL_POINTS : GL_TRIANGLES;

    glBindVertexArray(VAO);
    if (indices.size() > 0) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glDrawElements(primativeType, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    }
    else {
        glDrawArrays(primativeType, 0, static_cast<unsigned int>(vertices.size()));
    }
    glBindVertexArray(0);

    for (int i = 0; i < textures.size(); i++) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    if (wireframe) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
}
