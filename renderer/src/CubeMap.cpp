#include <iostream>

#include <stb_image.h>

#include <CubeMap.h>

void CubeMap::loadFromFiles(std::vector<std::string> faceFilePaths,
            GLenum format,
            GLint wrapS, GLint wrapT, GLint wrapR,
            GLint minFilter, GLint magFilter) {
    glGenTextures(1, &ID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, ID);

    int width, height, nrChannels;
    for (unsigned int i = 0; i < faceFilePaths.size(); i++) {
        unsigned char *data = stbi_load(faceFilePaths[i].c_str(), &width, &height, &nrChannels, 0);
        if (data) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                        0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data
            );
            stbi_image_free(data);
        }
        else {
            throw std::runtime_error("Cubemap tex failed to load at path: " + faceFilePaths[i]);
            stbi_image_free(data);
        }
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, wrapS);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, wrapT);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, wrapR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, magFilter);
}

