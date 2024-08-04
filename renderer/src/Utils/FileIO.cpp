#include <Utils/FileIO.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#ifdef __ANDROID__
#define CHECK_ASSET_MANAGER() if (assetManager == nullptr) { throw std::runtime_error("Asset manager not set!"); }
#endif

std::string FileIO::loadTextFile(const std::string &filename, unsigned int* sizePtr) {
#ifndef __ANDROID__
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    if (sizePtr != nullptr) {
        file.seekg(0, std::ios::end);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        *sizePtr = size;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return content;
#else
    CHECK_ASSET_MANAGER();

    AAsset *file = AAssetManager_open(assetManager, filename.c_str(), AASSET_MODE_BUFFER);
    size_t fileLength = AAsset_getLength(file);
    std::string text;
    text.resize(fileLength);
    AAsset_read(file, (void *)text.data(), fileLength);
    AAsset_close(file);
    return text;
#endif
}

std::vector<char> FileIO::loadBinaryFile(const std::string &filename, unsigned int* sizePtr) {
#ifndef __ANDROID__
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (sizePtr != nullptr) {
        *sizePtr = size;
    }

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Could not read file: " + filename);
    }

    file.close();
    return buffer;
#else
    CHECK_ASSET_MANAGER();

    AAsset *file = AAssetManager_open(assetManager, filename.c_str(), AASSET_MODE_BUFFER);
    size_t fileLength = AAsset_getLength(file);
    std::vector<char> binary(fileLength);
    AAsset_read(file, (void *)binary.data(), fileLength);
    AAsset_close(file);
    return binary;
#endif
}

std::ifstream::pos_type FileIO::getFileSize(const std::string &filename) {
#ifndef __ANDROID__
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::ifstream::pos_type size = file.tellg();
    file.close();
    return size;
#else
    CHECK_ASSET_MANAGER();

    AAsset *file = AAssetManager_open(assetManager, filename.c_str(), AASSET_MODE_BUFFER);
    std::ifstream::pos_type size = AAsset_getLength(file);
    AAsset_close(file);
    return size;
#endif
}

void FileIO::flipVerticallyOnLoad(bool flip) {
    stbi_set_flip_vertically_on_load(flip);
}

void FileIO::flipVerticallyOnWrite(bool flip) {
    stbi_flip_vertically_on_write(flip);
}

unsigned char* FileIO::loadImage(const std::string &filename, int* width, int* height, int* channels, int desiredChannels) {
    unsigned char* data = stbi_load(filename.c_str(), width, height, channels, desiredChannels);
    if (!data) {
        throw std::runtime_error("Failed to load image: " + filename);
    }
    return data;
}

unsigned char* FileIO::loadImageFromMemory(const unsigned char* data, int size, int* width, int* height, int* channels, int desiredChannels) {
    unsigned char* imageData = stbi_load_from_memory(data, size, width, height, channels, desiredChannels);
    if (!imageData) {
        throw std::runtime_error("Failed to load image from memory");
    }
    return imageData;
}

float* FileIO::loadImageHDR(const std::string &filename, int* width, int* height, int* channels, int desiredChannels) {
    float* data = stbi_loadf(filename.c_str(), width, height, channels, desiredChannels);
    if (!data) {
        throw std::runtime_error("Failed to load HDR image: " + filename);
    }
    return data;
}

void FileIO::freeImage(void* imageData) {
    stbi_image_free(imageData);
}

void FileIO::saveAsPNG(const std::string &filename, int width, int height, int channels, const void *data) {
    if (!stbi_write_png(filename.c_str(), width, height, channels, data, width * channels)) {
        throw std::runtime_error("Failed to save PNG image: " + filename);
    }
}

void FileIO::saveAsJPG(const std::string &filename, int width, int height, int channels, const void *data, int quality) {
    if (!stbi_write_jpg(filename.c_str(), width, height, channels, data, quality)) {
        throw std::runtime_error("Failed to save JPG image: " + filename);
    }
}

void FileIO::saveAsHDR(const std::string &filename, int width, int height, int channels, const float *data) {
    if (!stbi_write_hdr(filename.c_str(), width, height, channels, data)) {
        throw std::runtime_error("Failed to save HDR image: " + filename);
    }
}

#ifdef __ANDROID__
AAssetManager* FileIO::assetManager = nullptr;

void FileIO::registerAssetManager(AAssetManager* assetManager) {
    FileIO::assetManager = assetManager;
}
#endif
