#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <Utils/FileIO.h>

using namespace quasar;

#ifdef __ANDROID__
#define CHECK_ANDROID_ACTIVITY() if (activity == nullptr) { throw std::runtime_error("Android App Activity not set!"); }

ANativeActivity* FileIO::activity = nullptr;

void FileIO::registerIOSystem(ANativeActivity* activity) {
    FileIO::activity = activity;
}

std::string FileIO::copyFileToCache(std::string filename) {
    AAsset* asset = AAssetManager_open(getAssetManager(), filename.c_str(), AASSET_MODE_STREAMING);
    if (!asset) {
        throw std::runtime_error("Failed to open file " + filename);
        return "";
    }

    std::string internalAppPath = activity->internalDataPath;
    // Remove "files/" from end of path
    internalAppPath = internalAppPath.substr(0, internalAppPath.find_last_of('/'));
    internalAppPath += "/cache/";
    std::string tempPath = internalAppPath + filename;

    std::ofstream outFile(tempPath, std::ios::binary);
    if (!outFile) {
        throw std::runtime_error("Failed to create temp file: " + tempPath);
        AAsset_close(asset);
        return "";
    }

    char buffer[1024];
    int bytesRead;
    while ((bytesRead = AAsset_read(asset, buffer, sizeof(buffer))) > 0) {
        outFile.write(buffer, bytesRead);
    }

    AAsset_close(asset);
    outFile.close();
    return tempPath;
}
#endif

void FileIO::flipVerticallyOnLoad(bool flip) {
    stbi_set_flip_vertically_on_load(flip);
}

void FileIO::flipVerticallyOnWrite(bool flip) {
    stbi_flip_vertically_on_write(flip);
}

std::ifstream::pos_type FileIO::getFileSize(const std::string& filename) {
#ifndef __ANDROID__
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::ifstream::pos_type size = file.tellg();
    file.close();
    return size;
#else
    CHECK_ANDROID_ACTIVITY();

    AAsset *file = AAssetManager_open(getAssetManager(), filename.c_str(), AASSET_MODE_BUFFER);
    std::ifstream::pos_type size = AAsset_getLength(file);
    AAsset_close(file);
    return size;
#endif
}

std::string FileIO::loadFromTextFile(const std::string& filename, uint* sizePtr) {
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
    CHECK_ANDROID_ACTIVITY();

    AAsset *file = AAssetManager_open(getAssetManager(), filename.c_str(), AASSET_MODE_BUFFER);
    size_t fileLength = AAsset_getLength(file);
    std::string text;
    text.resize(fileLength);
    AAsset_read(file, (void *)text.data(), fileLength);
    AAsset_close(file);
    return text;
#endif
}

std::vector<char> FileIO::loadFromBinaryFile(const std::string& filename, uint* sizePtr) {
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
    CHECK_ANDROID_ACTIVITY();

    AAsset *file = AAssetManager_open(getAssetManager(), filename.c_str(), AASSET_MODE_BUFFER);
    size_t fileLength = AAsset_getLength(file);
    std::vector<char> binary(fileLength);
    AAsset_read(file, (void *)binary.data(), fileLength);
    AAsset_close(file);
    return binary;
#endif
}

unsigned char* FileIO::loadImage(const std::string& filename, int* width, int* height, int* channels, int desiredChannels) {
#ifndef __ANDROID__
    unsigned char* data = stbi_load(filename.c_str(), width, height, channels, desiredChannels);
    if (!data) {
        throw std::runtime_error("Failed to load image: " + filename);
    }
    return data;
#else
    CHECK_ANDROID_ACTIVITY();

    AAsset *file = AAssetManager_open(getAssetManager(), filename.c_str(), AASSET_MODE_BUFFER);
    size_t fileLength = AAsset_getLength(file);
    unsigned char* data = stbi_load_from_memory((unsigned char*)AAsset_getBuffer(file), fileLength, width, height, channels, desiredChannels);
    AAsset_close(file);
    if (!data) {
        throw std::runtime_error("Failed to load image: " + filename);
    }
    return data;
#endif
}

unsigned char* FileIO::loadImageFromMemory(const unsigned char* data, int size, int* width, int* height, int* channels, int desiredChannels) {
    unsigned char* imageData = stbi_load_from_memory(data, size, width, height, channels, desiredChannels);
    if (!imageData) {
        throw std::runtime_error("Failed to load image from memory");
    }
    return imageData;
}

float* FileIO::loadImageHDR(const std::string& filename, int* width, int* height, int* channels, int desiredChannels) {
#ifndef __ANDROID__
    float* data = stbi_loadf(filename.c_str(), width, height, channels, desiredChannels);
    if (!data) {
        throw std::runtime_error("Failed to load HDR image: " + filename);
    }
    return data;
#else
    CHECK_ANDROID_ACTIVITY();

    AAsset *file = AAssetManager_open(getAssetManager(), filename.c_str(), AASSET_MODE_BUFFER);
    size_t fileLength = AAsset_getLength(file);
    float* data = stbi_loadf_from_memory((unsigned char*)AAsset_getBuffer(file), fileLength, width, height, channels, desiredChannels);
    AAsset_close(file);
    if (!data) {
        throw std::runtime_error("Failed to load HDR image: " + filename);
    }
    return data;
#endif
}

size_t FileIO::writeToTextFile(const std::string& filename, const std::string& data, bool append) {
#ifndef __ANDROID__
    std::ofstream file;
    if (append) {
        file.open(filename, std::ios::app);
    }
    else {
        file.open(filename);
    }
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    file << data;
    file.close();
    return data.size();
#else
    CHECK_ANDROID_ACTIVITY();

    // If 'filename' is relative, write it under the app's internal data dir.
    std::string outPath = filename;
    if (filename.empty() || filename[0] != '/') {
        outPath = std::string(activity->internalDataPath) + "/" + filename;
    }

    std::ofstream file(outPath);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + outPath);
    }

    file << data;
    file.close();

    return data.size();
#endif
}

size_t FileIO::writeToBinaryFile(const std::string& filename, const void* data, size_t size, bool append) {
#ifndef __ANDROID__
    std::ofstream file;
    if (append) {
        file.open(filename, std::ios::app | std::ios::binary);
    }
    else {
        file.open(filename, std::ios::binary);
    }
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    file.write(static_cast<const char*>(data), size);
    file.close();
    return size;
#else
    CHECK_ANDROID_ACTIVITY();

    // If 'filename' is relative, write it under the app's internal data dir.
    std::string outPath = filename;
    if (filename.empty() || filename[0] != '/') {
        outPath = std::string(activity->internalDataPath) + "/" + filename;
    }

    std::ofstream file(outPath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + outPath);
    }

    file.write(static_cast<const char*>(data), size);
    file.close();

    return size;
#endif
}

void FileIO::writeToPNG(const std::string& filename, int width, int height, int channels, const void *data) {
    if (!stbi_write_png(filename.c_str(), width, height, channels, data, width * channels)) {
        throw std::runtime_error("Failed to save PNG image: " + filename);
    }
}

void FileIO::writeToJPG(const std::string& filename, int width, int height, int channels, const void *data, int quality) {
    if (!stbi_write_jpg(filename.c_str(), width, height, channels, data, quality)) {
        throw std::runtime_error("Failed to save JPG image: " + filename);
    }
}

void FileIO::writeToHDR(const std::string& filename, int width, int height, int channels, const float *data) {
    if (!stbi_write_hdr(filename.c_str(), width, height, channels, data)) {
        throw std::runtime_error("Failed to save HDR image: " + filename);
    }
}

size_t FileIO::writeJPGToMemory(std::vector<unsigned char>& outputData, int width, int height, int channels, const void *data, int quality) {
    auto write_func = [](void* context, void* d, int s) {
        MemBuffer* mb = static_cast<MemBuffer*>(context);
        if (mb->size + (size_t)s > mb->cap) {
            size_t new_cap = mb->cap ? mb->cap * 2 : 64;
            while (new_cap < mb->size + (size_t)s) new_cap *= 2;
            unsigned char* nd = (unsigned char*)realloc(mb->data, new_cap);
            if (!nd) return;
            mb->data = nd;
            mb->cap = new_cap;
        }
        memcpy(mb->data + mb->size, d, (size_t)s);
        mb->size += (size_t)s;
    };

    MemBuffer mb{};
    int ok = stbi_write_jpg_to_func(write_func, &mb, width, height, channels, data, quality);
    if (!ok || mb.size == 0) {
        if (mb.data) {
            free(mb.data);
        }
        throw std::runtime_error("Failed to write JPG to memory");
    }

    outputData.resize(mb.size);
    outputData.assign(mb.data, mb.data + mb.size);

    free(mb.data);
    return outputData.size();
}

void FileIO::freeImage(void* imageData) {
    stbi_image_free(imageData);
}
