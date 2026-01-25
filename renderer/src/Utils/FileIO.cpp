#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <ImfRgbaFile.h>
#include <ImfArray.h>

#include <Utils/FileIO.h>

using namespace quasar;

#ifdef __ANDROID__
#include <android/log.h>

#define CHECK_ANDROID_ACTIVITY() if (activity == nullptr) { throw std::runtime_error("Android App Activity not set!"); }

ANativeActivity* FileIO::activity = nullptr;

void FileIO::registerIOSystem(ANativeActivity* activity) {
    FileIO::activity = activity;
}

std::string FileIO::copyFileToCache(std::string filename) {
    CHECK_ANDROID_ACTIVITY();

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
        AAsset_close(asset);
        throw std::runtime_error("Failed to create temp file: " + tempPath);
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

    std::string assetName = filename;
    if (!assetName.empty() && assetName[0] == '/')
        assetName.erase(0, 1);

    AAsset* file = AAssetManager_open(getAssetManager(), assetName.c_str(), AASSET_MODE_STREAMING);
    if (!file) {
        throw std::runtime_error("Could not open asset: " + filename);
    }
    std::ifstream::pos_type size = (std::ifstream::pos_type)AAsset_getLength(file);
    AAsset_close(file);
    return size;
#endif
}

std::string FileIO::loadFromTextFile(const std::string& filename, size_t* sizePtr) {
#ifndef __ANDROID__
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    if (sizePtr != nullptr) {
        file.seekg(0, std::ios::end);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        *sizePtr = (size_t)size;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return content;
#else
    CHECK_ANDROID_ACTIVITY();

    std::string assetName = filename;
    if (!assetName.empty() && assetName[0] == '/')
        assetName.erase(0, 1);

    AAsset* file = AAssetManager_open(getAssetManager(), assetName.c_str(), AASSET_MODE_STREAMING);
    if (!file) {
        throw std::runtime_error("Could not open asset: " + filename);
    }

    const off_t fileLength = AAsset_getLength(file);
    if (fileLength < 0) {
        AAsset_close(file);
        throw std::runtime_error("Invalid asset length: " + filename);
    }

    std::string text;
    text.resize((size_t)fileLength);

    size_t total = 0;
    while (total < (size_t)fileLength) {
        const int r = AAsset_read(file, (void*)(text.data() + total), (size_t)fileLength - total);
        if (r <= 0) break;
        total += (size_t)r;
    }

    AAsset_close(file);

    if (total != (size_t)fileLength) {
        throw std::runtime_error("Short read while reading asset: " + filename);
    }

    if (sizePtr != nullptr) {
        *sizePtr = (size_t)fileLength;
    }

    return text;
#endif
}

std::vector<char> FileIO::loadFromBinaryFile(const std::string& filename, size_t* sizePtr) {
#ifndef __ANDROID__
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (sizePtr != nullptr) {
        *sizePtr = (size_t)size;
    }

    std::vector<char> buffer((size_t)size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Could not read file: " + filename);
    }

    file.close();
    return buffer;
#else
    CHECK_ANDROID_ACTIVITY();

    std::string assetName = filename;
    if (!assetName.empty() && assetName[0] == '/')
        assetName.erase(0, 1);

    AAsset* file = AAssetManager_open(getAssetManager(), assetName.c_str(), AASSET_MODE_STREAMING);
    if (!file) {
        throw std::runtime_error("Could not open asset: " + filename);
    }

    const off_t fileLength = AAsset_getLength(file);
    if (fileLength < 0) {
        AAsset_close(file);
        throw std::runtime_error("Invalid asset length: " + filename);
    }

    std::vector<char> binary;
    binary.resize((size_t)fileLength);

    size_t total = 0;
    while (total < (size_t)fileLength) {
        const int r = AAsset_read(file, (void*)(binary.data() + total), (size_t)fileLength - total);
        if (r <= 0) break;
        total += (size_t)r;
    }

    AAsset_close(file);

    if (total != (size_t)fileLength) {
        throw std::runtime_error("Short read while reading asset: " + filename);
    }

    if (sizePtr != nullptr) {
        *sizePtr = (size_t)fileLength;
    }

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

    std::string assetName = filename;
    if (!assetName.empty() && assetName[0] == '/')
        assetName.erase(0, 1);

    AAsset* file = AAssetManager_open(getAssetManager(), assetName.c_str(), AASSET_MODE_STREAMING);
    if (!file) {
        throw std::runtime_error("Could not open asset: " + filename);
    }

    const off_t fileLength = AAsset_getLength(file);
    if (fileLength <= 0) {
        AAsset_close(file);
        throw std::runtime_error("Invalid asset length: " + filename);
    }

    std::vector<unsigned char> bytes;
    bytes.resize((size_t)fileLength);

    size_t total = 0;
    while (total < (size_t)fileLength) {
        const int r = AAsset_read(file, (void*)(bytes.data() + total), (size_t)fileLength - total);
        if (r <= 0) break;
        total += (size_t)r;
    }

    AAsset_close(file);

    if (total != (size_t)fileLength) {
        throw std::runtime_error("Short read while reading asset: " + filename);
    }

    unsigned char* data = stbi_load_from_memory(bytes.data(), (int)bytes.size(), width, height, channels, desiredChannels);
    if (!data) {
        const char* reason = stbi_failure_reason();
        throw std::runtime_error(std::string("Failed to load image: ") + filename + (reason ? (std::string(" (") + reason + ")") : ""));
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

float* FileIO::loadImageFromHDR(const std::string& filename, int* width, int* height, int* channels, int desiredChannels) {
#ifndef __ANDROID__
    float* data = stbi_loadf(filename.c_str(), width, height, channels, desiredChannels);
    if (!data) {
        throw std::runtime_error("Failed to load HDR image: " + filename);
    }
    return data;
#else
    CHECK_ANDROID_ACTIVITY();

    std::string assetName = filename;
    if (!assetName.empty() && assetName[0] == '/')
        assetName.erase(0, 1);

    AAsset* file = AAssetManager_open(getAssetManager(), assetName.c_str(), AASSET_MODE_STREAMING);
    if (!file) {
        throw std::runtime_error("Could not open asset: " + filename);
    }

    const off_t fileLength = AAsset_getLength(file);
    if (fileLength <= 0) {
        AAsset_close(file);
        throw std::runtime_error("Invalid asset length: " + filename);
    }

    std::vector<unsigned char> bytes;
    bytes.resize((size_t)fileLength);

    size_t total = 0;
    while (total < (size_t)fileLength) {
        const int r = AAsset_read(file, (void*)(bytes.data() + total), (size_t)fileLength - total);
        if (r <= 0) break;
        total += (size_t)r;
    }

    AAsset_close(file);

    if (total != (size_t)fileLength) {
        throw std::runtime_error("Short read while reading asset: " + filename);
    }

    float* data = stbi_loadf_from_memory(bytes.data(), (int)bytes.size(), width, height, channels, desiredChannels);
    if (!data) {
        const char* reason = stbi_failure_reason();
        throw std::runtime_error(std::string("Failed to load HDR image: ") + filename + (reason ? (std::string(" (") + reason + ")") : ""));
    }
    return data;
#endif
}

float* FileIO::loadImageFromEXR(const std::string& filename, int* width, int* height, int* channels) {
    try {
#ifdef __ANDROID__
        // OpenEXR expects a real filesystem path on Android, so copy the asset to cache first
        CHECK_ANDROID_ACTIVITY();
        std::string path = filename;
        if (filename.empty() || filename[0] != '/') {
            path = FileIO::copyFileToCache(filename);
        }
        Imf::RgbaInputFile file(path.c_str());
#else
        Imf::RgbaInputFile file(filename.c_str());
#endif
        Imath::Box2i dw = file.dataWindow();
        *width = dw.max.x - dw.min.x + 1;
        *height = dw.max.y - dw.min.y + 1;
        *channels = 4; // RGBA

        Imf::Array2D<Imf::Rgba> pixels(*height, *width);
        file.setFrameBuffer(&pixels[0][0], 1, *width);
        file.readPixels(dw.min.y, dw.max.y);

        float* data = (float*)malloc((size_t)(*width) * (size_t)(*height) * 4 * sizeof(float));
        for (int y = 0; y < *height; y++) {
            for (int x = 0; x < *width; x++) {
                const Imf::Rgba& p = pixels[*height - 1 - y][x];
                int idx = (y * *width + x) * 4;
                data[idx + 0] = p.r;
                data[idx + 1] = p.g;
                data[idx + 2] = p.b;
                data[idx + 3] = p.a;
            }
        }
        return data;
    }
    catch (const std::exception& e) {
        std::cerr << "Error reading EXR file " << filename << ": " << e.what() << std::endl;
        return nullptr;
    }
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

    // If filename is relative, write it under the app's internal data dir.
    std::string outPath = filename;
    if (filename.empty() || filename[0] != '/') {
        outPath = std::string(activity->internalDataPath) + "/" + filename;
    }

    std::ofstream file;
    if (append) {
        file.open(outPath, std::ios::app);
    }
    else {
        file.open(outPath);
    }

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

    // If filename is relative, write it under the app's internal data dir.
    std::string outPath = filename;
    if (filename.empty() || filename[0] != '/') {
        outPath = std::string(activity->internalDataPath) + "/" + filename;
    }

    std::ofstream file;
    if (append) {
        file.open(outPath, std::ios::app | std::ios::binary);
    }
    else {
        file.open(outPath, std::ios::binary);
    }

    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + outPath);
    }

    file.write(static_cast<const char*>(data), size);
    file.close();

    return size;
#endif
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

void FileIO::writeToEXR(const std::string& filename, int width, int height, int channels, const float *data) {
    try {
        Imf::RgbaOutputFile file(filename.c_str(), width, height, Imf::WRITE_RGBA);
        Imf::Array2D<Imf::Rgba> pixels(height, width);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = ((height - 1 - y) * width + x) * channels;
                Imf::Rgba &p = pixels[y][x];
                if (channels >= 1) p.r = data[idx + 0]; else p.r = 0;
                if (channels >= 2) p.g = data[idx + 1]; else p.g = p.r; // Grayscale
                if (channels >= 3) p.b = data[idx + 2]; else p.b = p.r; // Grayscale
                if (channels >= 4) p.a = data[idx + 3]; else p.a = 1.0f;
            }
        }

        file.setFrameBuffer(&pixels[0][0], 1, width);
        file.writePixels(height);
    }
    catch (const std::exception& e) {
        std::cerr << "Error writing EXR file " << filename << ": " << e.what() << std::endl;
        throw std::runtime_error("Failed to save EXR image: " + filename);
    }
}

void FileIO::freeImage(void* imageData) {
    stbi_image_free(imageData);
}
