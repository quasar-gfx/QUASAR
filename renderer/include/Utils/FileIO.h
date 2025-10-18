#ifndef FILE_IO_H
#define FILE_IO_H

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

#ifdef __ANDROID__
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/native_activity.h>
#endif

namespace quasar {

class FileIO {
public:
    static void flipVerticallyOnLoad(bool flip);
    static void flipVerticallyOnWrite(bool flip);

    static std::ifstream::pos_type getFileSize(const std::string& filename);

    static std::string loadFromTextFile(const std::string& filename, uint* sizePtr = nullptr);
    static std::vector<char> loadFromBinaryFile(const std::string& filename, uint* sizePtr = nullptr);
    static unsigned char* loadImage(const std::string& filename, int* width, int* height, int* channels, int desiredChannels = 0);
    static unsigned char* loadImageFromMemory(const unsigned char* data, int size, int* width, int* height, int* channels, int desiredChannels = 0);
    static float* loadImageHDR(const std::string& filename, int* width, int* height, int* channels, int desiredChannels = 0);

    static size_t writeToTextFile(const std::string& filename, const std::string& data, bool append = false);
    static size_t writeToBinaryFile(const std::string& filename, const void* data, size_t size, bool append = false);
    static void writeToPNG(const std::string& filename, int width, int height, int channels, const void *data);
    static void writeToJPG(const std::string& filename, int width, int height, int channels, const void *data, int quality = 85);
    static void writeToHDR(const std::string& filename, int width, int height, int channels, const float *data);
    static size_t writeJPGToMemory(std::vector<unsigned char>& outputData, int width, int height, int channels, const void *data, int quality = 85);

    static void freeImage(void* imageData);

#ifdef __ANDROID__
    static void registerIOSystem(ANativeActivity* activity);
    static ANativeActivity* getNativeActivity() {
        return activity;
    }
    static AAssetManager* getAssetManager() {
        return activity->assetManager;
    }

    static std::string copyFileToCache(std::string filename);
#endif

private:
    struct MemBuffer {
        unsigned char* data;
        size_t size;
        size_t cap;
    };

#ifdef __ANDROID__
    static ANativeActivity* activity;
#endif
};

} // namespace quasar

#endif // FILE_IO_H
