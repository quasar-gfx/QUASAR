#ifndef PATH_H
#define PATH_H

#include <string>
#include <iostream>
#if __cplusplus >= 201703L
    #include <filesystem>
    namespace fs = std::filesystem;
#else
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
#endif

namespace quasar {

class Path {
public:
    Path(const std::string& p) : path(p) {}
    Path(const fs::path& p) : path(p) {}

    Path operator/(const std::string& other) const {
        return Path(path / other);
    }

    Path operator/(const Path& other) const {
        return Path(path / other.path);
    }

    operator std::string() const {
        return path.string();
    }

    std::string str() const {
        return path.string();
    }

    const char* c_str() const {
        return path.c_str();
    }

    std::string absolutePathStr() const {
        return fs::absolute(path).string();
    }

    bool exists() const {
        return fs::exists(path);
    }

    bool isFile() const {
        return fs::is_regular_file(path);
    }

    bool isDir() const {
        return fs::is_directory(path);
    }

    std::string name() const {
        return path.filename().string();
    }

    Path parent() const {
        return Path(path.parent_path());
    }

    bool mkdir() const {
        return fs::create_directory(path);
    }

    bool mkdirRecursive() const {
        return fs::create_directories(path);
    }

    bool createParentDirs() const {
        return fs::create_directories(path.parent_path());
    }

    bool remove() const {
        return fs::remove_all(path) > 0;
    }

    const fs::path& get() const {
        return path;
    }

    Path withExtension(const std::string& ext) const {
        fs::path newPath = path;
        newPath.replace_filename(newPath.filename().string() + ext);
        return Path(newPath);
    }

    Path replaceExtension(const std::string& ext) const {
        fs::path newPath = path;
        newPath.replace_extension(ext);
        return Path(newPath);
    }

    Path appendToName(const std::string& suffix) const {
        fs::path newPath = path;
        newPath.replace_filename(newPath.stem().string() + suffix + newPath.extension().string());
        return Path(newPath);
    }

private:
    fs::path path;
};

} // namespace quasar

#endif // PATH_H
