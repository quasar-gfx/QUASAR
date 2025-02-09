#!/usr/bin/env python3
import sys, os, re

def open_or_exit(fname, mode):
    try:
        f = open(fname, mode)
    except OSError as e:
        print(f"Error: {e.strerror} - {fname}")
        sys.exit(1)
    return f

def read_with_includes(fname, included=None):
    if included is None:
        included = set()

    if fname in included:
        return ""
    included.add(fname)

    lines = []
    with open_or_exit(fname, "r") as f:
        for line in f:
            m = re.match(r'^[ \t]*#include\s+"([^"]+)"', line)
            if m:
                path = os.path.join(os.path.dirname(fname), m.group(1))
                lines.append(read_with_includes(path, included))
                continue

            lines.append(line)

    return "".join(lines)

def main(argv):
    if len(argv) < 3:
        print(f"USAGE: {argv[0]} {{sym}} {{rsrc}} {{out_file}}\n\n"
              "  Creates {sym}.c from the contents of {rsrc}\n")
        return 1

    sym = argv[1]

    content = read_with_includes(argv[2])
    buf = content.encode("utf-8")

    out_file = open_or_exit(argv[3], "a")
    out_file.write(f"static const char {sym}[] = {{\n")

    linecount = 0
    for b in buf:
        out_file.write(f"0x{b:02x}, ")
        linecount += 1
        if linecount == 10:
            out_file.write("\n")
            linecount = 0

    if linecount > 0:
        out_file.write("\n")

    out_file.write("};\n")
    out_file.write(f"static const size_t {sym}_len = sizeof({sym});\n\n")
    out_file.close()

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
