#!/usr/bin/env python3
import sys

def open_or_exit(fname, mode):
    try:
        f = open(fname, mode)
    except OSError as e:
        print(f"Error: {e.strerror} - {fname}")
        sys.exit(1)
    return f

def main(argv):
    if len(argv) < 3:
        print(f"USAGE: {argv[0]} {{sym}} {{rsrc}} {{out_file}}\n\n"
              "  Creates {sym}.c from the contents of {rsrc}\n")
        return 1

    sym = argv[1]
    in_file = open_or_exit(argv[2], "rb")
    out_file = open_or_exit(argv[3], "a")

    out_file.write(f"static const char {sym}[] = {{\n")

    linecount = 0
    while True:
        buf = in_file.read(256)
        if not buf:
            break
        for byte in buf:
            out_file.write(f"0x{byte:02x}, ")
            linecount += 1
            if linecount == 10:
                out_file.write("\n")
                linecount = 0

    if linecount > 0:
        out_file.write("\n")

    out_file.write(f"}};\n")
    out_file.write(f"static const size_t {sym}_len = sizeof({sym});\n\n")

    in_file.close()
    out_file.close()

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
