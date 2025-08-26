#pragma once

#include <iostream>
#include <fstream>

namespace wowlib{

template<typename T>
static void WriteBinaryPOD(std::ostream &out, const T &podRef) {
    out.write((char *) &podRef, sizeof(T));
}

template<typename T>
static void ReadBinaryPOD(std::istream &in, T &podRef) {
    in.read((char *) &podRef, sizeof(T));
}

} // namespace spatt