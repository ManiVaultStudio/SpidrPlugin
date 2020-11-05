#pragma once

#include <vector>
#include <fstream>
#include <string>

/*! Write vector contents to disk
 * Stores content in little endian binary form.
 * Overrides existing files with at the given path.
 *
 * \param vec Data to write to disk
 * \param writePath Target path
 */
template<typename T>
void writeVecToBinary(std::vector<T> vec, std::string writePath) {
    std::ofstream fout(writePath, std::ofstream::out | std::ofstream::binary);
    fout.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(T));
    fout.close();
}
