#ifndef SPYNS_LIB_COMPONENTS_HPP
#define SPYNS_LIB_COMPONENTS_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <data.hpp>

namespace components {
using Names = std::vector<std::string>;
using Sublattices = std::vector<uint64_t>;
using NumberNeighbors = std::vector<uint64_t>;
using MapNeighbors = std::vector<uint64_t>;
using ListOfNeighbors = std::vector<uint64_t>;
using MapBilinearInteractions = std::vector<uint64_t>;
using ListOfBilinearInteractions = std::vector<double>;
using BinaryStates = std::vector<int8_t>;
using Vector3States = std::vector<data::Vector3>;
}   // namespace components
#endif   // SPYNS_LIB_COMPONENTS_HPP
