#ifndef SPYNS_LIB_COMPONENTS_HPP
#define SPYNS_LIB_COMPONENTS_HPP

#include <string>
#include <vector>

#include <data.hpp>

namespace components {
using Names = std::vector<std::string>;
using Sublattices = std::vector<int>;
using NumberNeighbors = std::vector<int>;
using MapNeighbors = std::vector<int>;
using ListOfNeighbors = std::vector<int>;
using MapBilinearInteractions = std::vector<int>;
using ListOfBilinearInteractions = std::vector<double>;
using BinaryStates = std::vector<int>;
using Vector3States = std::vector<data::Vector3>;
}   // namespace components
#endif   // SPYNS_LIB_COMPONENTS_HPP
