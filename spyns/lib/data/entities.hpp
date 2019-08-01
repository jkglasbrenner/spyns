#ifndef SPYNS_LIB_DATA_ENTITIES_HPP
#define SPYNS_LIB_DATA_ENTITIES_HPP

#include <cstdint>

#include <components.hpp>

namespace data {
struct Entity {
  uint64_t id;
};

struct Entities {
  components::Names names_;
  components::Sublattices sublattices_;
  components::NumberNeighbors number_neighbors_;
  components::ListOfNeighbors list_of_neighbors_;
  components::ListOfBilinearInteractions list_of_bilinear_interactions_;
  components::BinaryStates binary_states_;
  components::Vector3States vector3_states_;
  components::MapNeighbors map_neighbors_;
  components::MapBilinearInteractions map_bilinear_interactions_;

  void reserve(int n) { names_.reserve(n); }

  Entity add_entity(std::string &&name) {
    Entity entity = {names_.size()};

    names_.emplace_back(name);

    return entity;
  }
};
}   // namespace data
#endif   // SPYNS_LIB_DATA_ENTITIES_HPP
