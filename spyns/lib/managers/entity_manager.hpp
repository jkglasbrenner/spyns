#ifndef SPYNS_LIB_MANAGERS_ENTITY_MANAGER_HPP
#define SPYNS_LIB_MANAGERS_ENTITY_MANAGER_HPP

#include <string>
#include <vector>

#include <components.hpp>
#include <entities.hpp>

struct EntityManager {
  data::components::Names names_;
  data::components::Sublattices sublattices_;
  data::components::NumberNeighbors number_neighbors_;
  data::components::MapNeighbors map_neighbors_;
  data::components::ListOfNeighbors list_of_neighbors_;
  data::components::MapBilinearInteractions map_bilinear_interactions_;
  data::components::ListOfBilinearInteractions list_of_bilinear_interactions_;
  data::components::BinaryStates binary_states_;
  data::components::Vector3States vector3_states_;

  void reserve(int n) { names_.reserve(n); }

  data::entities::Entity add_entity(std::string &&name) {
    data::entities::Entity entity = {names_.size()};

    names_.emplace_back(name);

    return entity;
  }
};
#endif   // SPYNS_LIB_MANAGERS_ENTITY_MANAGER_HPP
