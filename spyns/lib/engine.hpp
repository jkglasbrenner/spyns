#ifndef SPYNS_LIB_ENGINE_HPP
#define SPYNS_LIB_ENGINE_HPP

#include <components.hpp>

class Engine {
private:
  data::components::Names names_;
  data::components::Sublattices sublattices_;
  data::components::NumberNeighbors number_neighbors_;
  data::components::MapNeighbors map_neighbors_;
  data::components::ListOfNeighbors list_of_neighbors_;
  data::components::MapBilinearInteractions map_bilinear_interactions_;
  data::components::ListOfBilinearInteractions list_of_bilinear_interactions_;
  data::components::BinaryStates binary_states_;
  data::components::Vector3States vector3_states_;

public:
  Engine();

  void update();
};

#endif   // SPYNS_LIB_ENGINE_HPP
