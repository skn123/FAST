fast_add_sources(
        RegionProperties.cpp
        RegionProperties.hpp
        RemoveRegions.cpp
        RemoveRegions.hpp
)
fast_add_process_object(RegionProperties RegionProperties.hpp)
fast_add_process_object(RemoveRegions RemoveRegions.hpp)
fast_add_test_sources(Tests.cpp)