fast_add_sources(
    BinaryThresholding.cpp
    BinaryThresholding.hpp
    OtsuThresholding.cpp
    OtsuThresholding.hpp
)
fast_add_process_object(BinaryThresholding BinaryThresholding.hpp)
fast_add_process_object(OtsuThresholding OtsuThresholding.hpp)

fast_add_test_sources(
    OtsuThresholdingTests.cpp
)
