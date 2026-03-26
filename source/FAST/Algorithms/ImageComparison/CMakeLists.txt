fast_add_sources(
    MSE.cpp
    MSE.hpp
    PSNR.cpp
    SSIM.cpp
)
fast_add_process_object(MeanSquaredError MSE.hpp)
fast_add_process_object(PeakSignalToNoiseRatio PSNR.hpp)
fast_add_process_object(StructuralSimilarityIndexMeasure SSIM.hpp)
fast_add_test_sources(ImageComparisonTests.cpp)