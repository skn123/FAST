#include "RegionProperties.hpp"
#include "RemoveRegions.hpp"
#include <FAST/Testing.hpp>
#include <FAST/Importers/ImageFileImporter.hpp>
#include <FAST/Algorithms/Thresholding/BinaryThresholding.hpp>
#include <FAST/Streamers/ImageFileStreamer.hpp>
#include <FAST/Algorithms/NeuralNetwork/SegmentationNetwork.hpp>
#include <FAST/Visualization/Shortcuts.hpp>

using namespace fast;

TEST_CASE("Region properties", "[regionproperties][fast]") {

    auto importer = ImageFileImporter::create(Config::getTestDataPath() + "US/CarotidArtery/Right/US-2D_0.mhd");

    auto segmentation = BinaryThresholding::create(100)->connect(importer);

    auto regionProperties = RegionProperties::create()->connect(segmentation);
    auto regionList = regionProperties->run()->getOutput<RegionList>();
    auto regions = regionList->get();

    // TODO validate
    for(auto& region : regions) {
        //std::cout << "Area: " << region.area << std::endl;
        //std::cout << "Label: " << (int)region.label << std::endl;
    }
}

TEST_CASE("Remove regions remove all but largest", "[RemoveRegions][fast]") {
    auto streamer = ImageFileStreamer::create(Config::getTestDataPath() + "US/JugularVein/US-2D_#.mhd", true, false, 20);

    auto segmentation = SegmentationNetwork::create(
            join(Config::getTestDataPath(), "NeuralNetworkModels/jugular_vein_segmentation.onnx"),
            1.0f/255.0f)
            ->connect(streamer);
    auto removeRegions = RemoveRegions::create(true)->connect(segmentation);
    Display2DArgs args;
    args.image = streamer;
    args.segmentation = removeRegions;
    args.timeout = 1000;
    display2D(args);
}

TEST_CASE("Remove regions with min area and max area", "[RemoveRegions][fast]") {
    auto streamer = ImageFileStreamer::create(Config::getTestDataPath() + "US/JugularVein/US-2D_#.mhd", true, false, 20);

    auto segmentation = SegmentationNetwork::create(
            join(Config::getTestDataPath(), "NeuralNetworkModels/jugular_vein_segmentation.onnx"),
            1.0f/255.0f)
                    ->connect(streamer);
    auto removeRegions = RemoveRegions::create(false, 0, 30, 80)->connect(segmentation);
    Display2DArgs args;
    args.image = streamer;
    args.segmentation = removeRegions;
    args.timeout = 1000;
    display2D(args);
}
