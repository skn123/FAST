#include <FAST/Testing.hpp>
#include <FAST/Importers/ImageFileImporter.hpp>
#include <FAST/Visualization/Shortcuts.hpp>
#include <FAST/Algorithms/ImageCaster/ImageCaster.hpp>
#include "OtsuThresholding.hpp"

using namespace fast;

TEST_CASE("Otsu thresholding 2D uint8 image", "[fast][OtsuThresholding][visual]") {
    auto importer = ImageFileImporter::create(Config::getTestDataPath() + "US/US-2D.jpg");
    auto segment = OtsuThresholding::create()->connect(importer);
    Display2DArgs args;
    args.image = importer;
    args.segmentation = segment;
    args.timeout = 1000;
    display2D(args);
}


TEST_CASE("Otsu thresholding 2D float image", "[fast][OtsuThresholding][visual]") {
    auto importer = ImageFileImporter::create(Config::getTestDataPath() + "US/US-2D.jpg");
    auto caster = ImageCaster::create(TYPE_FLOAT, 2.0f)->connect(importer);
    auto segment = OtsuThresholding::create()->connect(caster);
    Display2DArgs args;
    args.image = importer;
    args.segmentation = segment;
    args.timeout = 1000;
    display2D(args);
}
