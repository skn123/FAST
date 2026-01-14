#include "ImageInverter.hpp"
#include <FAST/Testing.hpp>
#include <FAST/Importers/ImageFileImporter.hpp>

using namespace fast;

TEST_CASE("ImageInverter 2D", "[fast][ImageInverter]") {
    auto importer = ImageFileImporter::create(Config::getTestDataPath() + "US/Heart/ApicalFourChamber/US-2D_0.mhd");

    auto inverter = ImageInverter::create()->connect(importer);
    CHECK_NOTHROW(inverter->run());
}

TEST_CASE("ImageInverter 3D", "[fast][ImageInverter]") {
    auto importer = ImageFileImporter::create(Config::getTestDataPath() + "CT/CT-Thorax.mhd");

    auto inverter = ImageInverter::create()->connect(importer);
    CHECK_NOTHROW(inverter->run());
}
