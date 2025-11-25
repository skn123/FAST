#include <FAST/Testing.hpp>
#include <FAST/Data/ImagePyramid.hpp>
#include <FAST/Data/Image.hpp>
#include <FAST/Importers/WholeSlideImageImporter.hpp>
#include <FAST/Algorithms/ImageResizer/ImageResizer.hpp>
#include <FAST/Visualization/SimpleWindow.hpp>
#include <FAST/Visualization/ImagePyramidRenderer/ImagePyramidRenderer.hpp>
#include <FAST/Exporters/TIFFImagePyramidExporter.hpp>
#include <FAST/Importers/TIFFImagePyramidImporter.hpp>

using namespace fast;

TEST_CASE("Convert image pyramid to JPEG XL compression", "[fast][ImagePyramid][JPEG]") {
    auto importer = WholeSlideImageImporter::create(Config::getTestDataPath() + "/WSI/CMU-1.svs");
    const int level = 2;
    const int targetTileSize = 512;
    auto imagePyramid = importer->runAndGetOutputData<ImagePyramid>();
    int tilesX = std::ceil(imagePyramid->getLevelWidth(level) / targetTileSize);
    int tilesY = std::ceil(imagePyramid->getLevelHeight(level) / targetTileSize);

    auto newImagePyramid = ImagePyramid::create(tilesX*targetTileSize, tilesY*targetTileSize, 3, targetTileSize, targetTileSize, ImageCompression::JPEGXL);

    {
        auto accessRead = imagePyramid->getAccess(ACCESS_READ);
        auto accessWrite = newImagePyramid->getAccess(ACCESS_READ_WRITE);

        int counter = 0;
        for(int newLevel = 0; newLevel < newImagePyramid->getNrOfLevels(); ++newLevel) {
            //int newLevel = 0;
            tilesX = std::ceil(newImagePyramid->getLevelWidth(newLevel) / targetTileSize);
            tilesY = std::ceil(newImagePyramid->getLevelHeight(newLevel) / targetTileSize);
            int totalTiles = tilesX*tilesY;
            for(int tileX = 0; tileX < tilesX; ++tileX) {
                for(int tileY = 0; tileY < tilesY; ++tileY) {
                    try {
                        auto x = tileX*targetTileSize;
                        auto y = tileY*targetTileSize;
                        int pow = (int)std::pow(2, newLevel);
                        auto patch = accessRead->getPatchAsImage(level, x*pow, y*pow, targetTileSize*pow, targetTileSize*pow);
                        if(newLevel != 0)
                            patch = ImageResizer::create(targetTileSize, targetTileSize)->connect(patch)->runAndGetOutputData<Image>();

                        if(tileX % 2 == 1) {
                            accessWrite->setBlankPatch(newLevel, x, y);
                        } else {
                            accessWrite->setPatch(newLevel, x, y, patch, true);
                        }
                    } catch(Exception &e) {
                        continue;
                    }
                    ++counter;
                    std::cout << "Progress: " << counter << " of " << totalTiles << std::endl;
                }
            }
        }
    }
    std::cout << "Done" << std::endl;

    TIFFImagePyramidExporter::create("test.tiff")->connect(newImagePyramid)->run();

    //auto importer2 = TIFFImagePyramidImporter::create("test.tiff");
    ////auto renderer = ImagePyramidRenderer::create()->connect(newImagePyramid);
    //auto renderer = ImagePyramidRenderer::create()->connect(importer2);
    //SimpleWindow2D::create()->connect(renderer)->run();
}

TEST_CASE("getMagnification()", "[fast][ImagePyramid]") {
    auto importer = WholeSlideImageImporter::create(Config::getTestDataPath() + "/WSI/CMU-1.svs");
    auto image = importer->runAndGetOutputData<ImagePyramid>();
    CHECK(image->getMagnification() == 20);
}

TEST_CASE("setMagnification()", "[fast][ImagePyramid]") {
    auto importer = WholeSlideImageImporter::create(Config::getTestDataPath() + "/WSI/CMU-1.svs");
    auto image = importer->runAndGetOutputData<ImagePyramid>();
    image->setMagnification(40);
    CHECK(image->getMagnification() == 40);
}

TEST_CASE("getLevelForMagnification()", "[fast][ImagePyramid]") {
    auto importer = WholeSlideImageImporter::create(Config::getTestDataPath() + "/WSI/CMU-1.svs");
    auto image = importer->runAndGetOutputData<ImagePyramid>();
    CHECK(image->getLevelForMagnification(20) == 0);
    CHECK(image->getLevelForMagnification(5) == 1);
    CHECK(image->getLevelForMagnification(1.25) == 2);

    CHECK_THROWS(image->getLevelForMagnification(10));
}

TEST_CASE("getClosestLevelForMagnification()", "[fast][ImagePyramid]") {
    auto importer = WholeSlideImageImporter::create(Config::getTestDataPath() + "/WSI/CMU-1.svs");
    auto image = importer->runAndGetOutputData<ImagePyramid>();
    {
        auto [level, resampleFactor] = image->getClosestLevelForMagnification(20);
        CHECK(level == 0);
        CHECK(resampleFactor == 1.0f);
    }
    /*
    {
        auto [level, resampleFactor] = image->getClosestLevelForMagnification(10);
        CHECK(level == 0);
        CHECK(resampleFactor == 2.0f);
    }
    {
        auto [level, resampleFactor] = image->getClosestLevelForMagnification(2.5);
        CHECK(level == 1);
        CHECK(resampleFactor == 2.0f);
    }
     */
}

TEST_CASE("Create image pyramid with default levels", "[fast][ImagePyramid]") {
    auto pyramid = ImagePyramid::create(12000, 8000, 3, 512, 512, ImageCompression::JPEG);

    CHECK(pyramid->getNrOfLevels() == 3);
    CHECK(pyramid->getWidth() == 12000);
    CHECK(pyramid->getHeight() == 8000);
    CHECK(pyramid->getLevelWidth(1) == 12000/2);
    CHECK(pyramid->getLevelHeight(1) == 8000/2);
    CHECK(pyramid->getLevelWidth(2) == 12000/4);
    CHECK(pyramid->getLevelHeight(2) == 8000/4);
    CHECK_THROWS(pyramid->getLevelWidth(3));
    CHECK(pyramid->getDataType() == TYPE_UINT8);
    CHECK(pyramid->getLevelTileWidth(0) == 512);
    CHECK(pyramid->getLevelTileHeight(0) == 512);
    CHECK(pyramid->getCompression() == ImageCompression::JPEG);
    CHECK(pyramid->getNrOfChannels() == 3);
    CHECK(pyramid->getLevelScale(0) == 1);
    CHECK(pyramid->getLevelScale(1) == 2);
    CHECK(pyramid->getLevelScale(2) == 4);
}

TEST_CASE("Create image pyramid with custom levels", "[fast][ImagePyramid]") {
    auto pyramid = ImagePyramid::create(22000, 13000, 1, 512, 256, ImageCompression::JPEG, 90,
                                        fast::TYPE_UINT8, {4, 2, 2});

    CHECK(pyramid->getNrOfLevels() == 4);
    CHECK(pyramid->getWidth() == 22000);
    CHECK(pyramid->getHeight() == 13000);
    CHECK(pyramid->getLevelWidth(1) == 22000/4);
    CHECK(pyramid->getLevelHeight(1) == 13000/4);
    CHECK(pyramid->getLevelWidth(2) == 22000/(4*2));
    CHECK(pyramid->getLevelHeight(2) == 13000/(4*2));
    CHECK(pyramid->getLevelWidth(3) == 22000/(4*2*2));
    CHECK(pyramid->getLevelHeight(3) == 13000/(4*2*2));
    CHECK_THROWS(pyramid->getLevelWidth(4));
    CHECK(pyramid->getDataType() == TYPE_UINT8);
    CHECK(pyramid->getLevelTileWidth(0) == 512);
    CHECK(pyramid->getLevelTileHeight(0) == 256);
    CHECK(pyramid->getCompression() == ImageCompression::JPEG);
    CHECK(pyramid->getNrOfChannels() == 1);
    CHECK(pyramid->getLevelScale(0) == 1);
    CHECK(pyramid->getLevelScale(1) == 4);
    CHECK(pyramid->getLevelScale(2) == 8);
    CHECK(pyramid->getLevelScale(3) == 16);
}
