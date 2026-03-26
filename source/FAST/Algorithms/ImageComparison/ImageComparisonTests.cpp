#include <FAST/Testing.hpp>
#include <FAST/Importers/ImageFileImporter.hpp>
#include <FAST/Importers/WholeSlideImageImporter.hpp>
#include <FAST/Data/ImagePyramid.hpp>
#include <FAST/Algorithms/GaussianSmoothing/GaussianSmoothing.hpp>
#include <FAST/Visualization/Plotting/LinePlotter.hpp>
#include <FAST/Algorithms/ImageCropper/ImageCropper.hpp>
#include "PSNR.hpp"
#include "SSIM.hpp"

using namespace fast;

TEST_CASE("MSE on 2D images", "[fast][MSE][ImageComparison]") {
    auto importer1 = ImageFileImporter::create(Config::getTestDataPath() + "US/Heart/ApicalFourChamber/US-2D_0.mhd");
    auto image1 = importer1->run()->getOutput<Image>();
    auto importer2 = ImageFileImporter::create(Config::getTestDataPath() + "US/Heart/ApicalFourChamber/US-2D_20.mhd");

    auto mse = MSE::create()
            ->connect(0, importer1)
            ->connect(1, importer2);
    CHECK_NOTHROW(
        mse->run();
    );

    auto output = mse->getOutput<Image>();
    CHECK(mse->get() == Approx(676.59));
    CHECK(output->getWidth() == image1->getWidth());
    CHECK(output->getHeight() == image1->getHeight());
    CHECK(output->getDataType() == TYPE_FLOAT);
    CHECK(output->getNrOfChannels() == image1->getNrOfChannels());
}

TEST_CASE("MSE on 2D color images", "[fast][MSE][ImageComparison]") {
    auto importer1 = WholeSlideImageImporter::create(Config::getTestDataPath() + "WSI/CMU-1.svs");
    auto wsi = importer1->run()->getOutput<ImagePyramid>();
    auto access = wsi->getAccess(ACCESS_READ);
    auto image1 = access->getLevelAsImage(wsi->getNrOfLevels()-1);
    auto image2 = GaussianSmoothing::create()->connect(image1);

    auto mse = MSE::create()->connect(image1)->connect(1, image2);
    CHECK_NOTHROW(
        mse->run();
    );

    auto output = mse->getOutput<Image>();
    CHECK(output->getWidth() == image1->getWidth());
    CHECK(output->getHeight() == image1->getHeight());
    CHECK(output->getDataType() == TYPE_FLOAT);
    CHECK(output->getNrOfChannels() == image1->getNrOfChannels());
}

TEST_CASE("MSE on 3D images", "[fast][MSE][ImageComparison]") {
    auto importer1 = ImageFileImporter::create(Config::getTestDataPath() + "US/Ball/US-3Dt_0.mhd");
    auto image1 = importer1->run()->getOutput<Image>();
    auto importer2 = ImageFileImporter::create(Config::getTestDataPath() + "US/Ball/US-3Dt_20.mhd");

    auto mse = MSE::create()
            ->connect(0, importer1)
            ->connect(1, importer2);
    CHECK_NOTHROW(
        mse->run();
    );

    auto output = mse->getOutput<Image>();
    CHECK(mse->get() == Approx(214.87));
    CHECK(output->getWidth() == image1->getWidth());
    CHECK(output->getHeight() == image1->getHeight());
    CHECK(output->getDepth() == image1->getDepth());
    CHECK(output->getDataType() == TYPE_FLOAT);
    CHECK(output->getNrOfChannels() == image1->getNrOfChannels());
}


TEST_CASE("PSNR on 2D image", "[fast][PSNR][ImageComparison]") {
    auto importer1 = ImageFileImporter::create(Config::getTestDataPath() + "US/Heart/ApicalFourChamber/US-2D_0.mhd");
    auto image1 = importer1->run()->getOutput<Image>();
    auto importer2 = ImageFileImporter::create(Config::getTestDataPath() + "US/Heart/ApicalFourChamber/US-2D_20.mhd");

    auto psnr = PSNR::create(255)
            ->connect(0, importer1)
            ->connect(1, importer2);
    CHECK_NOTHROW(
        psnr->run();
    );

    auto output = psnr->getOutput<Image>();
    CHECK(psnr->get() == Approx(19.8275));
    CHECK(output->getWidth() == image1->getWidth());
    CHECK(output->getHeight() == image1->getHeight());
    CHECK(output->getDepth() == image1->getDepth());
    CHECK(output->getDataType() == TYPE_FLOAT);
    CHECK(output->getNrOfChannels() == image1->getNrOfChannels());
}

TEST_CASE("SSIM on 2D image", "[fast][SSIM][ImageComparison]") {
    auto importer1 = ImageFileImporter::create(Config::getTestDataPath() + "US/Heart/ApicalFourChamber/US-2D_0.mhd");
    auto image1 = importer1->run()->getOutput<Image>();
    auto importer2 = ImageFileImporter::create(Config::getTestDataPath() + "US/Heart/ApicalFourChamber/US-2D_20.mhd");

    // Crop to be able to compare value with other implementations
    auto cropper1 = ImageCropper::create(Vector2i(image1->getWidth()-10, image1->getHeight()-10), Vector2i(5, 5))->connect(image1);
    auto cropper2 = ImageCropper::create(Vector2i(image1->getWidth()-10, image1->getHeight()-10), Vector2i(5, 5))->connect(importer2);
    image1 = cropper1->run()->getOutput<Image>();

    auto ssim = SSIM::create(255, 0, Vector3i::Constant(11), Vector3f::Constant(1.5f))
            ->connect(0, image1)
            ->connect(1, cropper2);
    ssim->enableRuntimeMeasurements();
    ssim->run();

    auto output = ssim->getOutput<Image>();
    auto value = ssim->getOutput<FloatScalar>(1);
    std::cout << ssim->get() << std::endl;
    ssim->getAllRuntimes()->printAll();

    CHECK(value->get() == ssim->get());
    CHECK(ssim->get() == Approx(0.59).epsilon(0.01));
    CHECK(output->getWidth() == image1->getWidth());
    CHECK(output->getHeight() == image1->getHeight());
    CHECK(output->getDepth() == image1->getDepth());
    CHECK(output->getDataType() == TYPE_FLOAT);
    CHECK(output->getNrOfChannels() == image1->getNrOfChannels());
}

TEST_CASE("SSIM on 2D color image", "[fast][SSIM][ImageComparison]") {
    auto importer1 = WholeSlideImageImporter::create(Config::getTestDataPath() + "WSI/CMU-1.svs");
    auto wsi = importer1->run()->getOutput<ImagePyramid>();
    auto access = wsi->getAccess(ACCESS_READ);
    auto image1 = access->getLevelAsImage(wsi->getNrOfLevels()-1);
    auto image2 = GaussianSmoothing::create()->connect(image1);

    auto ssim = SSIM::create(255)->connect(image1)->connect(1, image2);
    ssim->enableRuntimeMeasurements();

    CHECK_NOTHROW(
        ssim->run();
    );

    auto output = ssim->getOutput<Image>(0);
    auto value = ssim->getOutput<FloatScalar>(1);
    ssim->getAllRuntimes()->printAll();

    CHECK(value->get() == ssim->get());
    CHECK(output->getWidth() == image1->getWidth());
    CHECK(output->getHeight() == image1->getHeight());
    CHECK(output->getDepth() == image1->getDepth());
    CHECK(output->getDataType() == TYPE_FLOAT);
    CHECK(output->getNrOfChannels() == image1->getNrOfChannels());
}

TEST_CASE("SSIM on 3D images", "[fast][SSIM][ImageComparison]") {
    auto importer1 = ImageFileImporter::create(Config::getTestDataPath() + "US/Ball/US-3Dt_0.mhd");
    auto image1 = importer1->run()->getOutput<Image>();
    auto importer2 = ImageFileImporter::create(Config::getTestDataPath() + "US/Ball/US-3Dt_20.mhd");

    // Crop to be able to compare value with other implementations
    auto cropper1 = ImageCropper::create(Vector3i(image1->getWidth()-10, image1->getHeight()-10, image1->getDepth()-10), Vector3i(5, 5, 5))->connect(image1);
    auto cropper2 = ImageCropper::create(Vector3i(image1->getWidth()-10, image1->getHeight()-10, image1->getDepth()-10), Vector3i(5, 5, 5))->connect(importer2);
    image1 = cropper1->run()->getOutput<Image>();

    auto ssim = SSIM::create(255)
            ->connect(0, image1)
            ->connect(1, cropper2);
    CHECK_NOTHROW(
        ssim->run();
    );
    std::cout << ssim->get() << std::endl;

    auto output = ssim->getOutput<Image>(0);
    auto value = ssim->getOutput<FloatScalar>(1);

    CHECK(value->get() == ssim->get());
    CHECK(ssim->get() == Approx(0.73).epsilon(0.01));
    CHECK(output->getWidth() == image1->getWidth());
    CHECK(output->getHeight() == image1->getHeight());
    CHECK(output->getDepth() == image1->getDepth());
    CHECK(output->getDataType() == TYPE_FLOAT);
    CHECK(output->getNrOfChannels() == image1->getNrOfChannels());
}
