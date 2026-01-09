#include <FAST/Testing.hpp>
#include <FAST/ProcessObject.hpp>
#include <FAST/Importers/ImageFileImporter.hpp>
#include <FAST/Visualization/Shortcuts.hpp>

namespace fast {

class TestPO : public ProcessObject {
    FAST_OBJECT(TestPO)
    public:
        FAST_CONSTRUCTOR(TestPO)
    private:
        void execute() override;
};

TestPO::TestPO() {
    createInputPort(0);
    createOutputPort(0);
    createInlineOpenCLProgram(R"(
__const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;
__kernel void invert(__read_only image2d_t input, __write_only image2d_t output) {
    int2 pos = {get_global_id(0), get_global_id(1)};
    int value = read_imageui(input, sampler, pos).x;
    write_imageui(output, pos, 255 - value);
}
)");
}


void TestPO::execute() {
    auto input = getInputData<Image>();
    auto output = Image::createFromImage(input);

    auto kernel = getKernel("invert");
    kernel.setArg("input", input);
    kernel.setArg("output", output);

    getQueue().add(kernel, input->getSize());

    addOutputData(0, output);
}

} // End namespace

using namespace fast;

TEST_CASE("Inline OpenCL in ProcessObject", "[fast][OpenCL][ProcessObject]") {
    auto importer = ImageFileImporter::create(Config::getTestDataPath() + "US/Heart/ApicalFourChamber/US-2D_0.mhd");

    auto inverter = TestPO::create()->connect(importer);

    inverter->run();
    /*
    Display2DArgs args;
    args.image = inverter;
    args.timeout = 1000;
    display2D(args);
     */
}