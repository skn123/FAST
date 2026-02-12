#include <FAST/Data/Image.hpp>
#include "LabelModifier.hpp"

namespace fast {


LabelModifier::LabelModifier() {
    createInputPort<Image>(0);
    createOutputPort<Image>(0);

    createOpenCLProgram(Config::getKernelSourcePath() + "Algorithms/LabelModifier/LabelModifier.cl");

    createIntegerAttribute("label-changes", "Label changes", "List of label pairs oldValue1 newValue1 oldValue2 newValue2 ...", 0);
}


LabelModifier::LabelModifier(std::map<uint, uint> labelMap) : LabelModifier() {
    for(const auto& item : labelMap) {
        setLabelChange(item.first, item.second);
    }
}

void LabelModifier::execute() {
    if(m_labelChanges.empty())
        throw Exception("No label changes were given to LabelModifier");
    auto device = std::dynamic_pointer_cast<OpenCLDevice>(getMainDevice());
    cl::Buffer changesBuffer(
            device->getContext(),
            CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
            sizeof(uint)*m_labelChanges.size(),
            m_labelChanges.data()
    );

    auto input = getInputData<Image>();
    if(input->getDimensions() != 2 || (input->getDataType() != TYPE_UINT8 && input->getDataType() != TYPE_UINT16 && input->getDataType() != TYPE_UINT32))
        throw Exception("Input to LabelModifier must be 2D image of type uint8, uint16 or uint32");

    auto output = Image::create(input->getSize(), TYPE_UINT8, 1); // TODO support uint16 and uint32 as well
    output->setSpacing(input->getSpacing());
    SceneGraph::setParentNode(output, input);

    auto inputAccess = input->getOpenCLImageAccess(ACCESS_READ, device);
    auto outputAccess = output->getOpenCLImageAccess(ACCESS_READ_WRITE, device);

    cl::Program program(getOpenCLProgram(device));
    cl::Kernel kernel(program, "modifyLabels");
    kernel.setArg(0, *inputAccess->get2DImage());
    kernel.setArg(1, *outputAccess->get2DImage());
    kernel.setArg(2, changesBuffer);
    kernel.setArg(3, (int)m_labelChanges.size());

    device->getCommandQueue().enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange(input->getWidth(), input->getHeight()),
        cl::NullRange
    );

    addOutputData(0, output);
}

void LabelModifier::setLabelChange(uint oldLabel, uint newLabel) {
    m_labelChanges.push_back(oldLabel);
    m_labelChanges.push_back(newLabel);
    setModified(true);
}

void LabelModifier::loadAttributes() {
    auto list = getIntegerListAttribute("label-changes");

    for(int i = 0; i < list.size(); i += 2) {
        setLabelChange(list[i], list[i+1]);
    }
}

}