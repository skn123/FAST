import fast
import pytest
import numpy as np
import os


class OpenCLProcessObject(fast.PythonProcessObject):
    def __init__(self, useIndex=False, missingArgument=False, loadFromFile=False):
        super().__init__()
        self.createInputPort(0)
        self.createOutputPort(0)
        self.useIndex = useIndex
        self.missingArgument = missingArgument

        # Create an image invert OpenCL kernel
        if loadFromFile:
            self.createOpenCLProgram(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'opencl_kernel.cl'))
        else:
            self.createInlineOpenCLProgram(
                '''
                __const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;
                __kernel void invert(
                    __read_only image2d_t input, 
                    __write_only image2d_t output,
                    __private uint max
                    ) {
                    int2 pos = {get_global_id(0), get_global_id(1)};
                    int value = read_imageui(input, sampler, pos).x;
                    write_imageui(output, pos, max - value);
                }
                '''
            )

    def execute(self):
        # Get input image and create output image:
        image = self.getInputData()
        output = fast.Image.createFromImage(image)

        # Get kernel from OpenCL program
        kernel = self.getKernel('invert')
        # Provide arguments to the kernel
        ID1 = 0 if self.useIndex else 'input'
        ID2 = 1 if self.useIndex else 'output'
        ID3 = 2 if self.useIndex else 'max'
        kernel.setArg(ID1, image)
        kernel.setArg(ID2, output)
        if not self.missingArgument:
            kernel.setArg(ID3, 255)

        # Add the kernel to the command queue
        self.getQueue().add(kernel, image.getSize())

        # Pass on the output image
        self.addOutputData(0, output)


def test_simple_image_kernel():
    importer = fast.ImageFileImporter.create(fast.Config.getTestDataPath() + 'US/Heart/ApicalFourChamber/US-2D_0.mhd')
    inverter = fast.ImageInverter.create(min=0, max=255).connect(importer)
    result2 = inverter.runAndGetOutputData()

    PO = OpenCLProcessObject.create(useIndex=False).connect(importer)
    result = PO.runAndGetOutputData()
    assert (np.asarray(result) == np.asarray(result2)).all()

    PO = OpenCLProcessObject.create(useIndex=True).connect(importer)
    result = PO.runAndGetOutputData()
    assert (np.asarray(result) == np.asarray(result2)).all()

    PO = OpenCLProcessObject.create(loadFromFile=True).connect(importer)
    result = PO.runAndGetOutputData()
    assert (np.asarray(result) == np.asarray(result2)).all()


def test_missing_argument():
    importer = fast.ImageFileImporter.create(fast.Config.getTestDataPath() + 'US/Heart/ApicalFourChamber/US-2D_0.mhd')
    PO = OpenCLProcessObject.create(missingArgument=True).connect(importer)
    with pytest.raises(RuntimeError):
        result = PO.runAndGetOutputData()


class OpenCLProcessObject2(fast.PythonProcessObject):
    def __init__(self, useNumpy=False, createBuffer=False):
        super().__init__()
        self.createInputPort(0)
        self.createOutputPort(0)

        self.useNumpy = useNumpy
        self.useCustomBuffer = createBuffer
        self.data = list(range(255, -1, -1))

        self.createInlineOpenCLProgram(
            '''
            __const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;
            __kernel void map(
                __read_only image2d_t input, 
                __write_only image2d_t output,
                __global uint* data
                ) {
                int2 pos = {get_global_id(0), get_global_id(1)};
                int value = read_imageui(input, sampler, pos).x;
                write_imageui(output, pos, data[value]);
            }
            '''
        )

    def execute(self):
        # Get input image and create output image:
        image = self.getInputData()
        output = fast.Image.createFromImage(image)

        # Get kernel from OpenCL program
        kernel = self.getKernel('map')

        kernel.setArg('input', image)
        kernel.setArg('output', output)
        if self.useNumpy:
            kernel.setArg('data', np.array(self.data, dtype=np.uint8))
        elif self.useCustomBuffer:
            buffer = self.createUIntBuffer(self.data)
            kernel.setArg('data', buffer)
        else:
            kernel.setArg('data', self.data)

        # Add the kernel to the command queue
        self.getQueue().add(kernel, image.getSize())

        # Pass on the output image
        self.addOutputData(0, output)


def test_image_kernel_with_buffer():
    importer = fast.ImageFileImporter.create(fast.Config.getTestDataPath() + 'US/Heart/ApicalFourChamber/US-2D_0.mhd')
    inverter = fast.ImageInverter.create(min=0, max=255).connect(importer)
    result2 = inverter.runAndGetOutputData()

    PO = OpenCLProcessObject2.create().connect(importer)
    result = PO.runAndGetOutputData()
    assert (np.asarray(result) == np.asarray(result2)).all()

    PO = OpenCLProcessObject2.create(useNumpy=True).connect(importer)
    result = PO.runAndGetOutputData()
    assert (np.asarray(result) == np.asarray(result2)).all()

    PO = OpenCLProcessObject2.create(createBuffer=True).connect(importer)
    result = PO.runAndGetOutputData()
    assert (np.asarray(result) == np.asarray(result2)).all()


# TODO Test: setImageArg, setTensorArg
