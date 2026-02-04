import fast


class OpenCLLookupTable(fast.PythonProcessObject):
    def __init__(self):
        super().__init__()
        self.createInputPort(0)
        self.createOutputPort(0)

        # Create the lookup table
        # Make all pixels above 128, red and all pixels below blue
        self.table = []
        for i in range(256):
            if i > 128:
                self.table.append(i) # Red
                self.table.append(0) # Green
                self.table.append(0) # Blue
            else:
                self.table.append(0) # Red
                self.table.append(0) # Green
                self.table.append(i) # Blue

        # Create OpenCL kernel
        self.createInlineOpenCLProgram(
            '''
            __const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;
            __kernel void transform(
                __read_only image2d_t input,
                __write_only image2d_t output,
                __constant uchar* table
            ) {
                int2 pos = {get_global_id(0), get_global_id(1)};
                int value = read_imageui(input, sampler, pos).x;
                uchar3 color = vload3(value, table);
                write_imageui(output, pos, (uint4)(color.x, color.y, color.z, 0));
            }
            ''')
        self.tableBuffer = None

    def execute(self):
        if self.tableBuffer is None:
            # Only create OpenCL buffer for table once, since this data never changes.
            self.tableBuffer = self.createUCharBuffer(self.table, fast.KernelArgumentAccess_READ_ONLY)

        input = self.getInputData()
        output = fast.Image.create(input.getWidth(), input.getHeight(), fast.TYPE_UINT8, 3) # 3 channels since color
        output.setSpacing(input.getSpacing())

        self.kernel = self.getKernel('transform')
        self.kernel.setArg('table', self.tableBuffer)
        self.kernel.setArg('input', input)
        self.kernel.setArg('output', output)

        self.getQueue().add(self.kernel, input.getSize())

        self.addOutputData(0, output)


# Set up pipeline as normal

# Stream some ultrasound data
importer = fast.ImageFileStreamer.create(
    fast.Config.getTestDataPath() + 'US/Heart/ApicalFourChamber/US-2D_#.mhd',
    loop=True,
    framerate=40,
    )

# Set up the process object
converter = OpenCLLookupTable.create().connect(importer)

# Run pipeline and display
fast.display2D(converter)
