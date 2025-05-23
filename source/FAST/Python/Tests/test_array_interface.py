import fast
import numpy as np
import pytest


def test_2D_image_array_interface():
    types_to_test = [
        (np.uint8, fast.TYPE_UINT8, 255),
        (np.float32, fast.TYPE_FLOAT, 1),
        (np.uint16, fast.TYPE_UINT16, 128),
        (np.int16, fast.TYPE_INT16, 128)
    ]
    width = 64
    height = 37

    for np_type, fast_type, scale in types_to_test:
        # Test no channels specified
        data = (np.random.rand(height, width)*scale).astype(np_type)
        image = fast.Image.createFromArray(data)

        assert image.getWidth() == width
        assert image.getHeight() == height
        assert image.getNrOfChannels() == 1
        assert image.getDataType() == fast_type
        assert np.array_equal(np.asarray(image), data.reshape((height, width, 1)))

        for channels in range(1, 5):
            data = (np.random.rand(height, width, channels)*scale).astype(np_type)
            image = fast.Image.createFromArray(data)

            assert image.getWidth() == width
            assert image.getHeight() == height
            assert image.getNrOfChannels() == channels
            assert image.getDataType() == fast_type
            assert np.array_equal(np.asarray(image), data)


def test_3D_image_array_interface():
    types_to_test = [
        (np.uint8, fast.TYPE_UINT8, 255),
        (np.float32, fast.TYPE_FLOAT, 1),
        (np.uint16, fast.TYPE_UINT16, 128),
        (np.int16, fast.TYPE_INT16, 128)
    ]
    width = 64
    height = 37
    depth = 89

    for np_type, fast_type, scale in types_to_test:
        # Test no channels specified
        data = (np.random.rand(depth, height, width)*scale).astype(np_type)
        image = fast.Image.createFromArray(data)

        assert image.getWidth() == width
        assert image.getHeight() == height
        assert image.getDepth() == depth
        assert image.getNrOfChannels() == 1
        assert image.getDataType() == fast_type
        assert np.array_equal(np.asarray(image), data.reshape((depth, height, width, 1)))

        for channels in range(1, 5):
            data = (np.random.rand(depth, height, width, channels)*scale).astype(np_type)
            image = fast.Image.createFromArray(data)

            assert image.getWidth() == width
            assert image.getHeight() == height
            assert image.getDepth() == depth
            assert image.getNrOfChannels() == channels
            assert image.getDataType() == fast_type
            assert np.array_equal(np.asarray(image), data)


def test_image_array_interface_exceptions():
    data = ''
    with pytest.raises(ValueError):
        fast.Image.createFromArray(data)
    data = np.ndarray((16,), dtype=np.uint8)
    with pytest.raises(ValueError):
        fast.Image.createFromArray(data)
    data = np.ndarray((16,34,23,54,23), dtype=np.uint8)
    with pytest.raises(ValueError):
        fast.Image.createFromArray(data)
    data = np.ndarray((16,34), dtype=np.float64)
    with pytest.raises(TypeError):
        fast.Image.createFromArray(data)


def test_tensor_array_interface():
    types_to_test = [
        (np.uint8, 255),
        (np.int8, 127),
        (np.float32, 1),
        (np.float64, 1),
        (np.uint16, 128),
        (np.int16, 128)
    ]
    for type, scale in types_to_test:
        shape = (23,)
        numpy_tensor = (np.random.random(shape)*scale).astype(type)
        fast_tensor = fast.Tensor.createFromArray(numpy_tensor)
        assert fast_tensor.getShape().getDimensions() == len(shape)
        assert fast_tensor.getShape().getAll()[0] == shape[0]
        assert np.all(numpy_tensor.astype(np.float32) == np.array(fast_tensor))

        shape = (23,1,23,67)
        numpy_tensor = (np.random.random(shape)*scale).astype(type)
        fast_tensor = fast.Tensor.createFromArray(numpy_tensor)
        assert fast_tensor.getShape().getDimensions() == len(shape)
        for i in range(len(shape)):
            assert fast_tensor.getShape().getAll()[i] == shape[i]
        assert np.all(numpy_tensor.astype(np.float32) == np.array(fast_tensor))


def test_tensor_array_interface_exceptions():
    data = ''
    with pytest.raises(ValueError):
        fast.Tensor.createFromArray(data)


def test_image_array_interface_conversion():
    """ Test fortran -> C array conversion for images """
    # If channel dimension not given, it fails,
    # this is because a channel dimension is added for grayscale image array.
    # Not sure if this can be considered a bug..
    shapes = [
        (23, 28, 64, 3),
        (23, 31, 64, 1),
        #(120, 31, 12), # Fails
        (23, 28, 2),
        (23, 28, 1),
        #(120, 31), # Fails
    ]
    types_to_test = [
        (np.uint8, 255),
        (np.float32, 1),
        (np.uint16, 128),
        (np.int16, 128)
    ]
    for shape in shapes:
        for type, scale in types_to_test:
            data = (np.random.random(shape)*scale).astype(type)
            data = np.asfortranarray(data)
            assert data.flags['F_CONTIGUOUS']
            image = fast.Image.createFromArray(data)
            data2 = np.asarray(image)
            assert data.shape == data2.shape
            assert np.array_equal(data2, data)
            assert data2.flags['C_CONTIGUOUS']


def test_tensor_array_interface_conversion():
    """ Test fortran -> C array and float32 conversion for tensors"""
    shape = (23,20,2)
    types_to_test = [
        (np.uint8, 255),
        (np.int8, 127),
        (np.float32, 1),
        (np.float64, 1),
        (np.uint16, 128),
        (np.int16, 128),
        (np.uint32, 128),
        (np.int32, 128)
    ]
    for type, scale in types_to_test:
        data = (np.random.random(shape)*scale).astype(type)
        data = np.asfortranarray(data)
        assert data.flags['F_CONTIGUOUS']
        image = fast.Tensor.createFromArray(data)
        data2 = np.asarray(image)
        assert data2.flags['C_CONTIGUOUS']
        assert data2.dtype == np.float32
        assert np.array_equal(data2, data.astype(np.float32))
