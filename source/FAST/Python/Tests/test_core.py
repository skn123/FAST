import fast
import pytest


def test_callable_process_object():
    image = fast.Image.create(256, 256, fast.TYPE_UINT8, 1)
    image2 = fast.GrayscaleToColor.create()(image)
    assert image2.getNrOfChannels() == 3

    image2 = fast.GrayscaleToColor.create()((0, image))
    assert image2.getNrOfChannels() == 3

    convert = fast.GrayscaleToColor.create().connect(image)
    image2 = fast.ColorToGrayscale.create()(convert)
    assert image2.getNrOfChannels() == 1

    image2 = fast.ColorToGrayscale.create()((0, convert, 0))
    assert image2.getNrOfChannels() == 1


def test_callable_process_object_exceptions():
    with pytest.raises(ValueError):
        image2 = fast.GrayscaleToColor.create()(32)
    with pytest.raises(RuntimeError):
        image2 = fast.GrayscaleToColor.create()()
    image = fast.Image.create(256, 256, fast.TYPE_UINT8, 1)
    with pytest.raises(RuntimeError):
        image2 = fast.GrayscaleToColor.create()(image, image)
    with pytest.raises(RuntimeError):
        image2 = fast.GrayscaleToColor.create()((1, image))

    convert = fast.GrayscaleToColor.create().connect(image)
    with pytest.raises(RuntimeError):
        image2 = fast.ColorToGrayscale.create()((0, convert, 3))
    with pytest.raises(ValueError):
        image2 = fast.ColorToGrayscale.create()((0, convert, 3, 2))
