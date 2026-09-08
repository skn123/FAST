import fast
import pytest


def test_callable_process_object():
    image = fast.Image(256, 256, fast.TYPE_UINT8, 1)
    image.fill(0)
    image2 = fast.GrayscaleToColor()(image)
    assert image2.getNrOfChannels() == 3

    image2 = fast.GrayscaleToColor()((0, image))
    assert image2.getNrOfChannels() == 3

    convert = fast.GrayscaleToColor().connect(image)
    image2 = fast.ColorToGrayscale()(convert)
    assert image2.getNrOfChannels() == 1

    image2 = fast.ColorToGrayscale()((0, convert, 0))
    assert image2.getNrOfChannels() == 1


def test_callable_process_object_exceptions():
    with pytest.raises(ValueError):
        image2 = fast.GrayscaleToColor()(32)
    with pytest.raises(RuntimeError):
        image2 = fast.GrayscaleToColor()()
    image = fast.Image(256, 256, fast.TYPE_UINT8, 1)
    with pytest.raises(RuntimeError):
        image2 = fast.GrayscaleToColor()(image, image)
    with pytest.raises(RuntimeError):
        image2 = fast.GrayscaleToColor()((1, image))

    convert = fast.GrayscaleToColor().connect(image)
    with pytest.raises(RuntimeError):
        image2 = fast.ColorToGrayscale()((0, convert, 3))
    with pytest.raises(ValueError):
        image2 = fast.ColorToGrayscale()((0, convert, 3, 2))
