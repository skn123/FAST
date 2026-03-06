import fast
import pytest


def test_callable_process_object():
    image = fast.Image.create(256, 256, fast.TYPE_UINT8, 1)
    image2 = fast.ImageResizer.create(512, 512)(image)
    assert image2.getWidth() == 512
    assert image2.getHeight() == 512

    image2 = fast.ImageResizer.create(512, 512)((0, image))
    assert image2.getWidth() == 512
    assert image2.getHeight() == 512

    resize = fast.ImageResizer.create(512, 512).connect(image)
    image2 = fast.GrayscaleToColor.create()(resize)
    assert image2.getWidth() == 512
    assert image2.getHeight() == 512
    assert image2.getNrOfChannels() == 3

    image2 = fast.GrayscaleToColor.create()((0, resize, 0))
    assert image2.getWidth() == 512
    assert image2.getHeight() == 512
    assert image2.getNrOfChannels() == 3


def test_callable_process_object_exceptions():
    with pytest.raises(ValueError):
        image2 = fast.ImageResizer.create(512, 512)(32)
    with pytest.raises(RuntimeError):
        image2 = fast.ImageResizer.create(512, 512)()
    image = fast.Image.create(256, 256, fast.TYPE_UINT8, 1)
    with pytest.raises(RuntimeError):
        image2 = fast.ImageResizer.create(512, 512)(image, image)
    with pytest.raises(RuntimeError):
        image2 = fast.ImageResizer.create(512, 512)((1, image))

    resize = fast.ImageResizer.create(512, 512).connect(image)
    with pytest.raises(RuntimeError):
        image2 = fast.GrayscaleToColor.create()((0, resize, 3))
    with pytest.raises(ValueError):
        image2 = fast.GrayscaleToColor.create()((0, resize, 3, 2))
