"""Safe image loading for uploaded or camera-captured photos.

Images are handled in memory only. We never save the uploaded file to disk.
The file type is decided by looking at the image bytes, NOT the file name.
"""
import io

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 8 * 1024 * 1024        # 8 MB
MAX_PIXELS = 40_000_000                   # protects against "decompression bombs"
MAX_SIDE = 1024                           # larger photos are shrunk (the model uses 224x224)
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageRejected(Exception):
    """The upload cannot be used. `status` is the HTTP status code to return."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


def load_image(data):
    """Validate raw bytes and return a clean RGB PIL image."""
    if not data:
        raise ImageRejected("The file is empty. Choose a photo and try again.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageRejected("That image is larger than 8 MB. Choose a smaller photo.", 413)

    try:
        probe = Image.open(io.BytesIO(data))
        image_format = probe.format
        width, height = probe.size
    except (UnidentifiedImageError, OSError, ValueError):
        raise ImageRejected("That file is not a valid image. Use a JPG, PNG or WebP photo.", 415)

    if image_format not in ALLOWED_FORMATS:
        raise ImageRejected("Unsupported image type. Use a JPG, PNG or WebP photo.", 415)
    if width * height > MAX_PIXELS:
        raise ImageRejected("That image has too many pixels. Choose a smaller photo.", 413)

    try:
        probe.verify()                           # checks the file is not corrupt
        image = Image.open(io.BytesIO(data))     # verify() leaves the image unusable, so reopen
        image = ImageOps.exif_transpose(image)   # respect phone camera rotation
        image = image.convert("RGB")             # drops transparency / palette modes
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise ImageRejected("That image looks corrupted. Try another photo.", 415)

    image.thumbnail((MAX_SIDE, MAX_SIDE))
    return image
