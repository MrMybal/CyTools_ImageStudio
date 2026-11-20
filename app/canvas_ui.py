"""CyUI zoom/pan canvas with a native-resolution mask document and bounded previews."""
from collections import deque
from pathlib import Path
from uuid import uuid4

from imgui_bundle import imgui
from OpenGL import GL
from PIL import Image, ImageDraw, ImageOps

import cyui
from cyui_modules.canvas import Canvas, CanvasState
from settings import DATA


class Texture:
    def __init__(self):
        self.handle = 0

    def set(self, image):
        if not self.handle:
            self.handle = int(GL.glGenTextures(1))
        image = image.convert("RGBA")
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.handle)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, image.width, image.height, 0, GL.GL_RGBA,
                       GL.GL_UNSIGNED_BYTE, image.tobytes())
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def ref(self):
        return imgui.ImTextureRef(self.handle)

    def close(self):
        if self.handle:
            GL.glDeleteTextures([self.handle])
            self.handle = 0


class ImageCanvas:
    def __init__(self):
        self.state = CanvasState()
        self.image = Texture()
        self.overlay = Texture()
        self.path = None
        self.size = (0, 0)
        self.mask = None
        self.mask_path = ""
        self.undo = deque(maxlen=5)
        self.dirty = False
        self.mask_changed = False
        self.fit = True
        self.last = None
        self.brush = 34
        self.erase = False

    def load(self, path):
        path = Path(path)
        with Image.open(path) as source:
            if source.width * source.height > 40_000_000:
                raise ValueError("Image exceeds 40 megapixels.")
            preview = ImageOps.exif_transpose(source).convert("RGB")
            preview.thumbnail((3072, 3072))
        self.image.set(preview)
        self.path = path
        self.size = preview.size
        self.mask = Image.new("L", self.size)
        self.mask_path = ""
        self.undo.clear()
        self.dirty = True
        self.mask_changed = False
        self.fit = True

    def load_mask(self, path):
        if not self.path:
            raise ValueError("Load the source image before importing a mask.")
        with Image.open(path) as bitmap:
            if bitmap.width * bitmap.height > 40_000_000:
                raise ValueError("Mask exceeds 40 megapixels.")
            self.undo.append(self.mask.copy())
            self.mask = ImageOps.exif_transpose(bitmap).convert("L").resize(self.size, Image.Resampling.NEAREST)
        self.dirty = True
        self.mask_changed = True

    def reset(self):
        self.path = None
        self.mask = None
        self.mask_path = ""
        self.undo.clear()
        self.image.close()
        self.overlay.close()

    def clear(self):
        if self.mask is not None:
            self.undo.append(self.mask.copy())
            self.mask.paste(0, (0, 0, *self.size))
            self.dirty = True
            self.mask_changed = True

    def undo_mask(self):
        if self.undo:
            self.mask = self.undo.pop()
            self.dirty = True
            self.mask_changed = True

    def save_mask(self):
        if self.mask is None or self.mask.getbbox() is None:
            return ""
        if self.mask_changed or not self.mask_path:
            path = DATA / "inputs" / "canvas" / (uuid4().hex + ".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            self.mask.save(path)
            self.mask_path = str(path)
            self.mask_changed = False
        return self.mask_path

    def close(self):
        self.overlay.close()
        self.image.close()

    def draw(self, *, paint=False, empty="No image"):
        with Canvas("##image-canvas", self.state, size=(0, 0)) as canvas:
            if self.path is None:
                width = imgui.calc_text_size(empty).x
                center = (canvas.top[0] + (canvas.size[0] - width) / 2, canvas.top[1] + canvas.size[1] / 2)
                canvas.draw.add_text(center, cyui.palette().textDim, empty)
                return
            if self.fit:
                self.state.view.fit((0, 0), self.size, canvas.size)
                self.fit = False
            view = self.state.view
            top, bottom = view.screen((0, 0), canvas.top), view.screen(self.size, canvas.top)
            canvas.draw.add_image(self.image.ref(), top, bottom)
            if not paint:
                self.last = None
                return
            self.stroke(canvas, top, bottom)
            if self.dirty:
                layer = Image.new("RGBA", self.size, (175, 130, 255, 0))
                layer.putalpha(self.mask.point(lambda value: value * 0.52))
                self.overlay.set(layer)
                self.dirty = False
            canvas.draw.add_image(self.overlay.ref(), top, bottom)

    def stroke(self, canvas, top, bottom):
        if not (canvas.hovered and all(0 <= canvas.mouse[i] < self.size[i] for i in (0, 1))):
            self.last = None
            return
        position = tuple(round(x) for x in canvas.mouse)
        cursor = tuple(imgui.get_io().mouse_pos)
        canvas.draw.add_circle(cursor, max(2, self.brush / 2 * self.state.view.zoom), 0xFFCBB2FF, 0, 1.5)
        if canvas.state.panning or not imgui.is_mouse_down(0) or imgui.is_key_down(imgui.Key.space):
            self.last = None
            return
        if self.last is None:
            if not imgui.is_mouse_clicked(0):
                return
            self.undo.append(self.mask.copy())
            self.last = position
        draw = ImageDraw.Draw(self.mask)
        fill = 0 if self.erase else 255
        draw.line((self.last, position), fill=fill, width=self.brush)
        x, y = position
        radius = self.brush // 2
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
        self.last = position
        self.dirty = self.mask_changed = True
