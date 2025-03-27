import os
import random
import numpy as np
from tqdm import tqdm
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen
from fontTools.pens.ttGlyphPen import TTGlyphPen

# Define the target characters to process: 0-9, A-Z, a-z
TARGET_CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# -----------------------------------------------------------------------------
# Custom pen to extract glyph outlines and record drawing commands
# -----------------------------------------------------------------------------
class HandwrittenPen(BasePen):
    def __init__(self, glyphSet):
        super().__init__(glyphSet)
        self.commands = []  # List of (command, points) tuples

    def _moveTo(self, p0):
        self.commands.append(('moveTo', p0))

    def _lineTo(self, p1):
        # Introduce a slight random offset to simulate a handwritten variation
        offset = (random.uniform(-10, 10), random.uniform(-10, 10))
        new_point = (p1[0] + offset[0], p1[1] + offset[1])
        self.commands.append(('lineTo', new_point))

    def _curveToOne(self, p1, p2, p3):
        # Introduce slight random perturbation for a handwritten effect
        p1_mod = (p1[0] + random.uniform(-10, 10), p1[1] + random.uniform(-10, 10))
        p2_mod = (p2[0] + random.uniform(-10, 10), p2[1] + random.uniform(-10, 10))
        p3_mod = (p3[0] + random.uniform(-10, 10), p3[1] + random.uniform(-10, 10))
        self.commands.append(('curveTo', (p1_mod, p2_mod, p3_mod)))

    def _closePath(self):
        self.commands.append(('closePath', None))

    def getCommands(self):
        return self.commands

# -----------------------------------------------------------------------------
# Transformation functions to simulate handwritten style
# -----------------------------------------------------------------------------
def reduce_stroke_thickness(commands, factor=0.8):
    transformed = []
    for cmd, pts in commands:
        if pts is None:
            transformed.append((cmd, pts))
        elif cmd in ['moveTo', 'lineTo']:
            transformed.append((cmd, (pts[0] * factor, pts[1] * factor)))
        elif cmd == 'curveTo':
            p1, p2, p3 = pts
            transformed.append((cmd, ((p1[0] * factor, p1[1] * factor),
                                        (p2[0] * factor, p2[1] * factor),
                                        (p3[0] * factor, p3[1] * factor))))
        else:
            transformed.append((cmd, pts))
    return transformed

def add_subtle_curvature(commands, max_bend=5):
    transformed = []
    for cmd, pts in commands:
        if pts is None:
            transformed.append((cmd, pts))
        elif cmd in ['moveTo', 'lineTo']:
            offset = (random.uniform(-max_bend, max_bend), random.uniform(-max_bend, max_bend))
            transformed.append((cmd, (pts[0] + offset[0], pts[1] + offset[1])))
        elif cmd == 'curveTo':
            p1, p2, p3 = pts
            p1_mod = (p1[0] + random.uniform(-max_bend, max_bend), p1[1] + random.uniform(-max_bend, max_bend))
            p2_mod = (p2[0] + random.uniform(-max_bend, max_bend), p2[1] + random.uniform(-max_bend, max_bend))
            p3_mod = (p3[0] + random.uniform(-max_bend, max_bend), p3[1] + random.uniform(-max_bend, max_bend))
            transformed.append((cmd, (p1_mod, p2_mod, p3_mod)))
        else:
            transformed.append((cmd, pts))
    return transformed

# -----------------------------------------------------------------------------
# Update glyph outline with transformed commands
# -----------------------------------------------------------------------------
def update_glyph(glyph, commands, glyphSet):
    pen = TTGlyphPen(glyphSet)
    for cmd, pts in commands:
        if cmd == 'moveTo':
            pen.moveTo(pts)
        elif cmd == 'lineTo':
            pen.lineTo(pts)
        elif cmd == 'curveTo':
            pen.curveTo(*pts)
        elif cmd == 'closePath':
            pen.closePath()
    new_glyph = pen.glyph()
    if new_glyph.numberOfContours > 0:
        glyph.coordinates, glyph.endPtsOfContours, glyph.flags = (
            new_glyph.coordinates,
            new_glyph.endPtsOfContours,
            new_glyph.flags
        )

# -----------------------------------------------------------------------------
# Process a single font: transform glyphs for TARGET_CHARACTERS and save new font
# -----------------------------------------------------------------------------
def transform_font(font_file, target_characters):
    font = TTFont(font_file)
    cmap = font.getBestCmap()  # mapping from Unicode to glyph names
    glyphSet = font.getGlyphSet()

    for char in target_characters:
        code = ord(char)
        glyph_name = cmap.get(code)
        if not glyph_name:
            print(f"Glyph for '{char}' not found in {font_file}")
            continue
        glyph = font["glyf"][glyph_name]
        if glyph.numberOfContours is None or glyph.numberOfContours <= 0:
            print(f"Glyph for '{char}' has no contours in {font_file}")
            continue

        pen = HandwrittenPen(glyphSet)
        try:
            # Pass the glyf table as second argument
            glyph.draw(pen, font["glyf"])
        except Exception as e:
            print(f"Error drawing glyph '{char}' in {font_file}: {e}")
            continue

        commands = pen.getCommands()
        commands = reduce_stroke_thickness(commands, factor=0.8)
        commands = add_subtle_curvature(commands, max_bend=5)
        update_glyph(glyph, commands, glyphSet)

    new_font_path = font_file.replace('.ttf', '_Handwritten.ttf').replace('.otf', '_Handwritten.otf')
    font.save(new_font_path)
    print(f"Saved transformed font to {new_font_path}")

# -----------------------------------------------------------------------------
# Process all fonts in the current directory
# -----------------------------------------------------------------------------
def process_all_fonts():
    font_files = [f for f in os.listdir('.') if f.lower().endswith(('.ttf', '.otf'))]
    for font_file in tqdm(font_files, desc="Processing Fonts", unit="font"):
        try:
            transform_font(font_file, TARGET_CHARACTERS)
        except Exception as e:
            print(f"Error processing {font_file}: {e}")

if __name__ == "__main__":
    process_all_fonts()
