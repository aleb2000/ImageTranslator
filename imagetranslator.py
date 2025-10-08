from __future__ import annotations
import argparse
import sys
from enum import Enum
import pathlib
from typing import Any, Callable, Literal
import numpy as np
from pypdf import PageObject, PdfWriter
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import puremagic
from ocr import (
    OCR,
    CnOCR,
    EasyOCR,
    OCRResult,
    PaddleOCR,
    PyOCR,
    Rect,
    make_rtl,
    verticality,
)
from translator import (
    ArgosTranslator,
    EasyNMTTranslator,
    Translator,
)

if sys.platform != "win32":
    from translator import BergamotTranslator


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
):
    """
    Adapted from https://gist.github.com/sushaanttb/f0799d60ab99ab07422aa1dfc8dd0fa9
    """
    lines = []
    current_line = ""

    for word in text.split():
        test_line = current_line + " " + word if current_line else word
        test_width = draw.textlength(test_line, font=font)

        if test_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    rect: Rect,
    font_path: str | None,
    font_color=(0, 0, 0),
    stroke_color=None,
):
    """
    Adapted from https://gist.github.com/sushaanttb/f0799d60ab99ab07422aa1dfc8dd0fa9
    """

    def create_font(size):
        if font_path != None:
            return ImageFont.truetype(font_path, size)
        else:
            return ImageFont.load_default(size=size)

    # Load the custom font and initialize font size
    font_size = 1
    font = create_font(font_size)

    # Calculate the maximum font size that fits in the rectangle
    while True:
        font = create_font(font_size)

        text_lines = wrap_text(draw, text, font, rect.width)  # Pass rectangle width
        total_height = sum(
            [
                draw.textbbox((0, 0), line, font=font)[3]
                - draw.textbbox((0, 0), line, font=font)[1]
                for line in text_lines
            ]
        )
        if (total_height) < rect.height and font_size < 100:  # Pass rectangle height
            font_size += 1
        else:
            font_size -= 1
            break

    # print(f"Optimal font size found: {font_size}")

    stroke_width = 0
    if stroke_color:
        stroke_width = max(1, int(font_size * 0.05))
        font_size = int(font_size * 0.95)

    # print(f"Stroke width: {stroke_width}")
    # print(f"Adjusted font size: {font_size}")

    font = create_font(font_size)

    # Calculate the position to start drawing text at the center of the rectangle
    x_center = rect.x + rect.width // 2
    y_center = rect.y + rect.height // 2
    x_start = x_center - rect.width // 2
    y_start = y_center - total_height // 2

    # Draw the wrapped text
    y_draw = y_start
    for line in text_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        draw_x = x_start + (rect.width - line_width) // 2
        draw.text(
            (draw_x, y_draw),
            line,
            fill=font_color,
            font=font,
            stroke_fill=stroke_color,
            stroke_width=stroke_width,
        )
        y_draw += bbox[3] - bbox[1]


def color_to_image_mode(image: Image.Image, color: tuple[int, ...]):
    r = color[0]
    g = color[1]
    b = color[2]
    if image.mode == "L":
        return int((r + g + b) // 3)
    return color


def groupby(elements: list[Any], predicate: Callable[[Any, Any], bool]) -> list[Any]:
    groups: list[list[OCRResult]] = [[elem] for elem in elements]

    i = 0
    while i < len(groups):
        group1 = groups[i]
        merge_with_idx = None
        j = i + 1
        for group2 in groups[i + 1 :]:
            for candidate1 in group1:
                for candidate2 in group2:
                    if predicate(candidate1, candidate2):
                        merge_with_idx = j
                        break
                if merge_with_idx != None:
                    break
            if merge_with_idx != None:
                break
            j += 1

        if merge_with_idx != None:
            group_to_merge = groups[merge_with_idx]
            group1.extend(group_to_merge)
            groups.remove(group_to_merge)
        else:
            i += 1

    return groups


class TextErasure(str, Enum):
    INPAINT = "inpaint"
    INPAINT_LAMA = "inpaint-lama"
    BLUR = "blur"
    NONE = "none"


class ImageTranslator:
    ocr: OCR
    translator: Translator
    text_erasure: TextErasure
    font_path: str | None

    def __init__(
        self,
        ocr: OCR,
        translator: Translator,
        text_erasure: TextErasure = TextErasure.INPAINT,
        lama_device: Literal["cpu", "cuda"] | None = None,
        font_path: str | None = None,
    ) -> None:
        from simple_lama_inpainting import SimpleLama
        import torch

        self.ocr = ocr
        self.translator = translator
        self.text_erasure = text_erasure
        self.font_path = font_path

        if self.text_erasure == TextErasure.INPAINT_LAMA:
            if lama_device == None:
                self.simple_lama = SimpleLama()
            else:
                self.simple_lama = SimpleLama(device=torch.device(lama_device))

    @staticmethod
    def _inpaint(image: Image.Image, ocr_results: list[OCRResult]) -> Image.Image:
        from skimage.restoration import inpaint

        mask = Image.new("1", image.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        for res in ocr_results:
            assert res.text.strip() != ""
            mask_draw.rectangle(res.bbox.coords(), fill=True)

        image_arr = np.asarray(image)
        mask_arr = np.asarray(mask, copy=True)
        inpainted = inpaint.inpaint_biharmonic(image_arr, mask_arr, channel_axis=-1)
        inpainted = (inpainted * 255).astype(np.uint8)
        return Image.fromarray(inpainted)

    def _inpaint_lama(
        self, image: Image.Image, ocr_results: list[OCRResult]
    ) -> Image.Image:
        assert self.simple_lama
        mask = Image.new("L", image.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        for res in ocr_results:
            assert res.text.strip() != ""
            mask_draw.rectangle(res.bbox.coords(), fill=255)
        return self.simple_lama(image, mask)

    @staticmethod
    def _cover_blur(image: Image.Image, ocr_results: list[OCRResult]):
        blurred = image.filter(ImageFilter.GaussianBlur(15))
        blur_mask = Image.new("L", image.size, 0)
        blur_mask_draw = ImageDraw.Draw(blur_mask)
        for res in ocr_results:
            assert res.text.strip() != ""
            blur_mask_draw.rectangle(res.bbox.coords(), fill=255)

        image.paste(blurred, mask=blur_mask)

    def translate(self, image: Image.Image, vertical_rtl: bool = False):
        from Pylette import extract_colors

        results = list(
            filter(lambda res: res.text.strip() != "", self.ocr.ocr(image)),
        )

        # Group together intersecting results
        intersections = groupby(
            results,
            lambda elem1, elem2: elem1.bbox.enlarge(10, 10).intersects(
                elem2.bbox.enlarge(10, 10)
            ),
        )

        # Then merge them
        recognitions: list[OCRResult] = []

        for group in intersections:
            if vertical_rtl and verticality(group) > 0.5:
                group = make_rtl(group)

            group.sort()
            merged = group[0]
            for res in group[1:]:
                merged = merged.merge(res)
            recognitions.append(merged)

        translated_image = image.copy()

        # First cover all the recognized text boxes using the selected technique
        text_erasure = self.text_erasure
        if text_erasure == TextErasure.INPAINT:
            try:
                translated_image = self._inpaint(translated_image, results)
            except ValueError as e:
                print(f"Inpainting failed with error: {e}")
                print("Falling back to blur")
                text_erasure = TextErasure.BLUR
        if text_erasure == TextErasure.INPAINT_LAMA:
            translated_image = self._inpaint_lama(translated_image, results)
        if text_erasure == TextErasure.BLUR:
            self._cover_blur(translated_image, results)

        # Translate text
        text_to_draw: list[tuple[OCRResult, str]] = list(
            zip(
                recognitions,
                self.translator.batch_translate([recog.text for recog in recognitions]),
            )
        )

        # Now draw the text
        draw = ImageDraw.Draw(translated_image)
        for recog, trans in text_to_draw:
            print(f"{recog.text} -> {trans}")

            # Figure out the text fill and stroke colors
            cropped = image.crop(recog.bbox.coords())
            palette = extract_colors(cropped, palette_size=2)
            text_color = tuple(palette[1].rgb)
            inverted_color = tuple(np.subtract((255, 255, 255), text_color))

            # Conversion to the correct mode could be improved, considering the palette already gives different modes
            text_color = color_to_image_mode(translated_image, text_color)
            inverted_color = color_to_image_mode(translated_image, inverted_color)

            draw_wrapped_text(
                draw,
                trans,
                recog.bbox,
                self.font_path,
                font_color=text_color,
                stroke_color=inverted_color,
            )

        return translated_image


def translate_page(
    image_translator: ImageTranslator,
    page: PageObject,
    vertical_rtl=False,
):
    for image in page.images:
        assert image.image
        translated_image = image_translator.translate(image.image, vertical_rtl)
        image.replace(translated_image)


def translate_pdf(
    image_translator: ImageTranslator,
    path,
    output_path,
    vertical_rtl=False,
):
    writer = PdfWriter(path)

    for page in writer.pages:
        translate_page(image_translator, page, vertical_rtl)

    with open(output_path, "wb") as fp:
        writer.write(fp)


def translate_image_file(
    image_translator: ImageTranslator,
    path,
    output_path,
    vertical_rtl=False,
):
    image = Image.open(path)
    translated_image = image_translator.translate(image, vertical_rtl)
    translated_image.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Translate text in images")

    parser.add_argument(
        "file",
        type=pathlib.Path,
        nargs="+",
        help="The image or image-containig PDF file to translate",
    )
    parser.add_argument(
        "--ocr",
        type=str,
        default="auto",
        help="The Optical Image Recognition engine to use to extract text from images. By default it will be chosen automatically depending on the source language. CnOCR whill be used for Chinese, while PaddleOCR for other supported languages, falling back to EasyOCR and PyOCR (frontend for Tesseract) when necessary.",
        choices=["auto", "cn", "paddle", "easy", "py"],
    )
    if sys.platform != "win32":
        translator_choices = [
            "argos",
            "bergamot",
            "opus",
            "mbart50",
            "m2m-100-418M",
            "m2m-100-1.2B",
        ]
    else:
        translator_choices = [
            "argos",
            "opus",
            "mbart50",
            "m2m-100-418M",
            "m2m-100-1.2B",
        ]
    parser.add_argument(
        "-t",
        "--translator",
        type=str,
        default="argos",
        help="The translation model to use. Argos is the lightest model to run and the default. Try different models and decide which one works best for your text.",
        choices=translator_choices,
    )

    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="Output path of the file, either an existing directory or the full file path",
    )
    parser.add_argument(
        "-s",
        "--source-lang",
        type=str,
        required=True,
        help="Language of the original text, in the form of the two character language code. E.g. en = English, ko = Korean, zh = Chinese, etc.",
    )
    parser.add_argument(
        "-d",
        "--target-lang",
        "--destination-lang",
        type=str,
        default="en",
        help="Language to translate into, in the form of the two character language code.",
    )
    parser.add_argument(
        "-e",
        "--text-erasure",
        type=str,
        default="inpaint",
        help="Technique used to erase the original text (default: 'inpaint'). Use 'none' to disable text erasure. 'inpaint' uses a biharmonic inpaint technique to fill in the background, which is fast but can have inaccurate results. 'inpaint-lama' uses the LaMa inpainting model, which can give more accurate results, but is much slower. 'blur' simply blurs the original text to cover it.",
        choices=["inpaint", "inpaint-lama", "blur", "none"],
    )
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--cpu",
        action="store_true",
        help="Forces use of CPU during translation, only works for models opus, mbart50, m2m-100-418M, m2m-100-1.2B",
    )
    device_group.add_argument(
        "--cuda",
        action="store_true",
        help="Forces use of CUDA during translation, only works with supported NVIDIA GPUs and for models opus, mbart50, m2m-100-418M, m2m-100-1.2B",
    )
    parser.add_argument(
        "--file-list",
        action="store_true",
        help="Instead of translating the input files, treat them as text files containing lists of files to translate. Each input argument must be a text file containing an image path on each line. Useful to pass on large lists of files by first writing them to file and then inputting them as file list.",
    )
    parser.add_argument(
        "-v",
        "--vertical",
        action="store_true",
        help="For the CnOCR model, enable vertical text recognition",
    )
    parser.add_argument(
        "--vertical-rtl",
        "--vertical-rigth-to-left",
        action="store_true",
        help="Translate vertical text in right to left order. It also autmatically enables vertical text recognition for the CnOCR model without the need to pass the --vertical argument.",
    )
    parser.add_argument(
        "--lama-device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        help="Only works when using the 'inpaing-lama' text erasure. Selects the device to use when running the LaMa model. By default, it will try to automatically select the most appropriate device on the system.",
    )
    parser.add_argument(
        "-f", "--font", type=pathlib.Path, help="Font to use for the translated text."
    )
    args = parser.parse_args()

    if args.output and len(args.file) > 1 and args.output.is_file():
        print("When translating multiple files the output argument must be a directory")
        sys.exit(1)

    if args.vertical_rtl:
        args.vertical = True

    print("Source language:", args.source_lang)
    print("Target language:", args.target_lang)

    easynmt_device = None
    if args.cpu:
        print("Using CPU")
        easynmt_device = EasyNMTTranslator.Device.CPU
    elif args.cuda:
        print("Using CUDA")
        easynmt_device = EasyNMTTranslator.Device.CUDA

    if args.translator == "argos":
        translator = ArgosTranslator(args.source_lang, args.target_lang)
    elif args.translator == "bergamot":
        translator = BergamotTranslator(args.source_lang, args.target_lang)
    elif args.translator == "opus":
        translator = EasyNMTTranslator(
            EasyNMTTranslator.Model.OPUS,
            device=easynmt_device,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
        )
    elif args.translator == "mbart50":
        translator = EasyNMTTranslator(
            EasyNMTTranslator.Model.MBART_50,
            device=easynmt_device,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
        )
    elif args.translator == "m2m-100-418M":
        translator = EasyNMTTranslator(
            EasyNMTTranslator.Model.M2M_100_418M,
            device=easynmt_device,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
        )
    elif args.translator == "m2m-100-1.2B":
        translator = EasyNMTTranslator(
            EasyNMTTranslator.Model.M2M_100_1_2B,
            device=easynmt_device,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
        )
    else:
        print(f"Unknown translator: {args.translator}")
        sys.exit(1)

    print(f"Initialized translator: {translator.name}")

    if args.ocr == "auto":
        if args.source_lang == "zh":
            ocr = CnOCR(args.vertical)
        else:
            ocr = PaddleOCR(args.source_lang)
        # TODO: additional fallback options for languages not supported by PaddleOCR. Is this necessary?
    elif args.ocr == "cn" or args.ocr == "cnocr":
        ocr = CnOCR(args.vertical)
    elif args.ocr == "paddle" or args.ocr == "paddleocr":
        ocr = PaddleOCR(args.source_lang)
    elif args.ocr == "easy" or args.ocr == "easyocr":
        ocr = EasyOCR([args.source_lang])
    elif args.ocr == "py" or args.ocr == "pyocr":
        ocr = PyOCR(args.source_lang)
    else:
        print(f"Unknown OCR engine: {args.ocr}")
        sys.exit(1)

    if args.text_erasure == "inpaint":
        text_erasure = TextErasure.INPAINT
    elif args.text_erasure == "inpaint-lama":
        text_erasure = TextErasure.INPAINT_LAMA
    elif args.text_erasure == "blur":
        text_erasure = TextErasure.BLUR
    elif args.text_erasure == "none":
        text_erasure = TextErasure.NONE
    else:
        print(f"Invalid text erasure technique: {args.text_erasure}")
        sys.exit(1)

    if args.file_list:
        files: list[pathlib.Path] = []
        for list_path in args.file:
            with open(list_path, "r") as fp:
                lines = fp.readlines()
                files.extend([pathlib.Path(line.strip()) for line in lines])
    else:
        files: list[pathlib.Path] = args.file

    if args.lama_device == "auto":
        lama_device = None
    else:
        lama_device = args.lama_device

    image_translator = ImageTranslator(
        ocr, translator, text_erasure, lama_device, args.font
    )

    for path in files:
        if not path.exists():
            print("File does not exist:", path)
            continue

        if path.is_dir():
            print("Skipping directory:", path)
            continue

        if args.output:
            output: pathlib.Path = args.output
            if output.is_dir() or output.suffix == "":
                output.mkdir(parents=True, exist_ok=True)
                output_path = output.joinpath(path.stem + ".translated" + path.suffix)
            else:
                output_path = output
        else:
            parent_dir = path.parent.resolve()
            output_path = parent_dir.joinpath(path.stem + ".translated" + path.suffix)

        print("Translating:", path)

        guesses = puremagic.magic_file(path)
        accepted_guess = None
        for guess in guesses:
            if guess.mime_type == "application/pdf":
                accepted_guess = guess
                translate_pdf(
                    image_translator,
                    path,
                    output_path,
                    vertical_rtl=args.vertical_rtl,
                )
            elif guess.mime_type.startswith("image/"):
                accepted_guess = guess
                translate_image_file(
                    image_translator,
                    path,
                    output_path,
                    vertical_rtl=args.vertical_rtl,
                )

            if accepted_guess:
                break

        if not accepted_guess:
            print("Unsupported file format:", path)
            if len(guesses) > 0:
                print(
                    "Hint: The file seems to be one of the following types, which are not supported by this program."
                )
                for guess in guesses:
                    print(f"\t- {guess.extension} ({guess.mime_type}): {guess.name}")
            continue

        print("Written translated file to:", output_path)


if __name__ == "__main__":
    main()
