from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from PIL import Image
import statistics

def approx_equal(n, m, epsilon: int | float = 10):
    return abs(n - m) < epsilon


class Rect:
    x: int
    y: int
    width: int
    height: int

    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)

    def __repr__(self) -> str:
        return (
            f"Rect(x: {self.x}, y:{self.y}, width: {self.width}, height: {self.height})"
        )

    def xmin(self) -> int:
        return self.x

    def ymin(self) -> int:
        return self.y

    def xmax(self) -> int:
        return self.x + self.width

    def ymax(self) -> int:
        return self.y + self.height

    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def intersects(self, other: Rect) -> bool:
        if self.xmin() > other.xmax():
            return False
        if other.xmin() > self.xmax():
            return False
        if self.ymin() > other.ymax():
            return False
        if other.ymin() > self.ymax():
            return False
        return True

    def merge(self, other: Rect) -> Rect:
        xmin = min(self.xmin(), other.xmin())
        ymin = min(self.ymin(), other.ymin())
        xmax = max(self.xmax(), other.xmax())
        ymax = max(self.ymax(), other.ymax())
        return Rect(xmin, ymin, xmax - xmin, ymax - ymin)

    def enlarge(self, pixels_x: int, pixels_y: int) -> Rect:
        new_x = self.x - pixels_x
        new_y = self.y - pixels_y
        new_width = self.width + pixels_x * 2
        new_height = self.height + pixels_y * 2
        return Rect(new_x, new_y, new_width, new_height)

    def verticality(self) -> float:
        return 1.0 - min(1.0, self.width / self.height)

    # def transpose(self) -> Rect:
    #     return Rect(self.x, self.y, self.height, self.width)

    def coords(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.xmax(), self.ymax())

    @staticmethod
    def from_points_clockwise(points: list[list[int]]) -> Rect:
        x = points[0][0]
        y = points[0][1]
        width = points[1][0] - x
        height = points[3][1] - y
        return Rect(x, y, width, height)

    @staticmethod
    def from_coords(coords: list[int]) -> Rect:
        [xmin, ymin, xmax, ymax] = coords
        width = xmax - xmin
        height = ymax - ymin
        return Rect(xmin, ymin, width, height)



class OCRResult:
    text: str
    bbox: Rect

    def __init__(self, text: str, bbox: Rect) -> None:
        self.text = text
        self.bbox = bbox

    def merge(self, other: OCRResult) -> OCRResult:
        text = self.text + " " + other.text
        bbox = self.bbox.merge(other.bbox)
        return OCRResult(text, bbox)

    def __repr__(self) -> str:
        return f"OCRResult({self.text}, {self.bbox})"

    def __lt__(self, other: OCRResult) -> bool:
        return OCRResult.cmp_lt(self.bbox, other.bbox, rtl=False)

    @staticmethod
    def from_cnocr_result(result: dict[str, Any]) -> OCRResult:
        text = result["text"]
        bbox = Rect.from_points_clockwise(result["position"])
        return OCRResult(text, bbox)

    @staticmethod
    def from_easyocr_result(result) -> OCRResult:
        (box, text, _) = result
        bbox = Rect.from_points_clockwise(box)
        return OCRResult(text, bbox)

    @staticmethod
    def from_pyocr_result(result) -> OCRResult:
        text = result.content
        ((xmin, ymin), (xmax, ymax)) = result.position
        bbox = Rect.from_coords([xmin, ymin, xmax, ymax])
        return OCRResult(text, bbox)

    @staticmethod
    def cmp_lt(item1: Rect, item2: Rect, rtl=False) -> bool:
        (_, center_y) = item1.center()
        (_, other_center_y) = item2.center()
        epsilon = max(item1.height, item2.height) / 4
        if approx_equal(center_y, other_center_y, epsilon=epsilon) or approx_equal(
            item1.y, item2.y, epsilon=epsilon
        ):
            return (item1.x < item2.x) ^ rtl

        return item1.y < item2.y

class OCRResultRTL(OCRResult):
    def __init__(self, text: str, bbox: Rect) -> None:
        super().__init__(text, bbox)

    def __lt__(self, other) -> bool:
        return OCRResult.cmp_lt(self.bbox, other.bbox, rtl=True)


def make_rtl(result: list[OCRResult]) -> list[OCRResultRTL]:
    return [OCRResultRTL(res.text, res.bbox) for res in result]

def verticality(results: list[OCRResult]) -> float:
    return statistics.mean([res.bbox.verticality() for res in results])

class OCR(ABC):
    @abstractmethod
    def ocr(self, image: Image.Image) -> list[OCRResult]:
        pass


class CnOCR(OCR):
    def __init__(self, vertical: bool = False) -> None:
        from cnocr import CnOcr

        if vertical:
            self._ocr = CnOcr(rec_model_name="ch_PP-OCRv5_server")
        else:
            self._ocr = CnOcr()

    def ocr(self, image: Image.Image) -> list[OCRResult]:
        results = self._ocr.ocr(image)
        return [OCRResult.from_cnocr_result(res) for res in results]


class EasyOCR(OCR):
    def __init__(self, langs: list[str], gpu=False) -> None:
        import easyocr
        self.reader = easyocr.Reader(
            langs,
            gpu=gpu,
        )

    def ocr(self, image: Image.Image) -> list[OCRResult]:
        results = self.reader.readtext(image)
        return [OCRResult.from_easyocr_result(res) for res in results]


class PyOCR(OCR):
    def __init__(self, lang) -> None:
        import pyocr

        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            raise ValueError(
                "PyOCR could not find any available OCR tool, install a supported tool or pick another OCR option"
            )

        if len(tools) > 1:
            print(f"Found {len(tools)} PyOCR compatible tools")
            for tool in tools:
                print(f"\t - {tool.get_name()}")

        self.tool = tools[0]
        print(f"Using PyOCR tool: {self.tool.get_name()}")

        langs: list[str] = self.tool.get_available_languages()
        if lang not in langs:
            print(
                f"'{lang}' is not a supported language. Either install a language plugin or pick a different language"
            )
            print(f"Currently supported languages: {', '.join(langs)}")
            raise ValueError(f"Unsupported language '{lang}'")

        self.lang = lang

    def ocr(self, image: Image.Image) -> list[OCRResult]:
        import pyocr.builders

        results: list[pyocr.builders.Box] = self.tool.image_to_string(
            image, lang=self.lang, builder=pyocr.builders.WordBoxBuilder()
        )
        return [OCRResult.from_pyocr_result(res) for res in results]


class PaddleOCR(OCR):
    # For some reason RaddleOCR uses two character language codes for most languages
    # except some are different...
    LANG_REPLACE = {"zh": "ch", "ko": "korean", "jp": "japanese"}

    def __init__(self, lang) -> None:
        from paddleocr import PaddleOCR as _PaddleOCR

        if lang in PaddleOCR.LANG_REPLACE:
            lang = PaddleOCR.LANG_REPLACE[lang]

        self._ocr = _PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=lang,
        )

    def ocr(self, image: Image.Image) -> list[OCRResult]:
        import numpy as np

        # PaddleOCR expects an RGB image
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        results = self._ocr.predict(np.asarray(image))

        ocr_results = []
        for res in results:
            for text, [x, y, xmax, ymax] in zip(res["rec_texts"], res["rec_boxes"]):
                width = xmax - x
                height = ymax - y
                bbox = Rect(x, y, width, height)
                ocr_results.append(OCRResult(text, bbox))

        return ocr_results
