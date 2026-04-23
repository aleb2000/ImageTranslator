from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from PIL import Image, ImageDraw
import statistics

import cv2
from comic_text_detector.ctd_utils.basemodel import TextDetBaseDNN
from comic_text_detector.ctd_utils.textmask import refine_mask
from comic_text_detector.ctd_utils.utils.db_utils import SegDetectorRepresenter
from logger import get_logger
from language import Language, correct_lang
from resource_manager import RESOURCE_MANAGER
from util import letterbox, postprocess_mask
from comic_text_detector.utils import Quadrilateral, sort_pnts


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

    def scale(self, scale_x: float, scale_y: float) -> Rect:
        new_width = int(self.width * scale_x)
        new_height = int(self.height * scale_y)
        new_x = self.x - abs(new_width - self.width) // 2
        new_y = self.y - abs(new_height - self.height) // 2
        return Rect(int(new_x), int(new_y), new_width, new_height)

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
    vertical: bool | None

    def __init__(self, text: str, bbox: Rect, vertical: bool | None = None) -> None:
        self.text = text
        self.bbox = bbox
        self.vertical = vertical

    def merge(self, other: OCRResult | None) -> OCRResult:
        if other is None:
            return self
        text = self.text + " " + other.text
        bbox = self.bbox.merge(other.bbox)
        return OCRResult(text, bbox)

    def __repr__(self) -> str:
        return f"OCRResult({self.text}, {self.bbox})"

    def __lt__(self, other: OCRResult) -> bool:
        if self.is_vertical() or other.is_vertical():
            return OCRResult.cmp_lt_vertical(self.bbox, other.bbox, rtl=False)
        else:
            return OCRResult.cmp_lt(self.bbox, other.bbox, rtl=False)

    def is_vertical(self) -> bool:
        if self.vertical is not None:
            return self.vertical
        return self.bbox.verticality() > 0.5

    @staticmethod
    def from_cnocr_result(result: dict[str, Any]) -> OCRResult:
        text = result["text"]
        bbox = Rect.from_points_clockwise(result["position"])
        return OCRResult(text, bbox)

    @staticmethod
    def from_easyocr_result(result) -> OCRResult:
        box, text, _ = result
        bbox = Rect.from_points_clockwise(box)
        return OCRResult(text, bbox)

    @staticmethod
    def from_pyocr_result(result) -> OCRResult:
        text = result.content
        (xmin, ymin), (xmax, ymax) = result.position
        bbox = Rect.from_coords([xmin, ymin, xmax, ymax])
        return OCRResult(text, bbox)

    @staticmethod
    def cmp_lt(item1: Rect, item2: Rect, rtl=False) -> bool:
        _, center_y = item1.center()
        _, other_center_y = item2.center()
        # epsilon = max(item1.height, item2.height) / 4
        # if approx_equal(center_y, other_center_y, epsilon=epsilon) or approx_equal(
        #     item1.y, item2.y, epsilon=epsilon
        # ):
        #     return (item1.x < item2.x) ^ rtl

        # return item1.y < item2.y
        return center_y < other_center_y

    @staticmethod
    def cmp_lt_vertical(item1: Rect, item2: Rect, rtl=False) -> bool:
        center_x, _ = item1.center()
        other_center_x, _ = item2.center()
        # epsilon = max(item1.height, item2.height) / 4
        # if approx_equal(center_x, other_center_x, epsilon=epsilon) or approx_equal(
        #     item1.x, item2.x, epsilon=epsilon
        # ):
        #     return (item1.y > item2.y) ^ rtl
        #
        # return item1.x > item2.x

        return (center_x < other_center_x) ^ rtl

    @staticmethod
    def merge_all(results: list[OCRResult]) -> OCRResult | None:
        if len(results) == 0:
            return None
        if len(results) == 1:
            return results[0]

        merged = results[0]
        for res in results[1:]:
            merged = merged.merge(res)
        return merged


class OCRResultRTL(OCRResult):
    def __init__(self, text: str, bbox: Rect, vertical: bool | None = None) -> None:
        super().__init__(text, bbox, vertical)

    def __lt__(self, other) -> bool:
        if self.is_vertical() or other.is_vertical():
            return OCRResult.cmp_lt_vertical(self.bbox, other.bbox, rtl=True)
        else:
            return OCRResult.cmp_lt(self.bbox, other.bbox, rtl=True)


def make_rtl(result: list[OCRResult]) -> list[OCRResultRTL]:
    return [OCRResultRTL(res.text, res.bbox, res.vertical) for res in result]


def mostly_vertical(results: list[OCRResult]) -> bool:
    outcomes = map(lambda res: res.is_vertical(), results)
    return statistics.mode(outcomes)


def detect_text_lines(
    results: list[OCRResult] | list[OCRResultRTL], rtl: bool
) -> list[list[OCRResult]]:
    import numpy as np

    def find_anchor(result, direction):
        return np.array(result.bbox.center()) + direction * np.array(
            [result.bbox.width // 4, result.bbox.height // 4]
        )

    def normalized(vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return 0
        return vector / norm

    def projection(a, b):
        return np.dot(a, b) / np.linalg.norm(b)

    lines = []
    vertical = mostly_vertical(results)
    if vertical:
        direction = np.array([0, 1])
    elif rtl:
        direction = np.array([-1, 0])
    else:
        direction = np.array([1, 0])
    print(direction)

    perpendicular_direction = np.array([direction[1], -direction[0]])

    # FIXME: Sorting and taking the first one is unreliable, need a different approach
    results.sort()
    head = results.pop(0)

    line = []
    while len(results) > 0:
        assert head
        head_point = find_anchor(head, direction)

        chosen = None
        chosen_dist = float("inf")

        for res in results:
            res_point = find_anchor(res, -direction)
            distance_vector = res_point - head_point
            angle = np.acos(np.dot(direction, normalized(distance_vector)))
            orthogonal_distance = np.linalg.norm(distance_vector * np.sin(angle))

            # Find the threshold based on the size of the box and the line direction
            width_vector = np.array([head.bbox.width, 0])
            height_vector = np.array([0, head.bbox.height])
            perpendicular_width_component = np.linalg.norm(
                projection(width_vector, perpendicular_direction)
            )
            perpendicular_height_component = np.linalg.norm(
                projection(height_vector, perpendicular_direction)
            )
            assert (
                perpendicular_width_component >= 0
                and perpendicular_height_component >= 0
            )
            threshold = perpendicular_width_component + perpendicular_height_component

            if orthogonal_distance < threshold:
                distance = np.linalg.norm(distance_vector)
                if distance < chosen_dist:
                    chosen = res
                    chosen_dist = distance

        line.append(head)
        if chosen:
            results.remove(chosen)
            head = chosen
        else:
            lines.append(line)
            head = None
            line = []
            if len(results) > 0:
                results.sort()
                head = results.pop(0)

    if head is not None:
        line.append(head)

    if len(line) > 0 and line not in lines:
        lines.append(line)

    print(lines)

    return lines


def draw_text_lines_marker(image: Image.Image, lines: list[list[OCRResult]]):
    draw = ImageDraw.Draw(image)
    for line in lines:
        for i in range(len(line) - 1):
            a = line[i]
            b = line[i + 1]
            center_a = a.bbox.center()
            center_b = b.bbox.center()
            draw.line([center_a, center_b], "blue", 3)
            draw.circle(center_a, 5, "blue")
            draw.circle(center_b, 5, "blue")


class OCR(ABC):
    @abstractmethod
    def ocr(self, image: Image.Image) -> tuple[list[OCRResult], Image.Image | None]:
        pass


class CnOCR(OCR):
    def __init__(self, lang: str, vertical: bool = False) -> None:
        from cnocr import CnOcr

        if lang == Language.ZH and vertical:
            self._ocr = CnOcr(rec_model_name="ch_PP-OCRv5_server")
        elif lang == Language.JP:
            self._ocr = CnOcr(rec_model_name="japan_PP-OCRv3")
        else:
            self._ocr = CnOcr()

    def ocr(self, image: Image.Image) -> tuple[list[OCRResult], Image.Image | None]:
        results = self._ocr.ocr(image)
        return [OCRResult.from_cnocr_result(res) for res in results], None


class EasyOCR(OCR):
    LANG_MAP = [(Language.JP, "ja")]

    def __init__(self, langs: list[str], gpu=False) -> None:
        import easyocr

        langs = list(map(lambda lang: correct_lang(lang, EasyOCR.LANG_MAP), langs))
        self.reader = easyocr.Reader(
            langs,
            gpu=gpu,
        )

    def ocr(self, image: Image.Image) -> tuple[list[OCRResult], Image.Image | None]:
        import numpy as np

        results = self.reader.readtext(np.asarray(image))
        return [OCRResult.from_easyocr_result(res) for res in results], None


class PyOCR(OCR):
    vertical: bool

    LANG_MAP = [(Language.JP, "jpn")]

    def __init__(self, lang, vertical=False) -> None:
        import pyocr

        self.vertical = vertical

        lang = correct_lang(lang, PyOCR.LANG_MAP)
        if vertical:
            lang = lang + "_vert"

        l = get_logger("PYOCR")  # noqa: E741

        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            raise ValueError(
                "PyOCR could not find any available OCR tool, install a supported tool or pick another OCR option"
            )

        if len(tools) > 1:
            l.info(f"Found {len(tools)} PyOCR compatible tools")
            for tool in tools:
                l.info(f"\t - {tool.get_name()}")

        self.tool = tools[0]
        l.info(f"Using PyOCR tool: {self.tool.get_name()}")

        langs: list[str] = self.tool.get_available_languages()
        if lang not in langs:
            l.error(
                f"'{lang}' is not a supported language. Either install a language plugin or pick a different language"
            )
            l.error(f"Currently supported languages: {', '.join(langs)}")
            raise ValueError(f"Unsupported language '{lang}'")

        self.lang = lang

    def ocr(self, image: Image.Image) -> tuple[list[OCRResult], Image.Image | None]:
        import pyocr.builders

        box_results: list[pyocr.builders.Box] = self.tool.image_to_string(
            image, lang=self.lang, builder=pyocr.builders.WordBoxBuilder()
        )
        results = [OCRResult.from_pyocr_result(res) for res in box_results]
        for res in results:
            res.bbox = res.bbox.enlarge(10, 10)
            res.vertical = self.vertical
        return results, None


class PaddleOCR(OCR):
    # For some reason RaddleOCR uses two character language codes for most languages
    # except some are different...
    LANG_MAP = [(Language.ZH, "ch"), ("ko", "korean"), (Language.JP, "japan")]

    def __init__(self, lang) -> None:
        from paddleocr import PaddleOCR as _PaddleOCR

        lang = correct_lang(lang, PaddleOCR.LANG_MAP)

        self._ocr = _PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=lang,
        )

    def ocr(self, image: Image.Image) -> tuple[list[OCRResult], Image.Image | None]:
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

        return ocr_results, None


class MangaOCR(OCR):
    CTD_MODEL_URL = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt.onnx"
    CTD_INPUT_SIZE = (1024, 1024)

    def __init__(self, lang) -> None:
        from manga_ocr import MangaOcr

        textdetector_file = RESOURCE_MANAGER.get(
            "comictextdetector.pt.onnx",
            MangaOCR.CTD_MODEL_URL,
        )
        self.ctd_model = TextDetBaseDNN(self.CTD_INPUT_SIZE[0], textdetector_file)
        self.seg_rep = SegDetectorRepresenter(thresh=0.3)

        self._ocr = MangaOcr(force_cpu=True)

    def ocr(self, image: Image.Image) -> tuple[list[OCRResult], Image.Image | None]:
        textlines, mask = self.ctd(image)
        mask_img = Image.fromarray(mask)
        results = []
        for textline in textlines:
            textline = textline.enlarge(10, 10)
            region = image.crop(textline.coords())
            text = self._ocr(region)
            results.append(OCRResult(text, textline))
        return results, mask_img

    def ctd(self, image: Image.Image):
        """
        Run Comic Text Detector
        """
        import numpy as np
        import cv2

        im_h, im_w = image.height, image.width

        if image.mode != "RGB":
            image = image.convert("RGB")

        img = np.asarray(image, dtype=np.uint8)

        img_in, ratio, dw, dh = self.ctd_preprocess_image(img)
        # blks, mask, lines_map = self.ctd_session.run(None, {'images': img})
        blks, mask, lines_map = self.ctd_model(img_in)
        if mask.shape[1] == 2:  # some version of opencv spit out reversed result
            tmp = mask
            mask = lines_map
            lines_map = tmp
        assert isinstance(blks, np.ndarray)
        assert isinstance(mask, np.ndarray)
        assert isinstance(lines_map, np.ndarray)

        mask = mask.squeeze()
        mask = mask[..., : mask.shape[0] - dh, : mask.shape[1] - dw]
        lines_map = lines_map[..., : lines_map.shape[2] - dh, : lines_map.shape[3] - dw]

        mask = postprocess_mask(mask)

        lines, scores = self.seg_rep(None, lines_map, height=im_h, width=im_w)
        box_thresh = 0.6
        idx = np.where(scores[0] > box_thresh)
        lines, scores = lines[0][idx], scores[0][idx]

        # map output to input img
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)

        textlines_rects = [
            self._rect_from_pnts(pts.astype(int)) for pts, score in zip(lines, scores)
        ]
        textlines = [
            Quadrilateral(pts.astype(int), "", score)
            for pts, score in zip(lines, scores)
        ]
        mask_refined = refine_mask(img, mask, textlines, refine_mode=None)
        return textlines_rects, mask_refined

    def ctd_preprocess_image(self, img):
        """
        Comic Text Detector wants a 1024 by 1024 image tensor with shape color, height, width
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, ratio, (dw, dh) = letterbox(
            img, new_shape=MangaOCR.CTD_INPUT_SIZE, auto=False, stride=64
        )
        return img, ratio, int(dw), int(dh)

    @staticmethod
    def _rect_from_pnts(pts):
        pts, is_vertical = sort_pnts(pts)
        x = min(pts[0][0], pts[3][0])
        y = min(pts[0][1], pts[1][1])
        xmax = max(pts[1][0], pts[2][0])
        ymax = max(pts[2][1], pts[3][1])
        return Rect(x, y, xmax - x, ymax - y)
