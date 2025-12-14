from typing import Tuple
from enum import Enum
from typing import List


class LangName(List, Enum):
    # Japanese
    JP = ["ja", "jp"]

    # Chinese
    ZH = ["zh", "ch"]

def correct_lang(lang: str, lang_map: List[Tuple[LangName | str, str]] | List[Tuple[LangName, str]] | List[Tuple[str, str]]) -> str:
    for candidate, replacement in lang_map:
        if isinstance(candidate, LangName) and lang in candidate:
            return replacement
        elif lang == candidate:
            return replacement

    return lang

