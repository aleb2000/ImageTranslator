from typing import Tuple
from enum import Enum
from typing import List

class Language(List, Enum):
    # Japanese
    JP = ["jp", "ja"]

    # Chinese
    ZH = ["zh", "ch"]

    def __eq__(self, value: object, /) -> bool:
        if isinstance(value, str):
            return value in self
        return super().__eq__(value)

    def __ne__(self, value: object, /) -> bool:
        return not self == value


def correct_lang(lang: str | Language, lang_map: List[Tuple[Language | str, str]] | List[Tuple[Language, str]] | List[Tuple[str, str]]) -> str:
    for candidate, replacement in lang_map:
        if isinstance(candidate, Language) and lang in candidate:
            return replacement
        elif lang == candidate:
            return replacement

    if isinstance(lang, Language):
        return lang[0]

    return lang
