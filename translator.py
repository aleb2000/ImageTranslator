from __future__ import annotations
import pathlib as pl
from dataclasses import dataclass
import sys
import tarfile
import urllib.request
import json
import unicodedata
import re
import gzip
from pathlib import Path
import typing
from typing import Dict

from abc import ABC, abstractmethod
from enum import Enum
import zipfile
import ctranslate2
import sentencepiece as spm
import platformdirs

from logger import get_logger
from language import Language, correct_lang
from resource_manager import RESOURCE_MANAGER


class Translator(ABC):
    name: str

    @abstractmethod
    def __init__(self, name) -> None:
        self.name = name

    @abstractmethod
    def translate(self, text: str) -> str:
        pass

    @abstractmethod
    def batch_translate(self, texts: list[str]) -> list[str]:
        return [self.translate(text) for text in texts]


def space_punctuation(text):
    other_spacing_characters = ["～", "~"]

    def char_should_be_spaced(c: str):
        return unicodedata.category(c).startswith("P") or c in other_spacing_characters

    spaced_text = ""
    i = 0
    for i in range(0, len(text)):
        c = text[i]
        spaced_text += c
        if char_should_be_spaced(c) and i < len(text) - 1:
            next = text[i + 1]
            if next != " " and not char_should_be_spaced(next):
                spaced_text += " "

    return spaced_text


class ArgosTranslator(Translator):
    source_lang: str
    target_lang: str

    LANG_MAP = [(Language.JP, "ja")]

    def __init__(self, source_lang, target_lang="en") -> None:
        import argostranslate.package

        source_lang = correct_lang(source_lang, ArgosTranslator.LANG_MAP)
        target_lang = correct_lang(target_lang, ArgosTranslator.LANG_MAP)

        l = get_logger("ARGOS")  # noqa: E741

        super().__init__("argos")
        self.source_lang = source_lang
        self.target_lang = target_lang

        installed_packages = argostranslate.package.get_installed_packages()
        trans_packages = self.find_packages_for_translation(installed_packages)
        if trans_packages is None:
            l.info(
                "Could not find appropriate Argos packages installed, fetching from remote repository..."
            )
            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            trans_packages = self.find_packages_for_translation(available_packages)
            if trans_packages is None:
                raise ValueError(
                    f"Failed to find appropriate Argos packages for translation from {self.source_lang} to {self.target_lang}"
                )

            # We only need to download the packages that are not already installed
            def pkg_not_installed(pkg):
                return all(pkg != installed for installed in installed_packages)

            for package in filter(
                pkg_not_installed,
                trans_packages,
            ):
                assert isinstance(package, argostranslate.package.AvailablePackage)
                l.info(f"Argos package {package} not installed, downloading...")
                package.install()

        # Check that the packages are installed
        installed_packages = argostranslate.package.get_installed_packages()
        trans_packages = self.find_packages_for_translation(installed_packages)
        assert (
            trans_packages is not None
        ), "Somehow package installation failed without error"
        for package in trans_packages:
            l.info(f"Using Argos package: {package}")

    def translate(self, text: str) -> str:
        import argostranslate.translate

        return argostranslate.translate.translate(
            text, self.source_lang, self.target_lang
        )

    def batch_translate(self, texts: list[str]) -> list[str]:
        return super().batch_translate(texts)

    def find_packages_for_translation(self, repo) -> list | None:
        """Find translation packages, either by direct translation or by pivoting through another language"""
        import argostranslate.package

        pivots: dict[
            str,
            list[argostranslate.package.IPackage],
        ] = {}
        direct = None
        for package in repo:
            if (
                package.from_code == self.source_lang
                and package.to_code == self.target_lang
            ):
                direct = package
                break

            if package.from_code == self.source_lang:
                pivots.setdefault(package.to_code, []).insert(0, package)

            if package.to_code == self.target_lang:
                pivots.setdefault(package.from_code, []).insert(1, package)

        if direct:
            trans_packages = [direct]
        else:
            try:
                trans_packages = next(
                    filter(lambda packages: len(packages) == 2, pivots.values())
                )
            except StopIteration:
                return None
        return trans_packages


class EasyNMTTranslator(Translator):
    source_lang: str | None
    target_lang: str

    LANG_MAP = [(Language.JP, "ja")]

    class Model(str, Enum):
        OPUS = "opus-mt"
        MBART_50 = "mbart50_m2m"
        M2M_100_418M = "m2m_100_418M"
        M2M_100_1_2B = "m2m_100_1.2B"

    class Device(str, Enum):
        CPU = "cpu"
        CUDA = "cuda"

    def __init__(
        self,
        model_name: Model,
        device: Device | None = None,
        source_lang: str | None = None,
        target_lang: str = "en",
    ) -> None:
        from easynmt.EasyNMT import EasyNMT
        import nltk

        super().__init__(model_name)
        nltk.download("punkt_tab")

        if source_lang:
            source_lang = correct_lang(source_lang, EasyNMTTranslator.LANG_MAP)
        target_lang = correct_lang(target_lang, EasyNMTTranslator.LANG_MAP)

        self.model = EasyNMT(model_name, device=device)
        self.source_lang = source_lang
        self.target_lang = target_lang

    # The source_lang can be None, but whoever defined the argument types is a dumbass and made it str only
    # Hence I need to manually shut up the type checker
    @typing.no_type_check
    def translate(self, text: str) -> str:
        trans = self.model.translate(
            text, target_lang=self.target_lang, source_lang=self.source_lang
        )
        return trans

    @typing.no_type_check
    def batch_translate(self, texts: list[str]) -> list[str]:
        # This should be a list of strings, but the type checker does not believe me
        trans = self.model.translate(
            texts, target_lang=self.target_lang, source_lang=self.source_lang
        )
        return trans


class SugoiTranslator(Translator):
    """
    Model and implementation obtained by https://github.com/zyddnys/manga-image-translator
    """

    model: ctranslate2.Translator
    sentencepiece_processors: Dict[str, spm.SentencePieceProcessor]

    LANGUAGE_MAPPING = [(Language.JP, "ja")]
    MODEL_DOWNLOAD_URL = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/sugoi-models.zip"

    def __init__(
        self,
        source_lang: Language | str = "jp",
        target_lang: str = "en",
    ) -> None:
        super().__init__("sugoi")

        l = get_logger("SUGOI")  # noqa: E741

        if source_lang != Language.JP or target_lang != "en":
            raise ValueError(
                f"sugoi: Unsupported language pair {source_lang}->{target_lang}"
            )

        self.source_lang = correct_lang(source_lang, SugoiTranslator.LANGUAGE_MAPPING)
        self.target_lang = correct_lang(target_lang, SugoiTranslator.LANGUAGE_MAPPING)

        model_dir = RESOURCE_MANAGER.path("sugoi-models")

        if not model_dir.exists():
            l.info(f"Downloading model {self.source_lang}->{self.target_lang}")

            model_zip_path = RESOURCE_MANAGER.get(
                f"{model_dir.name}.zip",
                SugoiTranslator.MODEL_DOWNLOAD_URL,
            )
            with zipfile.ZipFile(model_zip_path, "r") as fp:
                fp.extractall(model_dir)
            model_zip_path.unlink()

        self.model = ctranslate2.Translator(
            str(model_dir / f"big-{self.source_lang}-{self.target_lang}")
        )
        self.model.load_model()

        self.sentencepiece_processors = {
            self.target_lang: spm.SentencePieceProcessor(
                model_file=str(model_dir / f"spm.{self.target_lang}.nopretok.model")
            ),
            self.source_lang: spm.SentencePieceProcessor(
                model_file=str(model_dir / f"spm.{self.source_lang}.nopretok.model")
            ),
        }

    def _tokenize(self, queries: str | list[str], lang: str):
        processor = self.sentencepiece_processors[lang]

        if isinstance(queries, list):
            return processor.encode(queries, out_type=str)
        else:
            return [processor.encode(queries, out_type=str)]

    def _detokenize(self, queries: list[str], lang: str):
        processor = self.sentencepiece_processors[lang]

        translation = processor.decode(queries)
        return translation

    def translate(self, text: str) -> str:
        return self.batch_translate([text])[0]

    def batch_translate(self, texts: list[str]) -> list[str]:
        queries_tokenized = self._tokenize(texts, self.source_lang)
        translated_tokenized = self.model.translate_batch(
            source=queries_tokenized,
            beam_size=5,
            num_hypotheses=1,
            return_alternatives=False,
            disable_unk=True,
            replace_unknowns=True,
            repetition_penalty=3,
        )
        translated = self._detokenize(
            list(map(lambda t: t[0]["tokens"], translated_tokenized)), self.target_lang
        )
        return translated
