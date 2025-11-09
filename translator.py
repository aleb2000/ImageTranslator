from __future__ import annotations
from dataclasses import dataclass
import sys
import urllib.request
import json
import unicodedata
import re
import gzip
from pathlib import Path
import typing

from abc import ABC, abstractmethod
from enum import Enum
import platformdirs

from logger import get_logger


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

    def __init__(self, source_lang, target_lang="en") -> None:
        import argostranslate.package

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
