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


if sys.platform != "win32":

    class BergamotTranslator(Translator):
        URL_MODELS_BY_HASH = "https://raw.githubusercontent.com/mozilla/firefox-translations-models/refs/heads/main/models/by-hash.json"
        RE_METADATA_PATH = re.compile(
            r"models/([-A-Za-z_]+)/([a-z]{2})([a-z]{2})/metadata\.json"
        )
        RE_LANGS_DIR = re.compile(r"([a-z]{2})([a-z]{2})")
        URL_FIREFOX_REPO_PREFIX = "https://github.com/mozilla/firefox-translations-models/raw/refs/heads/main/"

        @typing.no_type_check
        def __init__(
            self,
            source_lang,
            target_lang="en",
            models_install_path: Path = platformdirs.user_data_path(
                "imagetranslator", "aleb2000"
            ).joinpath("bergamot"),
        ) -> None:
            super().__init__("bergamot")
            l = get_logger("BERGAMOT") # noqa: E741

            self.source_lang = source_lang
            self.target_lang = target_lang
            import bergamot

            models_install_path.mkdir(parents=True, exist_ok=True)
            installed_models = BergamotTranslator.installed_models(models_install_path)
            model = self.choose_model(installed_models)
            if not model:
                l.info(
                    "Could not find locally installed model, fetching from Firefox repository..."
                )
                models = BergamotTranslator.fetch_models_from_repo()
                remote_model = self.choose_model(models)
                if not remote_model:
                    raise ValueError(
                        f"Could not find appropriate translation model for {source_lang} -> {target_lang}"
                    )
                l.info("Found suitable model, downloading...")
                model = remote_model.download(models_install_path)
                l.info("Downloads finished")

            l.info(f"Using model: {model.description()}")
            if not model.has_config():
                model.create_config()

            service_config = bergamot.ServiceConfig(numWorkers=1, logLevel="info")
            self.service = bergamot.Service(service_config)
            self.model = self.service.modelFromConfigPath(str(model.config_path()))
            self.response_options = bergamot.ResponseOptions(
                alignment=False, qualityScores=False, HTML=False
            )

        @typing.no_type_check
        def translate(self, text: str) -> str:
            return self.batch_translate([text])[0]

        @typing.no_type_check
        def batch_translate(self, texts: list[str]) -> list[str]:
            import bergamot

            responses = self.service.translate(
                self.model, bergamot.VectorString(texts), self.response_options
            )
            return [response.target.text for response in responses]

        def choose_model(
            self,
            models: typing.Sequence[Model],
            preference=["base", "base-memory", "tiny"],
        ) -> Model | None:
            correct_lang_models = list(
                filter(
                    lambda m: m.source_lang == self.source_lang
                    and m.target_lang == self.target_lang,
                    models,
                )
            )

            sort_order = {_type: i for (i, _type) in enumerate(preference)}
            correct_lang_models.sort(
                key=lambda model: (
                    sort_order[model._type]
                    if model._type in preference
                    else sys.maxsize
                )
            )

            if len(correct_lang_models) > 0:
                return correct_lang_models[0]
            else:
                return None

        @dataclass
        class Model(ABC):
            metadata_path: Path
            _type: str
            source_lang: str
            target_lang: str

            def description(self) -> str:
                return f"[{self._type}] {self.source_lang} -> {self.target_lang}"

            def shortlist_filename(self) -> str:
                return f"lex.50.50.{self.source_lang}{self.target_lang}.s2t.bin"

            def vocab_filename(self) -> str:
                return f"vocab.{self.source_lang}{self.target_lang}.spm"

            def model_filename(self) -> str:
                return f"model.{self.source_lang}{self.target_lang}.intgemm.alphas.bin"

        @dataclass
        class LocalModel(Model):
            # Obtained from:
            # https://github.com/mozilla/firefox-translations-models/blob/a06d4724eb95d7452f9251cf2cc4ca2706636d74/evals/translators/bergamot.config.yml
            CONFIG_TEMPLATE = """models:
  - {}
vocabs:
  - {}
  - {}
shortlist:
    - {}
    - false
beam-size: 1
normalize: 1.0
word-penalty: 0
max-length-break: 128
mini-batch-words: 1024
workspace: 128
max-length-factor: 2.0
skip-cost: true
cpu-threads: 0
quiet: false
quiet-translation: false
gemm-precision: int8shiftAlphaAll
alignment: soft
"""

            def parent_path(self) -> Path:
                return self.metadata_path.parent

            def model_path(self) -> Path:
                return self.parent_path().joinpath(self.model_filename())

            def vocab_path(self) -> Path:
                return self.parent_path().joinpath(self.vocab_filename())

            def shortlist_path(self) -> Path:
                return self.parent_path().joinpath(self.shortlist_filename())

            def config_path(self) -> Path:
                return self.parent_path().joinpath(
                    f"{self._type}.{self.source_lang}{self.target_lang}.config.yml"
                )

            def has_config(self) -> bool:
                return self.config_path().exists()

            def create_config(self):
                config_text = BergamotTranslator.LocalModel.CONFIG_TEMPLATE.format(
                    self.model_path().name,
                    self.vocab_path().name,
                    self.vocab_path().name,
                    self.shortlist_path().name,
                )

                with open(self.config_path(), "w") as fp:
                    fp.write(config_text)

        @dataclass
        class RemoteModel(Model):
            def model_filename(self) -> str:
                return super().model_filename() + ".gz"

            def vocab_filename(self) -> str:
                return super().vocab_filename() + ".gz"

            def shortlist_filename(self) -> str:
                return super().shortlist_filename() + ".gz"

            def metadata_url(self) -> str:
                return BergamotTranslator.URL_FIREFOX_REPO_PREFIX + str(
                    self.metadata_path
                )

            def base_url(self) -> str:
                return BergamotTranslator.URL_FIREFOX_REPO_PREFIX + str(
                    Path(self.metadata_path).parent
                )

            def shortlist_url(self) -> str:
                return self.base_url() + "/" + self.shortlist_filename()

            def vocab_url(self) -> str:
                return self.base_url() + "/" + self.vocab_filename()

            def model_url(self) -> str:
                return self.base_url() + "/" + self.model_filename()

            def download(
                self,
                models_path: Path,
            ) -> BergamotTranslator.LocalModel:
                relative_metadata_path = Path(*self.metadata_path.parts[1:])
                metadata_path = models_path.joinpath(relative_metadata_path)
                dest_parent = metadata_path.parent
                dest_parent.mkdir(parents=True, exist_ok=True)

                # Downloading the metadata for completeness but not sure if we need it
                urllib.request.urlretrieve(self.metadata_url(), metadata_path)

                model_path = dest_parent.joinpath(self.model_filename())
                urllib.request.urlretrieve(self.model_url(), model_path)

                vocab_path = dest_parent.joinpath(self.vocab_filename())
                urllib.request.urlretrieve(self.vocab_url(), vocab_path)

                shortlist_path = dest_parent.joinpath(self.shortlist_filename())
                urllib.request.urlretrieve(self.shortlist_url(), shortlist_path)

                # Decompress downloaded files
                decompress_file = BergamotTranslator.RemoteModel.decompress_file
                decompress_file(model_path)
                decompress_file(vocab_path)
                decompress_file(shortlist_path)

                # Remove archives
                model_path.unlink()
                vocab_path.unlink()
                shortlist_path.unlink()

                return BergamotTranslator.LocalModel(
                    metadata_path,
                    self._type,
                    self.source_lang,
                    self.target_lang,
                )

            @staticmethod
            def decompress_file(filepath: Path) -> Path:
                with open(filepath, "rb") as fp:
                    decompressed_data = gzip.decompress(fp.read())

                decompressed_filepath = filepath.with_suffix("")
                with open(decompressed_filepath, "wb") as fp:
                    fp.write(decompressed_data)

                return decompressed_filepath

        @staticmethod
        def fetch_models_from_repo() -> typing.Sequence[RemoteModel]:
            l = get_logger("BERGAMOT") # noqa: E741

            with urllib.request.urlopen(BergamotTranslator.URL_MODELS_BY_HASH) as fp:
                models_by_hash: dict = json.load(fp)

            models = []
            for path in models_by_hash.values():
                m = BergamotTranslator.RE_METADATA_PATH.match(path)
                if not m:
                    l.warning(
                        f"Found no match for firefox model path, this shouldn't happen, skipping path: {path}"
                    )
                    continue

                model_type = m.group(1)
                model_lang_from = m.group(2)
                model_lang_to = m.group(3)

                models.append(
                    BergamotTranslator.RemoteModel(
                        Path(path), model_type, model_lang_from, model_lang_to
                    )
                )

            return models

        @staticmethod
        def installed_models(models_path: Path) -> typing.Sequence[LocalModel]:
            models = []
            for type_dir in filter(lambda f: f.is_dir(), models_path.iterdir()):
                _type = type_dir.name
                for langs_dir in filter(lambda f: f.is_dir(), type_dir.iterdir()):
                    m = BergamotTranslator.RE_LANGS_DIR.match(langs_dir.name)
                    if not m:
                        continue
                    source_lang = m.group(1)
                    target_lang = m.group(2)
                    metadata_path = langs_dir.joinpath("metadata.json")
                    models.append(
                        BergamotTranslator.LocalModel(
                            metadata_path, _type, source_lang, target_lang
                        )
                    )

            return models


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
