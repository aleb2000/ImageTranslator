# Image Translator

A local image translation tool written in python. It supports several OCR and translation models. There is both a CLI program, as well as a [GUI](./gui) option

Requires Python 3.10

Suggestions and PRs are welcome!

# OCR Engines

The following OCR engines are supported.

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) as the default for most languages
- [CnOCR](https://github.com/breezedeus/cnocr) as the default for Chinese
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) as an alternative with many different languages
- [PyOCR](https://github.com/kyper999/pyocr) as a frontend for Tesseract

Most of these will automatically download the required models on demand for the selected language (if available),
with the exception being PyOCR which requires you to manually install the appropriate Tesseract language in advance.

# Translators

The following translators are supported.

- [Argos Translate](https://github.com/argosopentech/argos-translate) as the default translator, as it is very lightweight while having quite a good translation performance.
- [Bergamot Translator](https://github.com/browsermt/bergamot-translator/), using the [Firefox Translations](https://github.com/mozilla/firefox-translations-models) models (except on the Windows platform, where it is unsupported).
- [EasyNMT](https://github.com/UKPLab/EasyNMT), a wrapper for several translation models:
    - Opus-MT
    - mBART_50
    - M2M_100 (with variants of 418 million parameters and 1.2 billion parameters)

The options offered by EasyNMT tend to be heavier and require more storage space (the M2M_100 model weighing at 5GB),
although that does not necessarily indicate to better translation performance.
It is best to try different translation models and see what works best for the given text and languages.

# Text Erasure

The following options are available to erase the original text from the image.

- Inpainting using [scikit-image](https://github.com/scikit-image/scikit-image) biharmonic inpainting functionality. This is cheap but only works well in simple cases, where the text to replace covers a small area of the image and/or the background is quite predictable. It is the default option.
- Inpainting using the [LaMa](https://github.com/advimman/lama) model. This produces significantly better results but it is comparatively VERY heavy to run.
- Covering the text using blur. This produces a very evident mark on the image but it is trivially cheap and somtimes might be the best choice when other inpainting options would otherwise ruin the original image.

Text erasure can be disabled if necessary.

# Model download

All models (except the Tesseract ones) will be downloaded automatically when required.

NOTE: The downloads are handled by each OCR engine and translator on their own as such their download location will vary, check each model's own documentation if you want to learn where the model has been downloaded. The only exception is Bergamot, that did not include automatic model download functionality and as such I have implemented it myself: the models are downloaded at `~/.local/share/imagetranslator/bergamot` (on Linux).

Neither CLI nor GUI applications currently implement a way to manage downloaded models. To remove them you will have to manually delete the files yourself.

# Known issues

- (GUI) the "cancel" option during a translation is currently not implemented, don't use it

# LICENSE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
