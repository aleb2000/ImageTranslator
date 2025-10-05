from __future__ import annotations
import traceback
from enum import Enum, auto
from pathlib import Path
import threading
from typing import cast
import wx
import imagetranslator
from imagetranslator import (
    ArgosTranslator,
    BergamotTranslator,
    CnOCR,
    EasyNMTTranslator,
    EasyOCR,
    ImageTranslator,
    PaddleOCR,
    PyOCR,
    translate_image_file,
)
from iso639 import languages


class OCREngine(str, Enum):
    AUTO = "Auto"
    CN = "CnOCR"
    PADDLE = "PaddleOCR"
    EASY = "EasyOCR"
    PY = "PyOCR"


OCR_ENGINES = [
    OCREngine.AUTO,
    OCREngine.CN,
    OCREngine.PADDLE,
    OCREngine.EASY,
    OCREngine.PY,
]


class TranslationModel(str, Enum):
    ARGOS = "Argos Translate"
    BERGAMOT = "Bergamot"
    OPUS = "Opus-MT"
    MBART50 = "mBART_50"
    M2M_100_418M = "M2M_100_418M"
    M2M_100_1_2B = "M2M_100_1.2B"


TRANSLATION_MODELS_CPU = [TranslationModel.ARGOS, TranslationModel.BERGAMOT]
TRANSLATION_MODELS_SUPPORT_CUDA = [
    TranslationModel.OPUS,
    TranslationModel.MBART50,
    TranslationModel.M2M_100_418M,
    TranslationModel.M2M_100_1_2B,
]
TRANSLATION_MODELS = TRANSLATION_MODELS_CPU + TRANSLATION_MODELS_SUPPORT_CUDA


class TextErasure(str, Enum):
    INPAINT = "Inpaint"
    INPAINT_LAMA = "Inpaint LaMa"
    BLUR = "Blur"
    NONE = "None"


TEXT_ERASURE_CHOICES = [
    TextErasure.INPAINT,
    TextErasure.INPAINT_LAMA,
    TextErasure.BLUR,
    TextErasure.NONE,
]


class LaMaDevice(str, Enum):
    AUTO = "Auto"
    CPU = "CPU"
    CUDA = "CUDA"


LAMA_DEVICE_CHOICES = [LaMaDevice.AUTO, LaMaDevice.CPU, LaMaDevice.CUDA]


class TranslationState(Enum):
    NOT_STARTED = auto()
    STARTED = auto()
    INIT_OCR = auto()
    INIT_TRANSLATOR = auto()
    TRANSLATING = auto()
    FINISHED = auto()
    ERROR = auto()


class TranslationProgress:
    state: TranslationState
    files_count: int
    error: Exception | None
    error_msg: str
    translated_files: list[Path]
    errored_files: list[tuple[Path, Exception]]

    def __init__(self) -> None:
        self.state = TranslationState.NOT_STARTED
        self.files_count = 0
        self.error = None
        self.error_msg = ""
        self.translated_files = []
        self.errored_files = []

    def set_error(self, e: Exception, msg: str):
        self.error = e
        self.error_msg = msg
        self.state = TranslationState.ERROR

    def processed_count(self):
        return len(self.translated_files) + len(self.errored_files)

    def add_errored_file(self, path: Path, error: Exception):
        self.errored_files.append((path, error))

    def add_translated_file(self, path: Path):
        self.translated_files.append(path)


def enums_to_value(enums: list):
    return [enum.value for enum in enums]


lang_codes = sorted(
    list(filter(lambda code: code, map(lambda lang: lang.part1, languages)))
)

ERROR_BG_COLOR_LIGHT = wx.Colour(0xE0, 0x1B, 0x24)
ERROR_BG_COLOR_DARK = wx.Colour(0xC0, 0x1C, 0x28)


def error_bg_color() -> wx.Colour:
    if wx.SystemSettings.GetAppearance().IsDark():
        return ERROR_BG_COLOR_DARK
    else:
        return ERROR_BG_COLOR_LIGHT


def validate_recursively(control):
    """Validate this control and its children recursively"""
    validator = control.GetValidator()
    # no validator -> valid
    isValid = validator.Validate(control) if validator else True
    for childControl in control.GetChildren():
        # only validate enabled controls
        if childControl.IsEnabled():
            isValid &= validate_recursively(childControl)
    return isValid


class ContainsTextValidator(wx.Validator):
    def Clone(self) -> wx.Object:
        return ContainsTextValidator()

    def Validate(self, parent: wx.Window) -> bool:
        textctrl = cast(wx.TextCtrl, parent)

        if textctrl.GetValue() == "":
            textctrl.SetBackgroundColour(error_bg_color())
            return False
        else:
            textctrl.SetBackgroundColour(
                wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW)
            )
            return True


class ImageTranslatorFrame(wx.Frame):
    panel: wx.Panel
    panel_sizer: wx.BoxSizer
    ocr_vertical_checkbox: wx.CheckBox
    ocr_vertical_rtl_checkbox: wx.CheckBox
    translation_choice: wx.Choice
    translation_device_choice: wx.Choice
    text_erasure_choice: wx.Choice
    lama_device_choice: wx.Choice
    source_language_combo: wx.ComboBox
    target_language_combo: wx.ComboBox
    languages_text: wx.StaticText
    files_list: wx.ListBox
    translate_button: wx.Button
    files_list_book: wx.Simplebook
    empty_list_text_panel: wx.Panel
    ocr_choice: wx.Choice
    font_picker: wx.FilePickerCtrl
    translation_progress: TranslationProgress
    output_dirpicker: wx.DirPickerCtrl

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        self.translation_progress = TranslationProgress()
        self.panel = wx.Panel(self)
        self.panel_sizer = wx.BoxSizer(orient=wx.VERTICAL)

        self.build_language_box()

        second_row_sizer = wx.BoxSizer()
        self.panel_sizer.Add(second_row_sizer, flag=wx.ALL | wx.EXPAND, border=6)
        self.build_ocr_box(second_row_sizer)
        second_row_sizer.AddSpacer(6)
        self.build_translation_box(second_row_sizer)

        third_row_sizer = wx.BoxSizer()
        self.panel_sizer.Add(third_row_sizer, flag=wx.ALL | wx.EXPAND, border=6)
        self.build_misc_box(third_row_sizer)
        third_row_sizer.AddSpacer(6)
        self.build_text_erasure_box(third_row_sizer)

        self.build_files_list()

        self.panel.SetSizer(self.panel_sizer)
        self.panel_sizer.SetSizeHints(self)

        self.Bind(wx.EVT_SYS_COLOUR_CHANGED, self.on_sys_colour_changed)

        self.update_translate_button()
        self.update_files_list()

    def build_language_box(self):
        language_box = wx.StaticBox(self.panel, label="Language")
        language_box_sizer = wx.StaticBoxSizer(language_box)

        combo_size = wx.Size(150, -1)

        # Source Language
        source_language_label = wx.StaticText(language_box, label="Source: ")
        source_language_label.SetToolTip(
            "The original language of the text in the image."
        )
        self.source_language_combo = wx.ComboBox(
            language_box,
            choices=lang_codes,
            size=combo_size,
            validator=ContainsTextValidator(),
        )
        source_language_sizer = wx.BoxSizer()
        source_language_sizer.Add(source_language_label, flag=wx.CENTER)
        source_language_sizer.Add(self.source_language_combo, flag=wx.CENTER)
        self.source_language_combo.Bind(
            wx.EVT_TEXT, self.on_language_entered, self.source_language_combo
        )
        self.source_language_combo.Bind(
            wx.EVT_KILL_FOCUS, self.on_language_kill_focus, self.source_language_combo
        )

        # Target Language
        target_language_label = wx.StaticText(language_box, label="Target: ")
        target_language_label.SetToolTip("The language to translate the text into.")
        self.target_language_combo = wx.ComboBox(
            language_box,
            choices=lang_codes,
            size=combo_size,
            validator=ContainsTextValidator(),
        )
        self.target_language_combo.SetValue("en")
        target_language_sizer = wx.BoxSizer()
        target_language_sizer.Add(target_language_label, flag=wx.CENTER)
        target_language_sizer.Add(self.target_language_combo, flag=wx.CENTER)
        self.target_language_combo.Bind(
            wx.EVT_TEXT, self.on_language_entered, self.target_language_combo
        )
        self.target_language_combo.Bind(
            wx.EVT_KILL_FOCUS, self.on_language_kill_focus, self.target_language_combo
        )

        # Languages text
        self.languages_text = wx.StaticText(
            language_box,
            style=wx.ST_NO_AUTORESIZE | wx.ALIGN_CENTRE_HORIZONTAL,
        )
        languages_text_sizer = wx.BoxSizer()
        languages_text_sizer.Add(self.languages_text, proportion=1, flag=wx.CENTER)
        self.set_languages_text()

        language_box_sizer.Add(source_language_sizer, flag=wx.CENTER | wx.ALL, border=6)
        language_box_sizer.Add(languages_text_sizer, proportion=1, flag=wx.CENTER)
        language_box_sizer.Add(target_language_sizer, flag=wx.CENTER | wx.ALL, border=6)
        self.panel_sizer.Add(language_box_sizer, flag=wx.ALL | wx.EXPAND, border=6)

    def build_ocr_box(self, container):
        ocr_box = wx.StaticBox(self.panel, label="OCR")
        ocr_box_sizer = wx.StaticBoxSizer(ocr_box, orient=wx.VERTICAL)

        ocr_internal_sizer = wx.BoxSizer(orient=wx.VERTICAL)

        # Engine selection
        ocr_choice_label = wx.StaticText(ocr_box, label="Engine: ")
        ocr_choice_label.SetToolTip("The Optical Character Recognition engine to use.")
        self.ocr_choice = wx.Choice(ocr_box, choices=enums_to_value(OCR_ENGINES))
        self.ocr_choice.Select(0)
        ocr_choice_sizer = wx.BoxSizer()
        ocr_choice_sizer.Add(ocr_choice_label, flag=wx.CENTER)
        ocr_choice_sizer.Add(self.ocr_choice, flag=wx.CENTER)

        # Vertical options
        self.ocr_vertical_checkbox = wx.CheckBox(
            ocr_box, label="Vertical Text Recognition"
        )
        self.ocr_vertical_checkbox.SetToolTip(
            "Allows to recognize vertical text (e.g. in Chinise scripts)."
        )
        self.ocr_vertical_rtl_checkbox = wx.CheckBox(
            ocr_box, label="Read Vertical Text Right-To-Left"
        )
        self.ocr_vertical_rtl_checkbox.SetToolTip(
            "When enabled, vertical text columns will be read and translated from right to left. Requires vertical text recognition."
        )
        self.Bind(
            wx.EVT_CHECKBOX,
            self.on_ocr_vertical_rtl_checked,
            self.ocr_vertical_rtl_checkbox,
        )

        ocr_internal_sizer.Add(ocr_choice_sizer)
        ocr_internal_sizer.AddSpacer(6)
        ocr_internal_sizer.Add(self.ocr_vertical_checkbox)
        ocr_internal_sizer.Add(self.ocr_vertical_rtl_checkbox)
        ocr_box_sizer.Add(ocr_internal_sizer, flag=wx.ALL | wx.CENTER, border=6)
        container.Add(ocr_box_sizer, proportion=1, flag=wx.EXPAND)

    def build_translation_box(self, container):
        translation_box = wx.StaticBox(self.panel, label="Translation")
        translation_box_sizer = wx.StaticBoxSizer(translation_box, orient=wx.VERTICAL)

        translation_internal_sizer = wx.BoxSizer(orient=wx.VERTICAL)

        # Translation model selection
        translation_choice_label = wx.StaticText(translation_box, label="Model: ")
        translation_choice_label.SetToolTip(
            "The translation model used to translate the text extracted from the images."
        )
        self.translation_choice = wx.Choice(
            translation_box, choices=enums_to_value(TRANSLATION_MODELS)
        )
        translation_choice_sizer = wx.BoxSizer()
        translation_choice_sizer.Add(translation_choice_label, flag=wx.CENTER)
        translation_choice_sizer.Add(self.translation_choice, flag=wx.CENTER)
        self.Bind(
            wx.EVT_CHOICE, self.on_translation_model_selected, self.translation_choice
        )

        # Device selection
        translation_device_choice_label = wx.StaticText(
            translation_box, label="Device: "
        )
        translation_device_choice_label.SetToolTip(
            "The hardware used to run the translation model. GPU acceleration is only supported by some models and requires an Nvidia GPU supporting CUDA."
        )
        self.translation_device_choice = wx.Choice(
            translation_box, choices=["Auto", "CPU", "CUDA"]
        )
        self.translation_device_choice.Select(0)
        translation_device_choice_sizer = wx.BoxSizer()
        translation_device_choice_sizer.Add(
            translation_device_choice_label, flag=wx.CENTER
        )
        translation_device_choice_sizer.Add(
            self.translation_device_choice, flag=wx.CENTER
        )

        self.translation_choice.Select(0)
        self.set_translation_device_enabled(0)

        translation_internal_sizer.Add(translation_choice_sizer)
        translation_internal_sizer.AddSpacer(6)
        translation_internal_sizer.Add(translation_device_choice_sizer)

        translation_box_sizer.Add(
            translation_internal_sizer, flag=wx.ALL | wx.CENTER, border=6
        )
        container.Add(translation_box_sizer, proportion=1, flag=wx.EXPAND)

    def build_misc_box(self, container):
        misc_box = wx.StaticBox(self.panel, label="Misc")

        # Output directory
        output_label = wx.StaticText(misc_box, label="Output directory: ")
        output_label.SetToolTip("The directory to save the translated images into.")
        self.output_dirpicker = wx.DirPickerCtrl(
            misc_box, message="Select the translation output directory"
        )
        output_sizer = wx.BoxSizer()
        output_sizer.Add(output_label, flag=wx.CENTER)
        output_sizer.Add(self.output_dirpicker, flag=wx.CENTER)

        # Font
        font_label = wx.StaticText(misc_box, label="Font: ")
        font_label.SetToolTip(
            "The font used to write the translated text on the image."
        )
        self.font_picker = wx.FilePickerCtrl(
            misc_box, path=".", wildcard="*.ttf", size=wx.Size(250, -1)
        )
        font_sizer = wx.BoxSizer()
        font_sizer.Add(font_label, flag=wx.CENTER)
        font_sizer.Add(self.font_picker, flag=wx.CENTER)

        misc_internal_sizer = wx.BoxSizer(orient=wx.VERTICAL)
        misc_internal_sizer.Add(output_sizer)
        misc_internal_sizer.AddSpacer(6)
        misc_internal_sizer.Add(font_sizer)

        misc_box_sizer = wx.StaticBoxSizer(misc_box, orient=wx.VERTICAL)
        misc_box_sizer.Add(misc_internal_sizer, flag=wx.ALL | wx.CENTER, border=6)
        container.Add(misc_box_sizer, proportion=1, flag=wx.EXPAND)

    def build_text_erasure_box(self, container):
        text_erasure_box = wx.StaticBox(self.panel, label="Text Erasure")

        # Text erasure
        text_erasure_choice_label = wx.StaticText(
            text_erasure_box, label="Text Erasure: "
        )
        text_erasure_choice_label.SetToolTip(
            "The technique used to remove the original text from the image."
        )
        self.text_erasure_choice = wx.Choice(
            text_erasure_box, choices=enums_to_value(TEXT_ERASURE_CHOICES)
        )
        self.text_erasure_choice.Select(0)
        text_erasure_choice_sizer = wx.BoxSizer()
        text_erasure_choice_sizer.Add(text_erasure_choice_label, flag=wx.CENTER)
        text_erasure_choice_sizer.Add(self.text_erasure_choice, flag=wx.CENTER)
        self.Bind(
            wx.EVT_CHOICE, self.on_text_erasure_selected, self.text_erasure_choice
        )

        # LaMa device
        lama_device_choice_label = wx.StaticText(
            text_erasure_box, label="LaMa Device: "
        )
        lama_device_choice_label.SetToolTip(
            "The hardware used to run the LaMa inpainting model. GPU acceleration requires an Nvidia GPU supporting CUDA."
        )
        self.lama_device_choice = wx.Choice(
            text_erasure_box, choices=enums_to_value(LAMA_DEVICE_CHOICES)
        )
        self.lama_device_choice.Select(0)
        lama_device_choice_sizer = wx.BoxSizer()
        lama_device_choice_sizer.Add(lama_device_choice_label, flag=wx.CENTER)
        lama_device_choice_sizer.Add(self.lama_device_choice, flag=wx.CENTER)

        text_erasure_internal_sizer = wx.BoxSizer(orient=wx.VERTICAL)
        text_erasure_internal_sizer.Add(text_erasure_choice_sizer)
        text_erasure_internal_sizer.AddSpacer(6)
        text_erasure_internal_sizer.Add(lama_device_choice_sizer)

        self.text_erasure_choice.Select(0)
        self.set_lama_device_enabled(0)

        text_erasure_box_sizer = wx.StaticBoxSizer(text_erasure_box, orient=wx.VERTICAL)
        text_erasure_box_sizer.Add(
            text_erasure_internal_sizer, flag=wx.ALL | wx.CENTER, border=6
        )
        container.Add(text_erasure_box_sizer, proportion=1, flag=wx.EXPAND)

    def build_files_list(self):
        self.files_list_book = wx.Simplebook(self.panel)

        # Files list
        self.files_list = wx.ListBox(
            self.files_list_book,
            size=wx.Size(-1, 150),
            style=wx.LB_MULTIPLE | wx.RAISED_BORDER,
        )
        self.files_list.SetToolTip("The image files to translate.")
        self.files_list.Bind(wx.EVT_KEY_DOWN, self.on_list_key_down, self.files_list)
        self.Bind(wx.EVT_CONTEXT_MENU, self.on_list_context_menu, self.files_list)

        self.empty_list_text_panel = wx.Panel(
            self.files_list_book, style=wx.SUNKEN_BORDER
        )
        empty_list_text_sizer = wx.BoxSizer()
        self.empty_list_text_panel.SetSizer(empty_list_text_sizer)
        self.empty_list_text_panel.SetBackgroundColour(
            wx.SystemSettings.GetColour(wx.SYS_COLOUR_LISTBOX)
        )

        empty_list_text = wx.StaticText(
            self.empty_list_text_panel,
            label="No input files selected",
            style=wx.ALIGN_CENTER_HORIZONTAL,
        )
        empty_list_text.SetFont(empty_list_text.GetFont().Scale(1.5))

        empty_list_text_sizer.Add(empty_list_text, proportion=1, flag=wx.CENTER)

        self.files_list_book.AddPage(self.empty_list_text_panel, "list-empty")
        self.files_list_book.AddPage(self.files_list, "list")

        # Add button
        add_button = wx.Button(self.panel, wx.ID_ADD)
        add_button.SetToolTip("Add input image file.")
        self.Bind(wx.EVT_BUTTON, self.on_add_button, add_button)

        # Translate button
        self.translate_button = wx.Button(self.panel, label="Translate")
        self.translate_button.SetToolTip("Translate the selected images.")
        self.Bind(wx.EVT_BUTTON, self.on_translate_button, self.translate_button)

        buttons_sizer = wx.BoxSizer()
        buttons_sizer.Add(add_button)
        buttons_sizer.AddStretchSpacer()
        buttons_sizer.Add(self.translate_button)

        files_sizer = wx.BoxSizer(wx.VERTICAL)
        files_sizer.Add(self.files_list_book, proportion=1, flag=wx.EXPAND)
        files_sizer.AddSpacer(6)
        files_sizer.Add(buttons_sizer, flag=wx.EXPAND)
        self.panel_sizer.Add(
            files_sizer, proportion=1, flag=wx.EXPAND | wx.ALL, border=6
        )

    def on_ocr_vertical_rtl_checked(self, event: wx.CommandEvent):
        if event.IsChecked():
            self.ocr_vertical_checkbox.SetValue(True)

    def on_translation_model_selected(self, event: wx.CommandEvent):
        self.set_translation_device_enabled(event.GetSelection())

    def set_translation_device_enabled(self, selected_model_id: int):
        model = self.translation_choice.GetString(selected_model_id)
        self.translation_device_choice.Enable(model in TRANSLATION_MODELS_SUPPORT_CUDA)

    def on_text_erasure_selected(self, event: wx.CommandEvent):
        self.set_lama_device_enabled(event.GetSelection())

    def set_lama_device_enabled(self, selected_model_id: int):
        text_erasure = self.text_erasure_choice.GetString(selected_model_id)
        self.lama_device_choice.Enable(text_erasure == "Inpaint LaMa")

    def set_languages_text(self):
        source_lang_code = self.source_language_combo.GetValue()
        target_lang_code = self.target_language_combo.GetValue()

        try:
            source_lang = languages.get(part1=source_lang_code)
        except KeyError:
            source_lang = None

        try:
            target_lang = languages.get(part1=target_lang_code)
        except KeyError:
            target_lang = None

        source_lang_name = source_lang.name if source_lang else f"[{source_lang_code}]"
        target_lang_name = target_lang.name if target_lang else f"[{target_lang_code}]"

        self.languages_text.SetLabel(f"{source_lang_name} -> {target_lang_name}")

    def on_language_entered(self, _):
        self.set_languages_text()
        self.validate_recursively()
        self.update_translate_button()

    def update_translate_button(self):
        languages_selected = (
            self.source_language_combo.GetValue() != ""
            and self.target_language_combo.GetValue() != ""
        )
        has_input_files = self.files_list.GetCount() > 0
        self.translate_button.Enable(languages_selected and has_input_files)

    def on_add_button(self, _: wx.CommandEvent):
        with wx.FileDialog(
            self,
            "Choose image files",
            style=wx.FD_MULTIPLE,
            wildcard="Image File (*.bmp;*.gif;*.jpg;*.jpeg;*.png;*.webp;*.blp;*.dds;*.dib;*.eps;*.icns;*.ico;*.im;*.jfif;*.j2k;*.j2p;*.jpx;*.msp;*.pcx;*.pbm;*.pgm;*.ppm;*.pnm;*.sgi;*.spi;*.tga;*.tiff;*.xbm)|*.bmp;*.gif;*.jpg;*.jpeg;*.png;*.webp;*.blp;*.dds;*.dib;*.eps;*.icns;*.ico;*.im;*.jfif;*.j2k;*.j2p;*.jpx;*.msp;*.pcx;*.pbm;*.pgm;*.ppm;*.pnm;*.sgi;*.spi;*.tga;*.tiff;*.xbm",
        ) as fd:
            fd: wx.FileDialog = fd
            if fd.ShowModal() == wx.ID_CANCEL:
                return

            pathnames = fd.GetPaths()
            self.files_list.AppendItems(pathnames)
            self.update_translate_button()
            self.update_files_list()

    def list_delete_selected(self):
        selections = self.files_list.GetSelections()

        # Since we can only remove elements one at a time
        # we need to keep track of the previously deleted element
        # hence we subtract the index
        for i, selection in enumerate(selections):
            self.files_list.Delete(selection - i)

        self.update_translate_button()
        self.update_files_list()

    def list_select_all(self):
        for n in range(0, self.files_list.GetCount()):
            self.files_list.Select(n)

    def on_list_key_down(self, event: wx.KeyEvent):
        if event.GetKeyCode() == wx.WXK_DELETE:
            self.list_delete_selected()

        if chr(event.GetKeyCode()) == "A":
            self.list_select_all()

    def on_list_context_menu(self, _: wx.ContextMenuEvent):
        if not hasattr(self, "list_popup_id_delete"):
            self.list_popup_id_delete = wx.ID_DELETE
            self.list_popup_id_selectall = wx.ID_SELECTALL

            self.Bind(
                wx.EVT_MENU,
                self.on_list_context_menu_delete,
                id=self.list_popup_id_delete,
            )
            self.Bind(
                wx.EVT_MENU,
                self.on_list_context_menu_selectall,
                id=self.list_popup_id_selectall,
            )

        selected = self.files_list.GetSelections()
        has_selections = len(selected) > 0
        has_items = self.files_list.GetCount() > 0

        menu = wx.Menu()
        delete = wx.MenuItem(menu, self.list_popup_id_delete)
        select_all = wx.MenuItem(menu, self.list_popup_id_selectall)
        menu.Append(delete)
        menu.Append(select_all)

        delete.Enable(has_selections)
        select_all.Enable(has_items)
        self.PopupMenu(menu)
        menu.Destroy()

    def on_list_context_menu_delete(self, _):
        self.list_delete_selected()

    def on_list_context_menu_selectall(self, _):
        self.list_select_all()

    def on_language_kill_focus(self, _: wx.FocusEvent):
        self.validate_recursively()

    def validate_recursively(self):
        validate_recursively(self)

    def update_files_list(self):
        file_count = self.files_list.GetCount()
        if file_count > 0:
            self.files_list_book.ChangeSelection(1)
        else:
            self.files_list_book.ChangeSelection(0)

    def on_sys_colour_changed(self, _: wx.SysColourChangedEvent):
        self.empty_list_text_panel.SetBackgroundColour(
            wx.SystemSettings.GetColour(wx.SYS_COLOUR_LISTBOX)
        )

        # I could not find a way to properly update the border color to the correct one
        # after changing light/dark mode, hence here we just disable it entirely
        self.empty_list_text_panel.SetWindowStyleFlag(0)
        self.validate_recursively()

    def on_translate_button(self, _):
        translation_thread = threading.Thread(target=self.do_translation)
        progress_updater_thread = threading.Thread(target=self.progress_updater)

        self.translation_progress = TranslationProgress()
        self.progress_dialog = wx.ProgressDialog(
            "Translating",
            "Translating Images",
            maximum=self.files_list.GetCount(),
            parent=self,
            style=wx.PD_CAN_ABORT,
        )
        self.progress_dialog.Show()
        translation_thread.start()
        progress_updater_thread.start()

    def progress_updater(self, sleep=250):
        def show_error_dialog():
            error_dialog = wx.MessageDialog(
                self.progress_dialog,
                f"{self.translation_progress.error_msg} {self.translation_progress.error}",
                "Error translating images",
                style=wx.ICON_ERROR | wx.OK,
            )
            error_dialog.ShowModal()

        def destroy_progress_dialog():
            self.progress_dialog.Destroy()

        def pulse(msg: str) -> tuple[bool, bool]:
            return self.progress_dialog.Pulse(msg)

        def update(value: int, msg: str) -> tuple[bool, bool]:
            return self.progress_dialog.Update(value, msg)

        while (
            self.translation_progress.state != TranslationState.FINISHED
            and self.translation_progress.state != TranslationState.ERROR
        ):
            if self.translation_progress.state == TranslationState.STARTED:
                wx.CallAfter(pulse, "Starting translation.")
            elif self.translation_progress.state == TranslationState.INIT_OCR:
                wx.CallAfter(pulse, "Initializing OCR engine.")
            elif self.translation_progress.state == TranslationState.INIT_TRANSLATOR:
                wx.CallAfter(pulse, "Initializing translator.")
            elif self.translation_progress.state == TranslationState.TRANSLATING:
                processed_count = self.translation_progress.processed_count()
                files_count = self.translation_progress.files_count
                msg = f"Translating file {processed_count + 1} of {files_count}."
                error_count = len(self.translation_progress.errored_files)
                if error_count > 0:
                    msg += f" Translations failed: {error_count}."
                wx.CallAfter(update, processed_count, msg)

            wx.MilliSleep(sleep)
            wx.Yield()


        if self.translation_progress.state == TranslationState.ERROR:
            wx.CallAfter(show_error_dialog)
            wx.CallAfter(destroy_progress_dialog)
        elif self.translation_progress.state == TranslationState.FINISHED:
            msg = "Finished"
            error_count = len(self.translation_progress.errored_files)
            if error_count > 0:
                msg += f" with {error_count} error"
                if error_count > 1:
                    msg += "s"
                msg += "."
            wx.CallAfter(
                update, self.translation_progress.processed_count(), msg
            )
        else:
            raise ValueError(
                f"How did we end up outside the loop in state {self.translation_progress.state}?"
            )

        self.translation_progress.state = TranslationState.NOT_STARTED
        print("End")

    def do_translation(self):
        print("Starting translation")
        self.translation_progress.state = TranslationState.STARTED

        source_lang = self.source_language_combo.GetValue()
        target_lang = self.target_language_combo.GetValue()

        self.translation_progress.state = TranslationState.INIT_OCR

        vertical = self.ocr_vertical_checkbox.GetValue()
        vertical_rtl = self.ocr_vertical_rtl_checkbox.GetValue()
        ocr_engine = self.ocr_choice.GetStringSelection()

        if ocr_engine == OCREngine.AUTO:
            if source_lang == "zh":
                ocr_engine = OCREngine.CN
            else:
                ocr_engine = OCREngine.PADDLE

        print("Initializing OCR")
        try:
            if ocr_engine == OCREngine.CN:
                ocr = CnOCR(vertical)
            elif ocr_engine == OCREngine.PADDLE:
                ocr = PaddleOCR(source_lang)
            elif ocr_engine == OCREngine.EASY:
                ocr = EasyOCR([source_lang])
            elif ocr_engine == OCREngine.PY:
                ocr = PyOCR(source_lang)
            else:
                raise ValueError("Invalid ocr engine name, how did this happen?")
        except Exception as e:
            traceback.print_exc()
            self.translation_progress.set_error(e, f"{ocr_engine} error:")
            return

        self.translation_progress.state = TranslationState.INIT_TRANSLATOR

        translation_device = self.translation_device_choice.GetStringSelection()
        if translation_device == "CPU":
            easynmt_device = EasyNMTTranslator.Device.CPU
        elif translation_device == "CUDA":
            easynmt_device = EasyNMTTranslator.Device.CUDA
        else:
            easynmt_device = None

        translator_name = self.translation_choice.GetStringSelection()

        print("Initializing translator")
        try:
            if translator_name == TranslationModel.ARGOS:
                translator = ArgosTranslator(source_lang, target_lang)
            elif translator_name == TranslationModel.BERGAMOT:
                translator = BergamotTranslator(source_lang, target_lang)
            elif translator_name == TranslationModel.OPUS:
                translator = EasyNMTTranslator(
                    EasyNMTTranslator.Model.OPUS,
                    device=easynmt_device,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
            elif translator_name == TranslationModel.MBART50:
                translator = EasyNMTTranslator(
                    EasyNMTTranslator.Model.MBART_50,
                    device=easynmt_device,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
            elif translator_name == TranslationModel.M2M_100_418M:
                translator = EasyNMTTranslator(
                    EasyNMTTranslator.Model.M2M_100_418M,
                    device=easynmt_device,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
            elif translator_name == TranslationModel.M2M_100_1_2B:
                translator = EasyNMTTranslator(
                    EasyNMTTranslator.Model.M2M_100_1_2B,
                    device=easynmt_device,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
            else:
                raise ValueError("Invalid translator name, how did this happen?")
        except Exception as e:
            traceback.print_exc()
            self.translation_progress.set_error(e, f"{translator_name} error:")
            return

        text_erasure_name = self.text_erasure_choice.GetStringSelection()
        if text_erasure_name == TextErasure.INPAINT:
            text_erasure = imagetranslator.TextErasure.INPAINT
        elif text_erasure_name == TextErasure.INPAINT_LAMA:
            text_erasure = imagetranslator.TextErasure.INPAINT_LAMA
        elif text_erasure_name == TextErasure.BLUR:
            text_erasure = imagetranslator.TextErasure.BLUR
        elif text_erasure_name == TextErasure.NONE:
            text_erasure = imagetranslator.TextErasure.NONE
        else:
            raise ValueError(
                "Invalid text erasure technique name, how did this happen?"
            )

        lama_device_name = self.lama_device_choice.GetStringSelection()
        if lama_device_name == LaMaDevice.AUTO:
            lama_device = None
        elif lama_device_name == LaMaDevice.CPU:
            lama_device = "cpu"
        elif lama_device_name == LaMaDevice.CUDA:
            lama_device = "cuda"
        else:
            raise ValueError("Invalid LaMa device name, how did this happen?")

        font_path = self.font_picker.GetPath()
        if (
            not Path(font_path).is_file()
            or not Path(font_path).exists()
            or str(font_path) in ["", "."]
        ):
            font_path = None

        image_translator = ImageTranslator(
            ocr, translator, text_erasure, lama_device, font_path
        )

        file_paths = self.files_list.GetStrings()
        output_dir = self.output_dirpicker.GetPath()
        if output_dir == "":
            output_dir = None

        self.translation_progress.files_count = len(file_paths)
        self.translation_progress.state = TranslationState.TRANSLATING

        for path in file_paths:
            path = Path(path)
            if output_dir:
                output_path = Path(output_dir).joinpath(
                    path.stem + ".translated" + path.suffix
                )
            else:
                output_path = path.parent.joinpath(
                    path.stem + ".translated" + path.suffix
                )

            try:
                translate_image_file(image_translator, path, output_path, vertical_rtl)
                self.translation_progress.add_translated_file(path)
            except Exception as e:
                print(f"Failed to translate file {path} with error {e}")
                self.translation_progress.add_errored_file(path, e)

        self.translation_progress.state = TranslationState.FINISHED


def main():
    app = wx.App()
    frame = ImageTranslatorFrame(None, title="Image Translator")
    frame.Show()
    app.MainLoop()


if __name__ == "__main__":
    exit(main())
