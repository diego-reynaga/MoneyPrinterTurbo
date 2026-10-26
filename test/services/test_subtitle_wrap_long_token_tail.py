from pathlib import Path

from PIL import ImageFont

from app.services.video import wrap_text

FONT_PATH = str(
    Path(__file__).resolve().parents[2] / "resource/fonts/BeVietnamPro-Medium.ttf"
)


def _width(font, text):
    left, _, right, _ = font.getbbox(text)
    return right - left


def test_word_after_split_long_token_joins_its_last_fragment():
    # Un token más ancho que la línea se divide por caracteres. Su último fragmento no
    # llena la línea, así que la palabra corta siguiente debe colocarse a su lado en
    # lugar de empezar siempre otra línea de subtítulo.
    font = ImageFont.truetype(FONT_PATH, 60)
    text = "Supercalifragilisticexpialidocious is"

    wrapped, height = wrap_text(text, 400, font=FONT_PATH, fontsize=60)

    lines = wrapped.splitlines()
    assert lines[-1].endswith(" is")
    assert "".join(lines).replace(" ", "") == text.replace(" ", "")
    assert all(_width(font, line) <= 400 for line in lines)
    ascent, descent = font.getmetrics()
    assert height == len(lines) * (ascent + descent)


def test_word_after_split_long_token_still_wraps_when_it_does_not_fit():
    font = ImageFont.truetype(FONT_PATH, 60)
    text = "Supercalifragilisticexpialidocious extraordinarily"

    wrapped, _ = wrap_text(text, 400, font=FONT_PATH, fontsize=60)

    lines = wrapped.splitlines()
    assert all(_width(font, line) <= 400 for line in lines)
    assert "".join(lines).replace(" ", "") == text.replace(" ", "")
