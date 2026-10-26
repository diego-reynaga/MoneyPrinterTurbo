from datetime import timedelta

import pytest
from edge_tts import SubMaker
from edge_tts.srt_composer import Subtitle

from app.services import subtitle, voice

WORDS = ["Welcome", "to", "the", "show", "Let", "us", "begin"]


def _edge_sub_maker():
    sub_maker = SubMaker()
    start = 0.0
    for index, word in enumerate(WORDS, 1):
        sub_maker.cues.append(
            Subtitle(
                index=index,
                start=timedelta(seconds=start),
                end=timedelta(seconds=start + 0.4),
                content=word,
            )
        )
        start += 0.5
    return sub_maker


@pytest.mark.parametrize("separator", ["* * *", "———", "• • •", "= = ="])
def test_symbol_only_script_lines_do_not_drop_edge_subtitles(tmp_path, separator):
    # Las líneas de cambio de escena formadas solo por símbolos nunca se pronuncian, así
    # que Edge no emite límites de palabra para ellas. No deben bloquear el resto de
    # subtítulos.
    subtitle_file = tmp_path / "subtitle.srt"

    voice.create_subtitle(
        _edge_sub_maker(),
        f"Welcome to the show.\n{separator}\nLet us begin.",
        str(subtitle_file),
    )

    items = subtitle.file_to_subtitles(str(subtitle_file))
    assert [item[2] for item in items] == ["Welcome to the show", "Let us begin"]
    assert items[1][1] == "00:00:02,000 --> 00:00:03,400"
