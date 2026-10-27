"""Tests for the photo screensaver."""

from __future__ import annotations

import asyncio
import functools as ft
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
from PIL import Image
from pydantic import ValidationError
from StreamDeck.Devices.StreamDeckOriginal import StreamDeckOriginal

from home_assistant_streamdeck_yaml import (
    Button,
    Config,
    Page,
    Screensaver,
    _on_press_callback,
    _photo_tiles,
    _screensaver_photos,
    reset_screensaver_timer,
    stop_screensaver,
    turn_off,
    update_key_image,
)

if TYPE_CHECKING:
    from pathlib import Path

KEY = 72
ROWS, COLS = 3, 5


@pytest.fixture
def deck() -> Mock:
    """A Stream Deck (15 keys, 5x3) that records the images it gets."""
    deck = Mock(spec=StreamDeckOriginal)
    deck.key_image_format.return_value = {
        "size": (KEY, KEY),
        "format": "JPEG",
        "flip": (False, False),
        "rotation": 0,
    }
    deck.key_count.return_value = ROWS * COLS
    deck.key_layout.return_value = (ROWS, COLS)
    deck.dial_count.return_value = 0
    deck.__enter__ = Mock(return_value=deck)
    deck.__exit__ = Mock(return_value=False)
    return deck


@pytest.fixture
def photos(tmp_path: Path) -> Path:
    """A folder with two photos (one in a subfolder) and a non-photo file."""
    folder = tmp_path / "photos"
    (folder / "sub").mkdir(parents=True)
    Image.new("RGB", (800, 480), "red").save(folder / "a.jpg")
    Image.new("RGB", (480, 800), "blue").save(folder / "sub" / "b.png")
    (folder / "notes.txt").write_text("geen foto")
    return folder


def _config(photos: Path, **screensaver: Any) -> Config:
    home = Page(
        name="home",
        buttons=[
            Button(special_type="go-to-page", special_type_data="other"),
            Button(text="tekst"),
        ],
    )
    other = Page(
        name="other",
        buttons=[Button(special_type="go-to-page", special_type_data="home")],
    )
    settings = {"folder": str(photos), "after": 0.05, "interval": 0.05, **screensaver}
    return Config(pages=[home, other], screensaver=settings)


async def _wait_for(predicate: Any, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not predicate():
        if loop.time() > end:
            raise TimeoutError
        await asyncio.sleep(0.01)


def test_screensaver_config() -> None:
    """Defaults, validation and the `screensaver` option of Config."""
    s = Screensaver(folder="photos")
    assert (s.after, s.interval, s.shuffle, s.brightness) == (300, 15, True, None)
    assert Config(screensaver={"folder": "photos"}).screensaver == s
    assert Config().screensaver is None
    with pytest.raises(ValidationError):
        Screensaver(folder="photos", after=0)
    with pytest.raises(ValidationError):
        Screensaver(folder="photos", unknown=1)


@pytest.mark.usefixtures("photos")
def test_photos_relative_to_config_file(tmp_path: Path) -> None:
    """A relative folder is relative to the configuration file."""
    config_file = tmp_path / "streamdeck-config.yaml"
    config_file.write_text("screensaver:\n  folder: photos\npages:\n  - name: home\n")
    config = Config.load(config_file)
    assert [p.name for p in _screensaver_photos(config)] == ["a.jpg", "b.png"]


def test_photo_tiles_line_up_across_keys() -> None:
    """Each key shows its own part of the photo, skipping the gaps."""
    gap = 24
    grid = (COLS * KEY + (COLS - 1) * gap, ROWS * KEY + (ROWS - 1) * gap)
    photo = Image.new("RGB", grid)
    # Give each key area its own color; the gaps stay black
    for row in range(ROWS):
        for col in range(COLS):
            color = (row * 100, col * 50, 200)
            box = (col * (KEY + gap), row * (KEY + gap))
            photo.paste(Image.new("RGB", (KEY, KEY), color), box)
    tiles = _photo_tiles(photo, ROWS, COLS, (KEY, KEY), gap)
    assert len(tiles) == ROWS * COLS
    for index, tile in enumerate(tiles):
        row, col = divmod(index, COLS)
        assert tile.size == (KEY, KEY)
        assert tile.getpixel((KEY // 2, KEY // 2)) == (row * 100, col * 50, 200)


def test_photo_tiles_crop_any_aspect_ratio() -> None:
    """Landscape, portrait and square photos all give 72x72 tiles."""
    for size in [(4000, 3000), (1080, 1920), (100, 100)]:
        tiles = _photo_tiles(Image.new("RGB", size, "green"), ROWS, COLS, (KEY, KEY), 24)
        assert {t.size for t in tiles} == {(KEY, KEY)}


async def test_screensaver_starts_after_inactivity(deck: Mock, photos: Path) -> None:
    """After `after` seconds all keys show the photo, at the screensaver brightness."""
    config = _config(photos, brightness=20)
    reset_screensaver_timer(config, deck)
    await _wait_for(
        lambda: config._screensaver_active and deck.set_key_image.call_count >= ROWS * COLS,
    )
    keys = {c.args[0] for c in deck.set_key_image.call_args_list}
    assert keys == set(range(ROWS * COLS))
    assert all(isinstance(c.args[1], bytes) for c in deck.set_key_image.call_args_list)
    deck.set_brightness.assert_called_with(20)
    stop_screensaver(config, deck)


async def test_key_press_only_wakes_up(deck: Mock, photos: Path) -> None:
    """The press that stops the screensaver does not trigger the button."""
    config = _config(photos)
    state: dict[str, Any] = {}
    press = ft.partial(_on_press_callback(Mock(), state, config), deck)
    reset_screensaver_timer(config, deck)
    await _wait_for(lambda: config._screensaver_active)

    # Key 0 is "go to page other", but the wake-up press must not do that
    await press(0, True)  # noqa: FBT003
    assert not config._screensaver_active
    await press(0, False)  # noqa: FBT003
    assert config.current_page().name == "home"
    deck.set_brightness.assert_called_with(config.brightness)

    # The next press works normally
    await press(0, True)  # noqa: FBT003
    await press(0, False)  # noqa: FBT003
    assert config.current_page().name == "other"
    stop_screensaver(config, deck)


async def test_state_updates_do_not_draw_over_photos(deck: Mock, photos: Path) -> None:
    """State changes do not redraw keys while the screensaver runs."""
    config = _config(photos)
    reset_screensaver_timer(config, deck)
    await _wait_for(lambda: config._screensaver_active)
    deck.set_key_image.reset_mock()
    update_key_image(deck, key=1, config=config, complete_state={})
    assert deck.set_key_image.call_count == 0
    stop_screensaver(config, deck)


async def test_turn_off_stops_screensaver(deck: Mock, photos: Path) -> None:
    """Turning the deck off also stops the screensaver."""
    config = _config(photos)
    reset_screensaver_timer(config, deck)
    await _wait_for(lambda: config._screensaver_active)
    turn_off(config, deck)
    assert not config._screensaver_active
    assert config._screensaver_task is None


async def test_no_photos_no_screensaver(deck: Mock, tmp_path: Path) -> None:
    """Without photos the keys stay as they are."""
    config = _config(tmp_path / "does-not-exist")
    reset_screensaver_timer(config, deck)
    await asyncio.sleep(0.2)
    assert not config._screensaver_active
    assert deck.set_key_image.call_count == 0
