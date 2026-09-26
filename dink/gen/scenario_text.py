"""Scenario descriptions: one short text per gadget saying what it is and who uses it.

Dink encodes it once per device (like the names) and the mixer can attend to it. Nothing in
the stored data needs it; it's filled in when training / evaluation data is loaded:
  worlds       "<setting>. <about>" from the collector's record (data/llm/worlds.jsonl)
  smart home   a short house description built from the rooms and devices in the scenario
Training drops it at random (train.py --scenario-drop) so the model also works without one.
"""
import json
import random
from pathlib import Path

from ..data.scenario import SCENARIO_BYTES
from . import home as H
from .names import parse

ROOM = {r.code: r.words[0] for r in H.ROOMS}
DEVICE = {"lite": "lights", "temp": "thermostats", "fan": "ceiling fans", "blnd": "blinds",
          "lock": "door locks", "door": "a garage door"}
HOME = ["A family home with a {rooms}; it controls the {devices}.",
        "Smart home hub for the {rooms}: {devices}.",
        "Voice control for a house: {devices} in the {rooms}.",
        "An apartment's wall panel for the {rooms}. Devices: {devices}.",
        "Home assistant for a small house with a {rooms}; {devices}."]


def fit(text):
    b = text.encode()
    if len(b) <= SCENARIO_BYTES:
        return text
    return b[:SCENARIO_BYTES].decode(errors="ignore").rsplit(" ", 1)[0]


def listing(items):
    items = list(dict.fromkeys(items))
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def home_text(s, rng):
    rooms, devices = [], []
    for f in s.state + s.tools:
        where, kind = parse(f.name)
        if kind in DEVICE:
            devices.append(DEVICE[kind])
        if where in ROOM and kind != "occ":
            rooms.append(ROOM[where])
    if not rooms and not devices:
        return fit("A smart home voice assistant.")
    rng.shuffle(rooms)
    rng.shuffle(devices)
    t = rng.choice(HOME)
    return fit(t.format(rooms=listing(rooms) if rooms else "few rooms",
                        devices=listing(devices) if devices else "a few devices"))


class ScenarioText:
    """Fills Scenario.scenario for loaded data. World texts come from the collector's records."""

    def __init__(self, worlds="data/llm/worlds.jsonl"):
        self.worlds = {}
        p = Path(worlds)
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line)
                if "failed" not in r:
                    self.worlds[r["setting"]] = fit(f"{r['setting']}. {r['about']}")

    def fill(self, s, rng):
        if s.scenario:
            return s
        w = s.meta.get("uses", {}).get("world")
        s.scenario = self.worlds.get(w, "") if w else home_text(s, rng)
        return s
