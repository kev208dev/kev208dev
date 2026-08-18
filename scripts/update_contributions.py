from __future__ import annotations

import json
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

USERNAME = "kev208dev"
API_URL = f"https://github-contributions-api.jogruber.de/v4/{USERNAME}?y=last"
OUTPUT = Path("contrib-heatmap.svg")

COLORS = {
    0: "#161b22",
    1: "#0e4429",
    2: "#006d32",
    3: "#26a641",
    4: "#39d353",
}

CELL = 13
GAP = 3
STEP = CELL + GAP
LEFT = 34
TOP = 28
WIDTH = 891
HEIGHT = 181


def fetch_contributions() -> dict[str, dict[str, int]]:
    req = urllib.request.Request(
        API_URL,
        headers={"User-Agent": "kev208dev-profile-heatmap/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        payload = json.load(response)

    result: dict[str, dict[str, int]] = {}
    for item in payload.get("contributions", []):
        result[item["date"]] = {
            "count": int(item.get("count", 0)),
            "level": max(0, min(4, int(item.get("level", 0)))),
        }
    return result


def sunday_index(day: date) -> int:
    # Python: Monday=0 ... Sunday=6 -> Sunday=0 ... Saturday=6
    return (day.weekday() + 1) % 7


def rolling_window(today: date) -> tuple[date, date]:
    current_sunday = today - timedelta(days=sunday_index(today))
    start = current_sunday - timedelta(weeks=52)
    return start, today


def streaks(days: list[tuple[date, int]]) -> tuple[int, int]:
    best = 0
    running = 0
    for _, count in days:
        if count > 0:
            running += 1
            best = max(best, running)
        else:
            running = 0

    current = 0
    for _, count in reversed(days):
        if count <= 0:
            break
        current += 1
    return current, best


def month_labels(start: date, end: date) -> list[tuple[int, str]]:
    labels: list[tuple[int, str]] = []
    cursor = date(start.year, start.month, 1)
    if cursor < start:
        if start.month == 12:
            cursor = date(start.year + 1, 1, 1)
        else:
            cursor = date(start.year, start.month + 1, 1)

    while cursor <= end:
        week = (cursor - start).days // 7
        x = LEFT + week * STEP
        if x <= WIDTH - 35:
            labels.append((x, cursor.strftime("%b")))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return labels


def build_svg(data: dict[str, dict[str, int]], today: date) -> str:
    start, end = rolling_window(today)

    day_rows: list[tuple[date, int]] = []
    cursor = start
    while cursor <= end:
        item = data.get(cursor.isoformat(), {"count": 0, "level": 0})
        day_rows.append((cursor, item["count"]))
        cursor += timedelta(days=1)

    total = sum(count for _, count in day_rows)
    current_streak, best_streak = streaks(day_rows)

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">',
        "<style>",
        "  .cell { opacity: 0; animation: drop .45s ease-out forwards; }",
        "  @keyframes drop { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }",
        "  .fade { opacity: 0; animation: fade .8s ease-out forwards; }",
        "  @keyframes fade { to { opacity: 1; } }",
        "  text { fill: #8b949e; font-size: 11px; }",
        "</style>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="8" fill="#0d1117"/>',
    ]

    for x, label in month_labels(start, end):
        out.append(f'<text x="{x}" y="18" class="fade" style="animation-delay:.2s">{label}</text>')

    out.extend([
        '<text x="2" y="54" class="fade" style="animation-delay:.2s">Mon</text>',
        '<text x="2" y="86" class="fade" style="animation-delay:.2s">Wed</text>',
        '<text x="2" y="118" class="fade" style="animation-delay:.2s">Fri</text>',
    ])

    cursor = start
    while cursor <= end:
        week = (cursor - start).days // 7
        weekday = sunday_index(cursor)
        item = data.get(cursor.isoformat(), {"count": 0, "level": 0})
        x = LEFT + week * STEP
        y = TOP + weekday * STEP
        delay = (week + weekday) * 0.018
        color = COLORS[item["level"]]
        out.append(
            f'<rect class="cell" x="{x}" y="{y}" width="{CELL}" height="{CELL}" rx="3" fill="{color}" '
            f'style="animation-delay:{delay:.3f}s"><title>{cursor.isoformat()}: {item["count"]}</title></rect>'
        )
        cursor += timedelta(days=1)

    out.append(
        f'<text x="34" y="161" class="fade" style="animation-delay:1.4s">'
        f'{total} contributions in the last year&#160;&#160;·&#160;&#160;streak {current_streak}d (best {best_streak}d)</text>'
    )

    out.append('<text x="669" y="161" class="fade" style="animation-delay:1.4s">Less</text>')
    for i, level in enumerate(range(5)):
        x = 705 + i * STEP
        out.append(
            f'<rect class="fade" style="animation-delay:1.4s" x="{x}" y="150" width="{CELL}" height="{CELL}" rx="3" fill="{COLORS[level]}"/>'
        )
    out.append('<text x="789" y="161" class="fade" style="animation-delay:1.4s">More</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    today = datetime.now(ZoneInfo("Asia/Seoul")).date()
    data = fetch_contributions()
    OUTPUT.write_text(build_svg(data, today), encoding="utf-8")
    print(f"Updated {OUTPUT} through {today.isoformat()}")


if __name__ == "__main__":
    main()
