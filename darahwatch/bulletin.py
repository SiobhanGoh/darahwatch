"""A short bulletin in English and Bahasa Malaysia, written from the numbers by fixed templates,
so it can only ever say what the forecast file says.
"""
from __future__ import annotations

import pandas as pd

MONTHS_MS = ["Januari", "Februari", "Mac", "April", "Mei", "Jun", "Julai", "Ogos", "September", "Oktober", "November", "Disember"]
DAYS_MS = ["Isnin", "Selasa", "Rabu", "Khamis", "Jumaat", "Sabtu", "Ahad"]


def _n(x) -> str:
    return f"{int(round(x)):,}"


def _en_date(d) -> str:
    d = pd.Timestamp(d)
    return f"{d.strftime('%a')} {d.day} {d.strftime('%b %Y')}"


def _ms_date(d) -> str:
    d = pd.Timestamp(d)
    return f"{DAYS_MS[d.dayofweek]} {d.day} {MONTHS_MS[d.month - 1]} {d.year}"


def _list(items, joiner) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {joiner} " + items[-1]


def write(issue: dict) -> dict:
    """`issue` holds: data_end, week1 (first, last, total), warnings (list of {state, blood_type,
    level, week}), holidays (list of {date, name, states})."""
    w1 = issue["week1"]
    likely = [w for w in issue["warnings"] if w["level"] == "Likely low" and w["week"] == 1]
    watch = [w for w in issue["warnings"] if w["level"] == "Watch" and w["week"] == 1]
    later = [w for w in issue["warnings"] if w["week"] == 2]
    label = lambda w: f"{w['state']} {w['blood_type']}"

    en = [f"Data to {_en_date(issue['data_end'])}. For {_en_date(w1['first'])} to {_en_date(w1['last'])}, "
          f"{issue.get('where_en', 'the places covered')} are expected to collect about {_n(w1['total'])} donations in total."]
    ms = [f"Data sehingga {_ms_date(issue['data_end'])}. Bagi tempoh {_ms_date(w1['first'])} hingga {_ms_date(w1['last'])}, "
          f"{issue.get('where_ms', 'kawasan yang diliputi')} dijangka mengumpul kira-kira {_n(w1['total'])} derma darah secara keseluruhan."]
    if likely:
        en.append(f"Likely low week: {_list(map(label, likely), 'and')}.")
        ms.append(f"Minggu yang mungkin rendah: {_list(map(label, likely), 'dan')}.")
    if watch:
        en.append(f"Worth watching: {_list(map(label, watch), 'and')}.")
        ms.append(f"Perlu diperhatikan: {_list(map(label, watch), 'dan')}.")
    if not likely and not watch:
        en.append("No state or blood group is expected to have a low week.")
        ms.append("Tiada negeri atau kumpulan darah yang dijangka mengalami minggu yang rendah.")
    if later:
        en.append(f"The week after: {len(later)} more warning{'s' if len(later) != 1 else ''}.")
        ms.append(f"Minggu berikutnya: {len(later)} amaran lagi.")
    for h in issue["holidays"][:3]:
        everywhere = h["states"] >= issue.get("regions", 13)
        where = "everywhere covered" if everywhere else f"{h['states']} state{'s' if h['states'] != 1 else ''}"
        di = "semua kawasan yang diliputi" if everywhere else f"{h['states']} negeri"
        en.append(f"Public holiday ahead: {h['name']} on {_en_date(h['date'])} ({where}).")
        ms.append(f"Cuti umum akan datang: {h['name']} pada {_ms_date(h['date'])} ({di}).")
    en.append("These are forecasts of donations collected, not of blood stock or hospital demand.")
    ms.append("Ini ialah ramalan jumlah derma yang dikumpul, bukan ramalan stok darah atau permintaan hospital.")
    return {"en": " ".join(en), "ms": " ".join(ms)}
