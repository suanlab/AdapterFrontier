#!/usr/bin/env python3
"""Regenerate paper/custom.bib from authoritative sources.

Why this exists
---------------
The August 2026 ARR submission was desk-rejected for its references: several
did not report the full author list, and none carried a DOI or ACL Anthology
link. The bibliography had been a hand-written `thebibliography` block, so
every author list was whatever someone had typed, and "et al." had crept into
27 of the 53 entries. One entry (Nixon et al., 2019) was simply missing two
authors.

Rather than hand-fix it, the bibliography is now *derived*. Nothing below is
typed twice:

  * ACL-venue entries are the ACL Anthology's own BibTeX, fetched verbatim.
    Those already carry the full author list, the anthology DOI and the pages.
  * Author lists for everything else come from the arXiv API (the names as the
    authors themselves submitted them) or from Crossref (as the publisher
    registered them). Both beat a bibliographic aggregator: OpenAlex, which
    supplied the DOIs, renders "Jeff Dean" as "Jay B. Dean", abbreviates
    "Zhuohan Li" to "Z. Li", and inverts some names into surname-first order.
  * DOIs were taken from Crossref/OpenAlex and each one was checked to resolve.
    Where a work has both a venue DOI and an arXiv DOI, the arXiv DOI wins: the
    NeurIPS 10.52202/* DOIs resolve to a paywalled print-proceedings vendor,
    while the arXiv DOI reaches the full text.
  * Two works have no usable DOI. Holm (1979) has none in Crossref and the
    JSTOR DOI 10.2307/4615733 returns 404, so it carries the stable JSTOR URL
    instead; lm-evaluation-harness carries its maintainers' own citation block.

Usage
-----
    python3 scripts/build_bibliography.py              # rewrite paper/custom.bib
    python3 scripts/build_bibliography.py --check      # fail if it would change

`scripts/check_bibliography.py` then enforces the ACL rules on the result.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "custom.bib"
MAILTO = "anon@example.org"   # anonymised; any contact address works

# --- what to fetch ----------------------------------------------------------

ANTHOLOGY = {
    "adamix": "2022.emnlp-main.388",
    "wang2018glue": "W18-5446",
    "zellers2019hellaswag": "P19-1472",
    "pfeiffer2020adapterhub": "2020.emnlp-demos.7",
    "pfeiffer2021adapterfusion": "2021.eacl-main.39",
    "dodge2019show": "D19-1224",
    "reimers2017reporting": "D17-1035",
    "bowman2021will": "2021.naacl-main.385",
    "chen2022revisiting": "2022.emnlp-main.168",
}

ICLR = "International Conference on Learning Representations (ICLR)"
NEURIPS = "Advances in Neural Information Processing Systems (NeurIPS)"
MLSYS = "Proceedings of Machine Learning and Systems (MLSys)"


def icml(n: str) -> str:
    return f"Proceedings of the {n} International Conference on Machine Learning (ICML)"


# key -> (arXiv id, entry type, venue, pages or None)
ARXIV: dict[str, tuple[str, str, str, str | None]] = {
    "hu2021lora":               ("2106.09685", "inproceedings", ICLR, None),
    "wortsman2022model":        ("2203.05482", "inproceedings", icml("39th"), None),
    "yadav2023ties":            ("2306.01708", "inproceedings", NEURIPS, None),
    "yu2024dare":               ("2311.03099", "inproceedings", icml("41st"), None),
    "dora":                     ("2402.09353", "inproceedings", icml("41st"), None),
    "pissa":                    ("2404.02948", "inproceedings", NEURIPS, None),
    "adalora":                  ("2303.10512", "inproceedings", ICLR, None),
    "vera":                     ("2310.11454", "inproceedings", ICLR, None),
    "mole":                     ("2404.13628", "inproceedings", ICLR, None),
    "molora":                   ("2211.03831", "inproceedings", NEURIPS, None),
    "hydralora":                ("2404.19245", "inproceedings", NEURIPS, None),
    "multilora":                ("2311.11501", "article", "arXiv preprint arXiv:2311.11501", None),
    "nixon2019":                ("1904.01685", "inproceedings",
                                 "Proceedings of the IEEE/CVF Conference on Computer Vision "
                                 "and Pattern Recognition (CVPR) Workshops", "38--41"),
    "minderer2021":             ("2106.07998", "inproceedings", NEURIPS, None),
    "huang2023lorahub":         ("2307.13269", "inproceedings",
                                 "Proceedings of the First Conference on Language Modeling (COLM)", None),
    "lakshminarayanan2017simple": ("1612.01474", "inproceedings", NEURIPS, None),
    "guo2017calibration":       ("1706.04599", "inproceedings", icml("34th"), "1321--1330"),
    "sheng2024slora":           ("2311.03285", "inproceedings", MLSYS, None),
    "cobbe2021gsm8k":           ("2110.14168", "article", "arXiv preprint arXiv:2110.14168", None),
    "liu2019roberta":           ("1907.11692", "article", "arXiv preprint arXiv:1907.11692", None),
    "houlsby2019parameter":     ("1902.00751", "inproceedings", icml("36th"), None),
    "hinton2015distilling":     ("1503.02531", "article", "arXiv preprint arXiv:1503.02531", None),
    "karandikar2021soft":       ("2108.00106", "inproceedings", NEURIPS, None),
    "lucic2018gans":            ("1711.10337", "inproceedings", NEURIPS, None),
    "dehghani2022efficiency":   ("2110.12894", "inproceedings", ICLR, None),
    "chen2024punica":           ("2310.18547", "inproceedings", MLSYS, None),
    "mirzadeh2024gsmsymbolic":  ("2410.05229", "article", "arXiv preprint arXiv:2410.05229", None),
    "zhang2024gsm1k":           ("2405.00332", "inproceedings", NEURIPS, None),
    "abe2022deep":              ("2202.06985", "inproceedings", NEURIPS, None),
    "theisen2023ensembles":     ("2305.12313", "inproceedings", NEURIPS, None),
    "kondratyuk2020ensembling": ("2005.00570", "article", "arXiv preprint arXiv:2005.00570", None),
    "lobacheva2020power":       ("2007.08483", "inproceedings", NEURIPS, "2375--2385"),
    # Venues confirmed on the official proceedings pages: NeurIPS 2019
    # (proceedings.neurips.cc) and ICLR 2020 (iclr.cc/virtual_2020).
    "ovadia2019trust":          ("1906.02530", "inproceedings", NEURIPS, None),
    "ashukha2020pitfalls":      ("2002.06470", "inproceedings", ICLR, None),
}

YEAR = {  # publication year of the *venue*, which is not always the arXiv year
    "hu2021lora": "2022", "wortsman2022model": "2022", "yadav2023ties": "2023",
    "yu2024dare": "2024", "dora": "2024", "pissa": "2024", "adalora": "2023",
    "vera": "2024", "mole": "2024", "molora": "2023", "hydralora": "2024",
    "multilora": "2023", "nixon2019": "2019", "minderer2021": "2021",
    "huang2023lorahub": "2024", "lakshminarayanan2017simple": "2017",
    "guo2017calibration": "2017", "sheng2024slora": "2024",
    "cobbe2021gsm8k": "2021", "liu2019roberta": "2019",
    "houlsby2019parameter": "2019", "hinton2015distilling": "2015",
    "karandikar2021soft": "2021", "lucic2018gans": "2018",
    "dehghani2022efficiency": "2022", "chen2024punica": "2024",
    "mirzadeh2024gsmsymbolic": "2024", "zhang2024gsm1k": "2024",
    "abe2022deep": "2022", "theisen2023ensembles": "2023",
    "kondratyuk2020ensembling": "2020", "lobacheva2020power": "2020",
    "ovadia2019trust": "2019", "ashukha2020pitfalls": "2020",
}

# key -> (DOI, entry type, venue, fallback pages)
CROSSREF: dict[str, tuple[str, str, str, str | None]] = {
    "naeini2015": ("10.1609/aaai.v29i1.9602", "inproceedings",
                   "Proceedings of the AAAI Conference on Artificial Intelligence", "2901--2907"),
    "diciccio1996": ("10.1214/ss/1032280214", "article", "Statistical Science", "189--228"),
    "bh1995": ("10.1111/j.2517-6161.1995.tb02031.x", "article",
               "Journal of the Royal Statistical Society: Series B (Methodological)", None),
    "munafo2017manifesto": ("10.1038/s41562-016-0021", "article", "Nature Human Behaviour", "0021"),
    "loken2017measurement": ("10.1126/science.aal3618", "article", "Science", None),
    "hofman2021integrating": ("10.1038/s41586-021-03659-0", "article", "Nature", None),
    "musgrave2020metric": ("10.1007/978-3-030-58595-2_41", "inproceedings",
                           "Proceedings of the European Conference on Computer Vision (ECCV)", None),
    "dacrema2019progress": ("10.1145/3298689.3347058", "inproceedings",
                            "Proceedings of the 13th ACM Conference on Recommender Systems (RecSys)", None),
    "efron1979bootstrap": ("10.1214/aos/1176344552", "article", "The Annals of Statistics", "1--26"),
    "kwon2023efficient": ("10.1145/3600006.3613165", "inproceedings",
                          "Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)", None),
    "mackay1992evidence": ("10.1162/neco.1992.4.5.720", "article", "Neural Computation", None),
}

# Crossref registers Efron (1979) as "B. Efron"; the same publisher spells him
# out on DiCiccio & Efron (1996), and ACL requires full names.
NAME_OVERRIDE = {"efron1979bootstrap": ["Bradley Efron"]}

# Surnames that BibTeX's "last token is the family name" rule, and its
# von-particle rule, both get wrong. Braced so BibTeX keeps them whole.
COMPOUND = {
    "Nathalie Percie du Sert": ("{Percie du Sert}", "Nathalie"),
    "Nicolas Le Roux": ("{Le Roux}", "Nicolas"),
    "Quentin de Laroussilhe": ("{de Laroussilhe}", "Quentin"),
    "Maurizio Ferrari Dacrema": ("{Ferrari Dacrema}", "Maurizio"),
}

EVAL_HARNESS_CITATION = (
    "https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/CITATION.bib")

ACCENTS = {
    "à": r"\`a", "á": r"\'a", "â": r"\^a", "ã": r"\~a", "ä": r'\"a',
    "è": r"\`e", "é": r"\'e", "ê": r"\^e", "ë": r'\"e',
    "ì": r"\`i", "í": r"\'i", "î": r"\^i", "ï": r'\"i',
    "ò": r"\`o", "ó": r"\'o", "ô": r"\^o", "õ": r"\~o", "ö": r'\"o',
    "ù": r"\`u", "ú": r"\'u", "û": r"\^u", "ü": r'\"u',
    "ñ": r"\~n", "ç": r"\c c", "ł": r"\l", "š": r"\v s", "ž": r"\v z",
    "č": r"\v c", "ě": r"\v e", "ř": r"\v r", "ć": r"\'c", "ø": r"\o",
}

# --- fetching ---------------------------------------------------------------
# curl, not urllib: arXiv's API answers urllib with HTTP 406 regardless of the
# User-Agent it is given.


def fetch(url: str) -> bytes:
    p = subprocess.run(["curl", "-sS", "--fail", "--max-time", "60", url],
                       capture_output=True)
    if p.returncode:
        raise RuntimeError(f"fetch failed ({p.returncode}): {url}\n"
                           f"{p.stderr.decode(errors='replace')[:300]}")
    return p.stdout


def fetch_anthology(aid: str) -> str:
    return fetch(f"https://aclanthology.org/{aid}.bib").decode().strip()


def fetch_arxiv_abs(aid: str) -> dict:
    """The same record from the abstract page's citation_* meta tags.

    Fallback for when the export API refuses this host (it has answered HTTP
    406 to every query form for hours at a time). On 2026-09-27 all 33 entries
    then in the bibliography came back identical, title and author list, from
    both routes, so which one ran does not change the output.
    """
    import html
    page = fetch(f"https://arxiv.org/abs/{aid}").decode()

    def meta(name: str) -> list[str]:
        return [html.unescape(v) for v in
                re.findall(r'<meta name="' + name + r'" content="([^"]*)"', page)]

    authors = []
    for a in meta("citation_author"):          # "Family, Given"
        family, _, given = a.partition(", ")
        authors.append(f"{given} {family}".strip() if given else family)
    titles = meta("citation_title")
    if not titles or not authors:
        raise RuntimeError(f"arXiv abs page for {aid} carried no citation meta")
    return {"title": " ".join(titles[0].split()), "authors": authors}


def fetch_arxiv(ids: list[str]) -> dict[str, dict]:
    """Batched API first; the abstract pages if the API refuses us."""
    try:
        return fetch_arxiv_api(ids)
    except RuntimeError as e:
        print(f"arXiv API unavailable ({str(e).splitlines()[-1]}); "
              "reading the abstract pages instead", file=sys.stderr)
    out = {}
    for aid in ids:
        out[aid] = fetch_arxiv_abs(aid)
        time.sleep(1)
    return out


def fetch_arxiv_api(ids: list[str]) -> dict[str, dict]:
    """Batched; the API caps a reply at max_results entries."""
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out: dict[str, dict] = {}
    for i in range(0, len(ids), 8):
        chunk = ids[i:i + 8]
        root = ET.fromstring(fetch(
            "https://export.arxiv.org/api/query?id_list="
            + ",".join(chunk) + "&max_results=50"))
        for e in root.findall("a:entry", ns):
            aid = e.find("a:id", ns).text.rsplit("/", 1)[-1].split("v")[0]
            out[aid] = {
                "title": " ".join(e.find("a:title", ns).text.split()),
                "authors": [a.find("a:name", ns).text
                            for a in e.findall("a:author", ns)],
            }
        time.sleep(3)   # arXiv asks for one request every three seconds
    missing = set(ids) - set(out)
    if missing:
        raise RuntimeError(f"arXiv returned nothing for {sorted(missing)}")
    return out


def fetch_crossref(doi: str) -> dict:
    m = json.loads(fetch("https://api.crossref.org/works/"
                         + urllib.parse.quote(doi)
                         + f"?mailto={MAILTO}"))["message"]
    authors = []
    for a in m.get("author", []):
        given, family = a.get("given"), a.get("family")
        authors.append(f"{given} {family}".strip() if given
                       else (family or a.get("name", "")))
    return {
        "title": (m.get("title") or [""])[0],
        "authors": authors,
        "year": str((m.get("issued", {}).get("date-parts") or [[""]])[0][0]),
        "volume": m.get("volume"),
        "number": m.get("issue"),
        "pages": (m.get("page") or "").replace("-", "--") or None,
        "doi": m.get("DOI"),
    }


# --- formatting -------------------------------------------------------------


def tex(s: str) -> str:
    """UTF-8 -> LaTeX escapes, so the .bib stays pure ASCII and bibtex-safe."""
    out = []
    for ch in s:
        if ord(ch) < 128:
            out.append(ch)
            continue
        esc = ACCENTS.get(ch.lower())
        if esc:
            if ch.isupper():
                esc = esc[:-1] + esc[-1].upper()
            out.append("{" + esc + "}")
        else:
            stripped = "".join(c for c in unicodedata.normalize("NFKD", ch)
                               if not unicodedata.combining(c))
            out.append(stripped if stripped.isascii() else "?")
    return "".join(out)


def bib_authors(names: list[str]) -> str:
    parts = []
    for n in names:
        n = n.strip()
        if n in COMPOUND:
            family, given = COMPOUND[n]
            parts.append(f"{family}, {tex(given)}")
            continue
        toks = tex(n).split()
        # A bare initial ("D Sculley", as arXiv's meta tags give it) gets its
        # full stop, so both arXiv routes and the proceedings agree: "D. Sculley".
        toks = [t + "." if len(t) == 1 and t.isupper() else t for t in toks]
        parts.append(toks[0] if len(toks) == 1
                     else f"{toks[-1]}, {' '.join(toks[:-1])}")
    return " and\n            ".join(parts)


def emit(key: str, etype: str, fields: dict[str, str | None]) -> str:
    lines = [f"@{etype}{{{key},"]
    for name, value in fields.items():
        if value in (None, "", []):
            continue
        lines.append(f"    {name:<12} = {{{value}}},")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    return "\n".join(lines)


HEADER = """% Bibliography for AdapterFrontier.
%
% Every entry carries the complete author list and a DOI (or, where no DOI
% exists, a stable publisher URL), per the ACL formatting guidelines:
%   https://acl-org.github.io/ACLPUB/formatting.html
%
% Provenance: ACL-venue entries are the ACL Anthology's own BibTeX, verbatim.
% Author lists elsewhere come from the arXiv API (as the authors wrote them)
% or from Crossref (as the publisher registered them); DOIs come from
% Crossref/OpenAlex and were each checked to resolve. Generated by
% scripts/build_bibliography.py -- do not hand-edit.
"""


def build() -> str:
    chunks = [HEADER]

    for key, aid in ANTHOLOGY.items():
        raw = fetch_anthology(aid)
        raw = re.sub(r"^@(\w+)\{[^,]+,",
                     lambda m: f"@{m.group(1)}{{{key},", raw, count=1)
        raw = re.sub(r'\n\s*editor = "[^"]*",', "", raw)
        raw = re.sub(r"\n\s*editor = \{[^}]*\},", "", raw)
        chunks.append(raw + "\n")

    papers = fetch_arxiv([v[0] for v in ARXIV.values()])
    for key, (aid, etype, venue, pages) in ARXIV.items():
        rec = papers[aid]
        fields: dict[str, str | None] = {
            "title": "{" + tex(rec["title"]) + "}",
            "author": bib_authors(rec["authors"]),
        }
        fields["booktitle" if etype == "inproceedings" else "journal"] = venue
        fields["year"] = YEAR[key]
        fields["pages"] = pages
        fields["doi"] = f"10.48550/arXiv.{aid}"
        chunks.append(emit(key, etype, fields) + "\n")

    for key, (doi, etype, venue, pages) in CROSSREF.items():
        rec = fetch_crossref(doi)
        names = NAME_OVERRIDE.get(key) or rec["authors"]
        fields = {"title": "{" + tex(rec["title"]) + "}",
                  "author": bib_authors(names)}
        if etype == "inproceedings":
            fields["booktitle"] = venue
        else:
            fields["journal"] = venue
            fields["volume"] = rec["volume"]
            fields["number"] = rec["number"]
        fields["pages"] = rec["pages"] or pages
        fields["year"] = rec["year"]
        fields["doi"] = rec["doi"]
        chunks.append(emit(key, etype, fields) + "\n")
        time.sleep(0.4)

    chunks.append(emit("holm1979simple", "article", {
        "title": "{A Simple Sequentially Rejective Multiple Test Procedure}",
        "author": "Holm, Sture",
        "journal": "Scandinavian Journal of Statistics",
        "volume": "6", "number": "2", "pages": "65--70", "year": "1979",
        # Crossref has no DOI for this 1979 article, and JSTOR's
        # 10.2307/4615733 returns 404, so the stable JSTOR URL is the
        # best identifier available.
        "url": "https://www.jstor.org/stable/4615733"}) + "\n")

    harness = fetch(EVAL_HARNESS_CITATION).decode().strip()
    harness = harness.replace("@misc{eval-harness,", "@misc{gao2023lmeval,", 1)
    chunks.append(harness + "\n")

    return "\n".join(chunks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the generated file differs")
    args = ap.parse_args()

    text = build()
    n = len(re.findall(r"^@", text, re.M))
    expected = len(ANTHOLOGY) + len(ARXIV) + len(CROSSREF) + 2
    if n != expected:
        print(f"FAIL: generated {n} entries, expected {expected}")
        return 1

    if args.check:
        if not OUT.exists():
            print(f"FAIL: {OUT} does not exist")
            return 1
        if OUT.read_text() != text:
            print(f"FAIL: {OUT} is out of date; re-run without --check")
            return 1
        print(f"OK: {OUT} matches the sources ({n} entries)")
        return 0

    OUT.write_text(text)
    print(f"wrote {OUT} ({n} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
