#!/usr/bin/env python3
"""Enforce the ACL reference rules that got this paper desk-rejected.

The August 2026 submission was desk-rejected with:

    "several references do not report the full list of authors and do not
     include any DOI or link to ACL Anthology"

https://acl-org.github.io/ACLPUB/formatting.html requires, verbatim:
  * "All references are required to contain DOIs of all cited works when
     possible, or, as a second resort, links to ACL Anthology pages."
  * "Use full names for authors, not just initials."
  * "Arrange the references alphabetically by first author."

Offline by default. --online additionally resolves every DOI and URL, which is
what catches a dead identifier such as the JSTOR DOI 10.2307/4615733 (404).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

BIB = Path(__file__).resolve().parent.parent / "paper" / "custom.bib"


def parse(text: str) -> list[dict]:
    """Good-enough BibTeX reader: entries are '@type{key, field = {..}, ..}'."""
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        start = m.end()
        depth, i = 1, text.index("{", m.start())
        i += 1
        while i < len(text) and depth:
            depth += (text[i] == "{") - (text[i] == "}")
            i += 1
        body = text[start:i - 1]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*(\{|\")", body):
            name = fm.group(1).lower()
            openc = fm.group(2)
            closec = "}" if openc == "{" else '"'
            j, d = fm.end(), 1
            while j < len(body) and d:
                if openc == "{":
                    d += (body[j] == "{") - (body[j] == "}")
                elif body[j] == closec:
                    d -= 1
                j += 1
            fields[name] = body[fm.end():j - 1].strip()
        entries.append({"type": m.group(1).lower(), "key": m.group(2), **fields})
    return entries


def authors_of(entry: dict) -> list[str]:
    raw = re.sub(r"\s+", " ", entry.get("author", ""))
    return [a.strip() for a in raw.split(" and ") if a.strip()]


def given_is_only_initials(author: str) -> bool:
    """The rule is "not *just* initials", so flag only a fully abbreviated given
    name: 'Efron, B.' and 'Liu, Y. H.' yes; 'Hu, Edward J.' no; and
    'Buchanan, E. Kelly' no -- that is the byline the author publishes under."""
    if "," not in author:
        return False
    given = author.split(",", 1)[1].strip()
    if not given:
        return False
    return all(re.fullmatch(r"[A-Z]\.?", t) for t in given.split())


def resolve(url: str) -> tuple[bool, str]:
    p = subprocess.run(
        ["curl", "-sIL", "--max-time", "30", url,
         "-o", "/dev/null", "-w", "%{http_code}"],
        capture_output=True, text=True)
    code = (p.stdout or "").strip()
    return code.startswith("2") or code == "403", code  # 403 = bot wall, not dead


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true",
                    help="also check that every DOI and URL resolves")
    args = ap.parse_args()

    if not BIB.exists():
        print(f"FAIL: {BIB} not found")
        return 1
    entries = parse(BIB.read_text())
    problems: list[str] = []

    for e in entries:
        key = e["key"]
        if not e.get("doi") and not e.get("url"):
            problems.append(f"{key}: no doi and no url")
        auths = authors_of(e)
        if not auths:
            problems.append(f"{key}: no author field")
        for a in auths:
            if re.search(r"\bet\s+al\b", a, re.I):
                problems.append(f"{key}: author list truncated with 'et al.'")
            if given_is_only_initials(a):
                problems.append(f"{key}: given name is initials only ({a!r})")

    keys = [e["key"] for e in entries]
    if len(set(keys)) != len(keys):
        dupes = {k for k in keys if keys.count(k) > 1}
        problems.append(f"duplicate keys: {sorted(dupes)}")

    print(f"[bib] {len(entries)} entries in {BIB.name}")
    print(f"[bib] with DOI: {sum(1 for e in entries if e.get('doi'))}"
          f"   with URL: {sum(1 for e in entries if e.get('url'))}")

    if args.online:
        dead = []
        for e in entries:
            target = (f"https://doi.org/{e['doi']}" if e.get("doi")
                      else e.get("url"))
            ok, code = resolve(target)
            if not ok:
                dead.append(f"{e['key']}: {target} -> HTTP {code}")
            print(f"  {'ok ' if ok else 'DEAD'} {code:>3}  {e['key']}")
        problems += dead

    if problems:
        print(f"\nFAIL: {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK: every entry has a resolvable identifier and a full author list.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
