#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TOC.md 를 pages/*.md 의 실제 H1 제목과 동기화하고 무결성을 검사한다.

위키독스는 TOC.md 로 책의 구조와 **책 제목**을 만든다.
- 첫 줄의 `# ...` 이 **책 제목(최상위)** 이 된다. 이 줄을 절대 다른 것으로 바꾸지 말 것.
- 각 항목의 대괄호 안 문자열이 페이지 제목이 된다.

교재를 개편해 페이지의 H1 을 바꿨는데 TOC.md 를 안 고치면 목차와 본문 제목이 어긋난다.
개편 후 배포 전에 반드시 이 스크립트를 돌릴 것:

    python3 sync_toc.py           # 검사만 (CI/배포 전 게이트)
    python3 sync_toc.py --write   # TOC.md 제목을 실제 H1 로 갱신
"""
import re
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
BOOK_TITLE = "# 지능형 IoT를 위한 온디바이스AI"
LINK = re.compile(r'^(\s*\*\s*)\[(.+?)\]\((pages/[^)]+\.md)\)\s*$')


def h1_of(path: pathlib.Path) -> str:
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def main() -> int:
    write = "--write" in sys.argv
    toc = ROOT / "TOC.md"
    lines = toc.read_text(encoding="utf-8").split("\n")
    problems, out, refs = [], [], set()

    if lines[0].strip() != BOOK_TITLE:
        problems.append(f"[치명] TOC.md 첫 줄이 책 제목이 아니다: {lines[0]!r}\n"
                        f"       반드시 {BOOK_TITLE!r} 이어야 최상위 목차가 주차 제목으로 바뀌지 않는다.")

    for ln in lines:
        m = LINK.match(ln)
        if not m:
            out.append(ln)
            continue
        indent, title, rel = m.groups()
        refs.add(rel)
        f = ROOT / rel
        if not f.exists():
            problems.append(f"[죽은 링크] {rel} (TOC 제목: {title})")
            out.append(ln)
            continue
        h1 = h1_of(f)
        if h1 and h1 != title:
            problems.append(f"[제목 불일치] {rel}\n    TOC: {title}\n    본문: {h1}")
            out.append(f"{indent}[{h1}]({rel})")
        else:
            out.append(ln)

    have = {f"pages/{p.name}" for p in (ROOT / "pages").glob("*.md")}
    for orphan in sorted(have - refs):
        problems.append(f"[고아 페이지] {orphan} 이 TOC.md 에 없다")

    # 제로패딩 검사 — 위키독스는 제목 문자열 기준으로 정렬한다
    for ln in out:
        m = LINK.match(ln)
        if m and re.match(r'^\d주차', m.group(2)):
            problems.append(f"[정렬 위험] 한 자리 주차 표기: {m.group(2)}")

    # 그림 참조 무결성
    for p in sorted((ROOT / "pages").glob("*.md")):
        body = p.read_text(encoding="utf-8")
        for png in re.findall(r'!\[[^\]]*\]\(\.\./assets/([^)]+)\)', body):
            if not (ROOT / "assets" / png).exists():
                problems.append(f"[그림 없음] {p.name} → assets/{png}")
        if re.search(r'\.svg\)', body):
            problems.append(f"[.svg 잔존] {p.name} 이 SVG 를 직접 참조한다 (PNG 로 바꿀 것)")

    if write:
        toc.write_text("\n".join(out), encoding="utf-8")
        print("TOC.md 를 갱신했다.")

    if problems:
        print(f"\n검사 실패 — {len(problems)}건\n")
        for x in problems:
            print(" ·", x)
        if not write:
            print("\n제목 불일치는 `python3 sync_toc.py --write` 로 자동 수정된다.")
        return 1
    print(f"검사 통과 — 페이지 {len(have)}개 · 죽은 링크 0 · 고아 0 · 제목 불일치 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
