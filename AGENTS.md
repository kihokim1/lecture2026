# AGENTS.md — 교재 저장소 작성 규칙
# date 2026. 8.12. 
# Author: Kim Kiho

위키독스 깃허브 연동 문서 저장소. push하면 자동 발행됩니다.

## 파일 명명
- 페이지: `pages/wNN-i-이름.md` (예: `w04-1-intro.md`). H1 제목은 `# 4주차 1교시. 프루닝의 원리`처럼 TOC 번호와 일치.
- 개요 페이지: `pages/wNN-0-overview.md`.
- 그림: `assets/wNN_pX_주제_번호.png`. 페이지에서 `![설명](../assets/파일.png)`로 참조.

## 새 주차 추가 절차
1. `assets/`에 그림 PNG 추가.
2. `pages/`에 교시별 md 추가(개요 + 1·2·3교시).
3. `TOC.md`에 항목 추가(주차=최상위, 교시=하위 들여쓰기 2칸).
4. `git push origin main`.

## 규칙
- H1(`#`)은 페이지당 하나(제목). 이후 `##`, `###`.
- 이미지는 SVG 대신 PNG 사용(위키독스 렌더 안정성).
