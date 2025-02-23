# translateAPI
**개요**

- 버그 바운티 플랫폼 보안 기업 연계 프로젝트
- WYSIWYG로 작성된 보안 리포트를 다국어로 번역 및 정형화 지원

**요구 사항**

- 다국어 번역, 우선순위 : 영어, 아라비아어, 인도어, 러시아어, 일본어
- aws에서 구동하는 private llm 구축. (openai api와 같은 모델 금지)
- 리포트 다국어 처리 기능 API 구현
- UI 개발

**기능 설명**

- 리포트 db의 데이터와 번역할 target language를 input으로 받고, output으로 번역 출력
- 텍스트 에디터로 작성된 리포트의 마크다운, html, base64 등 형식 구조 유지

**모델**

- qwen 2.5 32b instruct 4bit 양자화

**데이터 및 평가**

- 기업 소유 리포트 (비공개 데이터)
- CWE(https://cwe.mitre.org/index.html)
- 보완 관련 데이터  (한국인터넷진흥원)
