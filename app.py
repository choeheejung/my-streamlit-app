
import os, base64, json, requests
import streamlit as st
from openai import OpenAI

# OpenAI 클라이언트 생성
client = OpenAI()

# 페이지 설정
st.set_page_config(page_title="에이블 AI 개인비서", page_icon="🤖", layout="wide")
st.title("🤖 에이블 | 김 대리의 AI 개인비서")
st.write("회의록 분석부터 업무 관리까지 도와드립니다.")

# 역할 지정
sys_role = """
당신은 김 대리의 AI 개인비서 '에이블'입니다.

[역할]
1. 회의록의 주요 안건과 결정 사항 요약
2. 김 대리에게 배정된 업무와 마감기한 정리
3. 업무별 긴급도와 중요도 구분
4. 회의록을 바탕으로 사용자의 질문에 답변

[지시사항]
- 답변은 명확하고 간결하게 작성합니다.
- 회의록에 없는 내용은 임의로 만들어내지 않습니다.
- 사용자가 특정 직원의 담당 업무를 질문하면, 회의록에서 해당 직원의 업무만 추출합니다.
- 음성 인식 과정에서 담당자 이름이 잘못 인식될 수 있습니다.
- 회의록에 없는 이름이 언급되었지만 비슷한 이름이 존재한다면, 
  해당 인물을 의미하는지 사용자에게 확인합니다.
- 업무 마감기한은 회의록에 명시된 경우에만 작성합니다.
- 회의록에 없는 업무나 마감기한은 생성하지 않습니다.
- 긴급도와 중요도는 근거가 있을 때만 판단하며, 판단 근거가 부족하면 '확인 필요'로 표시합니다.
"""

# 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 지시 영역
with st.form("task_form"):
    uploaded_file = st.file_uploader("텍스트 파일 업로드", type=["txt"])
    audio_value = st.audio_input("무엇을 도와드릴까요?")
    submit = st.form_submit_button("실행")

# 버튼을 눌렀을 때만 실행
if submit:
    # 업로드 된 파일이 있으면 디코딩
    text = ""
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")

    # 녹음이 없으면 녹음 요청
    if audio_value is None:
        st.warning("음성을 녹음해주세요.")
        st.stop()

    # 시스템 메세지
    system_content = sys_role
    if text:
        system_content += f"\n\n다음 내용을 참고하세요:\n{text}"
    input_messages = [{"role": "system", "content": system_content}]

    # STT: 음성 → 텍스트
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.wav", audio_value, "audio/wav"),
        prompt="김 대리, 에이블, 회의록, 담당 업무, 마감기한"
        )
    st.subheader("🎤 음성 인식 결과")
    st.write(transcript.text)

    # 사용자 메시지
    input_messages.append({"role": "user", "content": transcript.text})

    # LLM 호출 및 응답 듣기
    with st.spinner("답변을 준비하고 있습니다..."):
        # LLM 호출
        response = client.responses.create(
            model="gpt-4o-mini",
            input=input_messages
        )
        answer = response.output_text

        # 응답 출력
        st.subheader("AI 답변")
        st.write(answer)
