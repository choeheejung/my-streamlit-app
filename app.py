
import os, base64, json, requests
import smtplib
from email.mime.text import MIMEText
import streamlit as st
from openai import OpenAI

# OpenAI 클라이언트 생성
client = OpenAI()

# 이메일 발송 함수
def send_email(to, subject, body):
    sender = st.secrets["GMAIL_ADDRESS"]
    password = st.secrets["GMAIL_APP_PASSWORD"]

    # 메일 구성
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    # 메일 발송
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(sender, password)
            s.sendmail(sender, to, msg.as_string())
        return {"status": "success"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


# 페이지 설정
st.set_page_config(
    page_title="에이블 AI 개인비서",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 에이블 | 김 대리의 AI 개인비서")
st.write("회의록 분석부터 업무 관리까지 도와드립니다.")


# 역할 지정
sys_role = """
사용자는 '김 대리'이며, 당신은 김 대리의 AI 개인비서 '에이블'입니다.

[역할]
1. 회의록의 주요 안건과 결정 사항 요약
2. 김 대리에게 배정된 업무와 마감기한 정리
3. 업무별 긴급도와 중요도 구분
4. 회의록을 바탕으로 사용자의 질문에 답변

[지시사항]
- 답변은 명확하고 간결하게 작성합니다.
- 회의록에 없는 내용은 임의로 만들어내지 않습니다.
- 사용자가 특정 직원의 담당 업무를 질문하면, 회의록에서 해당 직원의 업무만 추출합니다.
- 음성 인식 오류로 담당자 이름이 잘못 변환될 수 있습니다.
- 회의록에 없는 이름이 언급되었지만 비슷한 이름이 존재한다면, 해당 인물을 의미하는지 사용자에게 확인합니다.
- 업무 마감기한은 회의록에 명시된 경우에만 작성합니다.
- 회의록에 없는 업무나 마감기한은 생성하지 않습니다.
- 긴급도와 중요도는 근거가 있을 때만 판단하며, 판단 근거가 부족하면 '확인 필요'로 표시합니다.
"""


# 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이메일 초안 저장
if "email_draft" not in st.session_state:
    st.session_state.email_draft = None

# 이메일 초안 버전
if "draft_version" not in st.session_state:
    st.session_state.draft_version = 0


# 지시 영역
with st.form("task_form"):
    uploaded_file = st.file_uploader(
        "텍스트 파일 업로드",
        type=["txt"]
    )
    audio_value = st.audio_input("무엇을 도와드릴까요?")
    submit = st.form_submit_button("실행")


# 버튼을 눌렀을 때만 실행
if submit:

    # 업로드된 파일 디코딩
    text = ""
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")

    # 녹음 확인
    if audio_value is None:
        st.warning("음성을 녹음해주세요.")
        st.stop()

    # 시스템 메시지
    system_content = sys_role

    if text:
        system_content += f"\n\n다음 내용을 참고하세요:\n{text}"

    input_messages = [
        {"role": "system", "content": system_content}
    ]

    # STT: 음성 → 텍스트
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.wav", audio_value, "audio/wav"),
        prompt="김 대리, 에이블, 회의록, 담당 업무, 마감기한"
    )

    st.subheader("🎤 음성 인식 결과")
    st.write(transcript.text)

    # 사용자 메시지
    input_messages.append({
        "role": "user",
        "content": transcript.text
    })

    # 이메일 작성 요청인지 확인
    email_request = (
        "이메일" in transcript.text or
        "메일" in transcript.text
    ) and (
        "작성" in transcript.text or
        "보내" in transcript.text or
        "발송" in transcript.text
    )

    # LLM 호출 및 응답
    with st.spinner("답변을 준비하고 있습니다..."):

        # 1. 회의록을 바탕으로 AI 답변 생성
        response = client.responses.create(
            model="gpt-4o-mini",
            input=input_messages
        )

        answer = response.output_text

        st.subheader("AI 답변")
        st.write(answer)

        # 2. 이메일 요청이라면 AI 답변을 이메일로 변환
        if email_request:

            email_response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "system",
                        "content": """
당신은 김 대리의 AI 개인비서 에이블입니다.

제공된 업무 정리 결과를 이메일 형식으로 변환하세요.

[이메일 작성 규칙]
- 김 대리는 이메일을 받는 사람입니다.
- 이메일 작성자는 AI 비서 에이블입니다.
- 제공된 업무 내용과 마감기한을 그대로 유지하세요.
- 긴급도와 중요도 정보가 있다면 유지하세요.
- '확인 필요'라고 표시된 내용도 그대로 유지하세요.
- 새로운 업무, 마감기한, 회의 내용을 추가하지 마세요.
- 업무 정보는 임의로 생략하거나 변경하지 마세요.
- 인사말과 맺음말만 자연스럽게 추가하세요.
- 첫 인사는 '안녕하세요, 김 대리님.'으로 작성하세요.
- 마지막 서명은 'AI 비서 에이블 드림'으로 작성하세요.
- 실제 이메일을 발송하지 말고 초안만 작성하세요.

반드시 다음 형식으로 작성하세요.

제목: 이메일 제목
본문:
이메일 본문
"""
                    },
                    {
                        "role": "user",
                        "content": answer
                    }
                ]
            )

            draft_text = email_response.output_text

            # 제목과 본문 분리
            if "본문:" in draft_text:

                subject_text, body_text = draft_text.split(
                    "본문:", 1
                )

                st.session_state.email_draft = {
                    "subject": subject_text.replace(
                        "제목:", ""
                    ).strip(),
                    "body": body_text.strip()
                }

                # 새 초안 생성 시 입력창 갱신
                st.session_state.draft_version += 1

            else:
                st.session_state.email_draft = None
                st.error("이메일 초안 형식을 확인할 수 없습니다.")

    # AI 답변을 음성으로 변환
    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="nova",
        input=answer
    )

    # 음성 파일 저장
    speech.stream_to_file("answer.mp3")

    # 음성 재생
    st.audio(
        "answer.mp3",
        format="audio/mp3",
        autoplay=True
    )


# 대화 기록 저장
if submit:
    st.session_state.messages.append({
        "user": transcript.text,
        "assistant": answer
    })

# 최근 대화 10개 유지
st.session_state.messages = st.session_state.messages[-10:]

# 대화 기록 초기화
if st.button("초기화"):
    st.session_state.messages.clear()

# 최근 대화 기록 출력
st.divider()
st.subheader("💬 최근 대화 기록")

for msg in reversed(st.session_state.messages):

    with st.chat_message("user"):
        st.write(msg["user"])

    with st.chat_message("assistant"):
        st.write(msg["assistant"])


# 이메일 발송 화면
st.divider()
st.subheader("📧 이메일 발송")

with st.form("email_form"):

    receiver = st.text_input("받는 사람 이메일")

    subject = st.text_input(
        "이메일 제목",
        value=(
            st.session_state.email_draft["subject"]
            if st.session_state.email_draft else ""
        ),
        key=f"subject_{st.session_state.draft_version}"
    )

    body = st.text_area(
        "이메일 내용",
        value=(
            st.session_state.email_draft["body"]
            if st.session_state.email_draft else ""
        ),
        key=f"body_{st.session_state.draft_version}"
    )

    send = st.form_submit_button("발송 승인")


# 발송 승인 후 이메일 전송
if send:

    if not receiver or not subject or not body:
        st.warning(
            "받는 사람, 제목, 내용을 모두 입력해주세요."
        )

    else:
        result = send_email(receiver, subject, body)

        if result["status"] == "success":
            st.success("이메일 발송 완료!")

        else:
            st.error("이메일 발송 실패")


# AI가 작성한 이메일 초안 표시
if st.session_state.email_draft:

    st.divider()
    st.subheader("✉️ AI 이메일 초안")

    st.write(
        "**제목:**",
        st.session_state.email_draft["subject"]
    )

    st.write("**본문:**")
    st.write(st.session_state.email_draft["body"])
