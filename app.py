
import os
import hashlib
import tempfile
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter

# 1. 페이지 설정
st.set_page_config(
    page_title="AIVLE School 학습 도우미",
    page_icon="📘",
    layout="wide"
)

CHROMA_DIR = "chroma_db"
COLLECTION = "aivle_docs"

# 2. 문서 포맷터 (어제 코드)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 3. RAG 체인 생성
def make_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        streaming=True
    )

    # 시스템 프롬프트
    s_msg = """
당신은 KT AIVLE School을 처음 시작하는 교육생을 위한
친절하고 긍정적인 AI 학습 멘토입니다.

다음 지침을 따르세요.
1. 제공된 문서(context)에 근거하여 정확하고 명확하게 답변하세요.
2. 문서와 질문의 표현이 달라도 의미가 같다면 관련 내용을 활용하세요.
3. 문서에 없는 사실은 추측하지 말고, 확인되지 않는다고 안내하세요.
4. 초보 교육생도 이해하기 쉽도록 핵심부터 간결하게 설명하세요.
5. 학습에 어려움이나 불안을 표현하면 공감하고 용기를 북돋아 주세요.
6. 교육생이 적극적으로 도전할 수 있도록 상황에 맞는 실천 방법을 제안하세요.
7. 따뜻하고 긍정적인 말투를 사용하되, 과장하거나 매번 억지로 격려하지 마세요.

[문서]
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", s_msg),
        ("human", "{input}")
    ])

    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# 4. 기존 백서 RAG (캐싱)
@st.cache_resource
def get_default_chain():
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding,
        persist_directory=CHROMA_DIR
    )

    if vectorstore._collection.count() == 0:
        raise ValueError(
            "기본 백서 DB가 비어 있습니다. "
            "chroma_db 폴더를 확인하세요."
        )

    return make_chain(vectorstore)


# 5. 어제 만든 RAG 파이프라인을 업로드용으로 적용
def rag_pipeline(filepath):
    # PDF 읽기
    loader = PyMuPDF4LLMLoader(filepath)
    docs = loader.load()

    # 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("PDF에서 텍스트를 추출하지 못했습니다.")

    # 임베딩
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # 업로드 문서 전용 임시 벡터 저장소
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding
    )

    # RAG 체인 생성
    return make_chain(vectorstore)


# 6. 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_chain" not in st.session_state:
    st.session_state.uploaded_chain = None

if "file_key" not in st.session_state:
    st.session_state.file_key = None

# 7. 사이드바
with st.sidebar:
    st.title("📘 AIVLE School")
    st.caption("AI 학습 도우미")

    st.divider()

    st.subheader("📁 문서 업로드")

    uploaded_file = st.file_uploader(
        "PDF 파일을 선택하세요.",
        type=["pdf"]
    )

    st.caption("PDF를 업로드하면 해당 문서로 답변합니다.")

    st.divider()

    # 학습 카테고리
    st.subheader("📚 학습 카테고리")

    category = st.selectbox(
        "궁금한 분야를 선택하세요.",
        ["입과 준비", "교육생활", "학습·복습", "프로젝트", "취업 준비"]
    )

    # 카테고리별 추천 질문
    question_dict = {
        "입과 준비": [
            "AIVLE School 지원 자격은 무엇인가요?",
            "입과 전에 무엇을 준비하면 좋을까요?"
        ],
        "교육생활": [
            "복습 영상은 언제 올라오나요?",
            "교육 중 출결 관리는 어떻게 하나요?"
        ],
        "학습·복습": [
            "코딩이 처음인데 어떻게 공부하면 좋을까요?",
            "효율적으로 복습하는 방법은 무엇인가요?"
        ],
        "프로젝트": [
            "미니프로젝트는 어떻게 준비하면 좋을까요?",
            "팀 프로젝트에서 협업을 잘하려면 어떻게 해야 하나요?"
        ],
        "취업 준비": [
            "취업 준비는 언제부터 시작해야 하나요?",
            "어떤 취업 지원 프로그램이 있나요?"
        ]
    }

    # 추천 질문
    st.subheader("💡 추천 질문")

    recommended_questions = question_dict[category]

    selected_question = None

    for i, question in enumerate(recommended_questions):
        if st.button(
            question,
            key=f"recommend_{i}",
            use_container_width=True
        ):
            selected_question = question

    st.divider()

    # 대화 내용을 텍스트로 변환
    chat_text = ""

    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            role = "사용자"
        else:
            role = "AI 학습 도우미"

        chat_text += f"[{role}]\n{message.content}\n\n"

    # 대화 다운로드
    st.download_button(
        label="📥 대화 내용 다운로드",
        data=chat_text,
        file_name="aivle_chat_history.txt",
        mime="text/plain",
        use_container_width=True,
        disabled=not st.session_state.messages
    )

    # 대화 삭제
    if st.button("🗑️ 대화 삭제", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# 8. 업로드 문서 처리
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_key = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.file_key != file_key:
        with st.spinner("문서를 분석하고 있습니다..."):
            tmp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                new_chain = rag_pipeline(tmp_path)

                st.session_state.uploaded_chain = new_chain
                st.session_state.file_key = file_key
                st.session_state.messages = []

            except Exception as e:
                st.error(f"문서 처리 오류: {e}")
                st.stop()

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

    chain = st.session_state.uploaded_chain
    st.sidebar.success(f"문서 준비 완료: {uploaded_file.name}")

else:
    if st.session_state.file_key is not None:
        st.session_state.file_key = None
        st.session_state.uploaded_chain = None
        st.session_state.messages = []

    try:
        chain = get_default_chain()
    except Exception as e:
        st.error(f"기본 백서 연결 오류: {e}")
        st.stop()


# 9. 메인 화면
st.title("📘 AIVLE School 학습 도우미")
st.caption("AIVLE School에 대한 궁금증을 AI에게 질문해 보세요.")

st.divider()

if uploaded_file is None:
    st.info("현재 기본 AIVLE School 학습백서를 사용 중입니다.")
else:
    st.info(f"현재 문서: {uploaded_file.name}")


# 10. 기존 대화 출력
for message in st.session_state.messages:
    role = (
        "user"
        if isinstance(message, HumanMessage)
        else "assistant"
    )

    with st.chat_message(role):
        st.markdown(message.content)


# 11. 질문 입력
user_input = st.chat_input("궁금한 내용을 입력하세요.")

if selected_question:
    user_input = selected_question


# 12. RAG 답변
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하고 있습니다..."):
            try:
                response_stream = chain.stream(
                    {"input": user_input}
                )
                answer = st.write_stream(response_stream)

            except Exception as e:
                st.error(f"답변 생성 오류: {e}")
                st.stop()

    # 대화 저장
    st.session_state.messages.append(
        HumanMessage(content=user_input)
    )

    st.session_state.messages.append(
        AIMessage(content=answer)
    )
