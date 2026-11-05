
import os
import re
import json
import base64
import hashlib
import random
import smtplib
import tempfile
from pathlib import Path
from email.message import EmailMessage
from typing import Literal

import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field

# 이메일 HTML은 html_report.py에서 관리합니다.
from html_report import create_html_email, create_html_report

# =========================================================
# 1. 기본 설정
# =========================================================

st.set_page_config(
    page_title="에이블 | 김대리의 개인비서",
    page_icon="🦦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Secrets 및 환경 변수에서 설정 읽기 ──
def get_secret(name):
    try:
        value = st.secrets.get(name, None)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name, "")

api_key = get_secret("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY를 Streamlit Cloud Secrets에 등록하세요.")
    st.stop()

client = OpenAI(api_key=api_key)

MODEL = "gpt-4o-mini"
TEMPERATURE = 0

# =========================================================
