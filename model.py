import os

from typing import Literal

import google.generativeai as genai

genai.configure(api_key=os.getenv("API_KEY"))


def get_model(name: Literal["gemini-pro", "gemini-pro-vision"] = "gemini-pro"):
    return genai.GenerativeModel(name)
