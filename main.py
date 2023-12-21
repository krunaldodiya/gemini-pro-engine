import io
import google.ai.generativelanguage as glm
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from model import get_model
from PIL import Image

gemini_pro, gemini_pro_vision = st.tabs(["Gemini Pro", "Gemini Pro Vision"])


def get_mime_type(image: Image) -> str:
    image_format: str = image.format
    image_format = image_format.lower()
    return f"image/{image_format}"


def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def main():
    with gemini_pro:
        st.header("Gemini Pro Text Generation")
        st.write("")

        with st.form("text_form"):
            text_prompt = st.text_input(
                "Enter your prompt:",
                placeholder="Prompt",
                label_visibility="visible",
                key="text_prompt",
            )

            submit_button = st.form_submit_button("Generate Response")

            if submit_button:
                try:
                    if text_prompt == "":
                        return st.write("Provide Prompt")

                    model = get_model("gemini-pro")
                    response = model.generate_content(text_prompt)
                    response.resolve()

                    st.write("")
                    st.header(":blue[Response]")
                    st.write("")
                    st.markdown(response.text)
                except Exception as e:
                    st.error("Error: {}".format(e))

    with gemini_pro_vision:
        st.header("Gemini Pro Image Generation")
        st.write("")

        with st.form("image_form"):
            uploaded_file = st.file_uploader(
                "Choose an image",
                accept_multiple_files=False,
                type=["png", "jpg", "jpeg", "img", "webp"],
            )

            image_prompt = st.text_input(
                "Enter your prompt:",
                placeholder="Prompt",
                label_visibility="visible",
                key="image_prompt",
            )

            submit_button = st.form_submit_button("Generate Response")

            if submit_button:
                try:
                    if uploaded_file is None:
                        return st.write("Provide Image")

                    if image_prompt == "":
                        return st.write("Provide Image Prompt")

                    model = get_model("gemini-pro-vision")

                    image_file = Image.open(uploaded_file)

                    st.image(image_file)

                    response = model.generate_content(
                        glm.Content(
                            parts=[
                                glm.Part(text=image_prompt),
                                glm.Part(
                                    inline_data=glm.Blob(
                                        mime_type=get_mime_type(image_file),
                                        data=image_to_byte_array(image_file),
                                    )
                                ),
                            ]
                        ),
                    )

                    response.resolve()

                    st.write("")
                    st.header(":blue[Response]")
                    st.write("")
                    st.markdown(response.text)
                except Exception as e:
                    st.error("Error: {}".format(e))


if __name__ == "__main__":
    main()
