import os
import time

import requests
import streamlit as st

API_URL = os.getenv(
    "OCR_API_URL",
    "http://backend:8002/ocr",
)

MAX_UPLOAD_SIZE = 20 * 1024 * 1024

st.set_page_config(
    page_title = "OCR Application",
    page_icon = "📝",
)

st.title("OCR Application")

st.write("Upload an image or PDF document and extract its text.")

uploaded_file = st.file_uploader(
    "Document",
    type = ["jpg", "jpeg", "png", "pdf"],
)

if "recognized_text" not in st.session_state:
    st.session_state.recognized_text = ""

if st.button("OCR", type = "primary"):
    if uploaded_file is None:
        st.warning("Please choose an image or PDF first.")
    else:
        file_data = uploaded_file.getvalue()

        if len(file_data) > MAX_UPLOAD_SIZE:
            st.error("The uploaded file is too large. Maximum size is 20 MB.")
        else:
            try:
                file_data = uploaded_file.getvalue()

                files = {
                    "file": (
                        uploaded_file.name,
                        file_data,
                        uploaded_file.type or "application/octet-stream",
                    )
                }

                response = requests.post(
                    API_URL,
                    files = files,
                    timeout = 60,
                )

                if not response.ok:
                    try:
                        error_data = response.json()
                        error_message = error_data.get(
                            "detail",
                            f"Backend returned HTTP {response.status_code}.",
                        )
                    except ValueError:
                        error_message = (
                            f"Backend returned HTTP {response.status_code}."
                        )

                    st.error(error_message)
                else:
                    response_data = response.json()
                    job_id = response_data.get("job_id")

                    if not isinstance(job_id, str):
                        st.error("The backend did not return a valid OCR job ID.")
                    else:
                        status_url = f"{API_URL}/status/{job_id}"
                        progress_bar = st.progress(0)
                        progress_text = st.empty()

                        while True:
                            status_response = requests.get(
                                status_url,
                                timeout=30,
                            )

                            if not status_response.ok:
                                try:
                                    error_data = status_response.json()
                                    error_message = error_data.get(
                                        "detail",
                                        (
                                            "The backend could not provide OCR "
                                            "status."
                                        ),
                                    )
                                except ValueError:
                                    error_message = (
                                        "The backend returned an invalid status response."
                                    )

                                st.error(error_message)
                                break

                            status_data = status_response.json()
                            job_status = status_data.get("status")

                            if job_status == "processing":
                                current_page = status_data.get("current_page", 0)
                                total_pages = status_data.get("total_pages", 0)

                                if total_pages > 0:
                                    progress_value = min(
                                        current_page / total_pages,
                                        1.0,
                                    )
                                    progress_bar.progress(progress_value)
                                    progress_text.write(
                                        f"Processing PDF page "
                                        f"{current_page}/{total_pages}"
                                    )
                                else:
                                    progress_text.write(
                                        "Preparing document..."
                                    )

                                time.sleep(1)

                            elif job_status == "completed":
                                progress_bar.progress(1.0)
                                progress_text.success("OCR completed.")

                                recognized_text = status_data.get("text")

                                if not isinstance(recognized_text, str):
                                    st.error(
                                        "The backend did not return valid recognized text."
                                    )
                                else:
                                    st.session_state.recognized_text = recognized_text

                                break

                            elif job_status == "failed":
                                st.error(
                                    status_data.get(
                                        "error",
                                        "OCR processing failed.",
                                    )
                                )
                                break

                            else:
                                st.error("The backend returned an unknown job status.")
                                break

            except requests.RequestException as error:
                st.error(
                    "Could not connect the OCR backend. "
                    "Make sure the backend is running."
                )
                st.exception(error)

            except ValueError as error:
                st.error("The backend returned invalid JSON.")
                st.exception(error)


if st.session_state.recognized_text:
    st.subheader("Recognized text")

    st.text_area(
        "OCR result",
        value = st.session_state.recognized_text,
        height = 300,
        label_visibility = "collapsed"
    )

    st.download_button(
        label = "Download TXT",
        data = st.session_state.recognized_text,
        file_name = "recognized_text.txt",
        mime = "text/plain",
    )

### Starting the frontend server ###
# cd ~/projects/ocr_web_app
# streamlit run frontend/app.py
### ---------------------------- ###