import json
import os
import re
import tempfile
import uuid

from openai import OpenAI
import pandas as pd
import pdfplumber
from flask import Flask, flash, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {".pdf", ".xls", ".xlsx", ".csv"}
OUTPUT_DIR = tempfile.gettempdir()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret")
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def extract_pdf_text(pdf_path: str) -> str:
    text_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)
    return "\n".join(text_pages)


def load_spreadsheet(path: str):
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".csv":
        df = pd.read_csv(path, dtype=str)
    else:
        df = pd.read_excel(path, dtype=str)

    normalized_columns = {str(column).strip().lower()                          : column for column in df.columns}
    if "label" not in normalized_columns or "value" not in normalized_columns:
        raise ValueError(
            "Spreadsheet must contain columns named 'label' and 'value'.")

    df = df.fillna("").astype(str)
    df = df[[normalized_columns["label"], normalized_columns["value"]]]
    df.columns = ["label", "value"]
    return df.to_dict(orient="records")


def find_value_in_text(value: str, text: str):
    normalized_value = normalize_text(value)
    normalized_text = normalize_text(text)
    if not normalized_value:
        return None
    index = normalized_text.find(normalized_value)
    if index == -1:
        return None
    start = max(index - 80, 0)
    end = min(index + len(normalized_value) + 80, len(normalized_text))
    return normalized_text[start:end]


def find_value_by_label_context(label: str, value: str, text: str):
    normalized_label = normalize_text(label)
    normalized_value = normalize_text(value)
    normalized_text = normalize_text(text)
    if not normalized_label or not normalized_value:
        return None

    label_tokens = [token for token in re.findall(
        r"\w+", normalized_label) if len(token) > 2]
    if not label_tokens:
        label_tokens = [normalized_label]

    lines = [line.strip()
             for line in normalized_text.splitlines() if line.strip()]
    threshold = max(1, len(label_tokens) // 2)
    for idx, line in enumerate(lines):
        matches = sum(1 for token in label_tokens if token in line)
        if matches >= threshold:
            search_window = " ".join(lines[idx: idx + 3])
            if normalized_value in search_window:
                value_index = search_window.find(normalized_value)
                start = max(value_index - 80, 0)
                end = min(value_index + len(normalized_value) +
                          80, len(search_window))
                return search_window[start:end]
    return None


def check_with_openai(label: str, value: str, pdf_text: str):
    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "present": False,
            "found_text": "",
            "reason": "OPENAI_API_KEY is not configured.",
        }

    snippet = pdf_text
    prompt = (
        "You are a data verification assistant. "
        "Determine whether the provided label and value are present in the excel file and correctly matched in the pdf document. The label does not have to be an exact match, but should be contextually relevant. The value should be present in the vicinity of the label or anywhere in the document if the label is ambiguous.\n\n"
        "Respond with JSON only, using keys present, found_text, and reason.\n\n"
        f"Label: {label}\n"
        f"Value: {value}\n\n"
        "PDF text excerpt:\n"
        f"{snippet}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a verifier."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return {
            "present": bool(data.get("present") is True),
            "found_text": data.get("found_text", ""),
            "reason": data.get("reason", ""),
        }
    except Exception as exc:
        return {
            "present": False,
            "found_text": "",
            "reason": f"OpenAI fallback failed: {exc}",
        }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    download_name = None

    if request.method == "POST":
        pdf_file = request.files.get("pdf_file")
        spreadsheet_file = request.files.get("spreadsheet_file")

        if not pdf_file or not spreadsheet_file:
            flash("Please upload both a PDF and an Excel/CSV file.")
            return redirect(request.url)

        if not allowed_file(pdf_file.filename) or not allowed_file(spreadsheet_file.filename):
            flash(
                "Unsupported file type. Use PDF for the loan estimate and CSV/XLS/XLSX for the spreadsheet.")
            return redirect(request.url)

        pdf_path = None
        spreadsheet_path = None
        try:
            pdf_path = tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(pdf_file.filename)[1]).name
            spreadsheet_path = tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(spreadsheet_file.filename)[1]).name
            pdf_file.save(pdf_path)
            spreadsheet_file.save(spreadsheet_path)

            pdf_text = extract_pdf_text(pdf_path)
            rows = load_spreadsheet(spreadsheet_path)

            comparisons = []
            for row in rows:
                label = row.get("label", "").strip()
                value = row.get("value", "").strip()
                snippet = find_value_in_text(
                    value, pdf_text) if value else None
                matched = bool(snippet)
                method = "direct"
                issue = None

                if not matched and label and value:
                    snippet = find_value_by_label_context(
                        label, value, pdf_text)
                    if snippet:
                        matched = True
                        method = "label-context"

                if not matched:
                    if os.environ.get("OPENAI_API_KEY"):
                        method = "openai"
                        ai_result = check_with_openai(label, value, pdf_text)
                        matched = ai_result["present"]
                        snippet = ai_result["found_text"]
                        issue = ai_result["reason"]
                    else:
                        issue = "Value not found by direct search. Set OPENAI_API_KEY for AI fallback."

                comparisons.append({
                    "label": label,
                    "value": value,
                    "matched": matched,
                    "match_method": method,
                    "found_text": snippet or "",
                    "issue": None if matched else issue,
                })

            output = {
                "summary": {
                    "total_rows": len(comparisons),
                    "matched": sum(1 for item in comparisons if item["matched"]),
                    "unmatched": sum(1 for item in comparisons if not item["matched"]),
                },
                "rows": comparisons,
                "discrepancies": [item for item in comparisons if not item["matched"]],
            }

            download_name = f"loan_estimate_discrepancies_{uuid.uuid4().hex}.json"
            output_path = os.path.join(OUTPUT_DIR, download_name)
            with open(output_path, "w", encoding="utf-8") as outfile:
                json.dump(output, outfile, indent=2, ensure_ascii=False)

            result = output
        except Exception as exc:
            flash(f"Error processing files: {exc}")
            return redirect(request.url)
        finally:
            for path in (pdf_path, spreadsheet_path):
                if path and os.path.exists(path):
                    os.unlink(path)

    return render_template("index.html", result=result, download_name=download_name)


@app.route("/download/<filename>")
def download(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        flash("Download file not found.")
        return redirect(url_for("index"))
    return send_file(path, as_attachment=True, download_name=filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
