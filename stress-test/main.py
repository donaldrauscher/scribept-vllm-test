import aiohttp, asyncio, time, os, math

import pandas as pd
from datasets import load_dataset, Dataset


URL = "http://localhost:8000/v1/completions"
API_KEY = "scr1b3pt"

PROMPT = """A medical scribe needs to extract documentation from a doctor/patient conversation which is included below.

### Conversation:
{}

### Header:
{}

### Summary:
"""

SECTION_HEADING_MAP = {
    "FAM/SOCHX": "Family/Social History",
    "GENHX": "General History",
    "PASTMEDICALHX": "Past Medical History",
    "CC": "Chief Complaint",
    "ROS": "Review of Systems",
    "ALLERGY": "Allergies",
    "PASTSURGICAL": "Past Surgical History",
    "MEDICATIONS": "Current Medications",
    "ASSESSMENT": "Clinical Assessment or Evaluation",
    "EXAM": "Physical Examination",
    "DIAGNOSIS": "Diagnosis or Diagnoses",
    "DISPOSITION": "Patient Disposition (e.g., discharge, transfer)",
    "PLAN": "Plan of Care or Treatment Plan",
    "EDCOURSE": "Emergency Department Course",
    "IMMUNIZATIONS": "Immunization History or Records",
    "IMAGING": "Radiology or Imaging Studies",
    "GYNHX": "Gynecological History",
    "OTHER_HISTORY": "Other Relevant History",
    "PROCEDURES": "Medical or Surgical Procedures",
    "LABS": "Laboratory Results"
}


def load_notes() -> Dataset:
    def _prepare_prompts(example):
        example['text'] = PROMPT.format(
            example['dialogue'],
            example['section_header'])
        return example


    def _map_section_header_description(example):
        example["section_header"] = SECTION_HEADING_MAP[example["section_header"]]
        return example

    base_path = "clinical_visit_note_summarization_corpus/data/mts-dialog"
    train_files = [
        os.path.join(base_path, 'MTS_Dataset_TrainingSet.csv'),
        os.path.join(base_path, 'MTS_Dataset_Final_200_TestSet_1.csv'),
        os.path.join(base_path, 'MTS_Dataset_Final_200_TestSet_2.csv')
    ]
    test_files = [
        os.path.join(base_path, 'MTS_Dataset_ValidationSet.csv')
    ]

    notes = load_dataset(
        "csv",
        data_files={
            "train": train_files,
            "test": test_files
        }
    )

    notes = notes.map(_map_section_header_description)
    notes = notes.map(_prepare_prompts)
    return notes


async def send_request(session: aiohttp.ClientSession, note: str, request_id: int, delay_between_batches: float = 0) -> dict:
    """
    Sends request to vLLM service
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": "donaldrauscher/medical-scribe-vllm",
        "prompt": note,
        "temperature": 0.0,
        "max_tokens": 500
    }

    # multiple by request_id to kick off each request at different time
    if delay_between_batches > 0:
        await asyncio.sleep(request_id*delay_between_batches)

    start_time = time.perf_counter()
    async with session.post(URL, json=payload, headers=headers) as response:
        end_time = time.perf_counter()
        duration = end_time - start_time

        try:
            json_response: dict | None = await response.json()
        except aiohttp.ContentTypeError:
            json_response = None

        return {
            "request_id": request_id, 
            "status": response.status,
            "response": json_response['choices'][0]['text'], 
            "duration": duration
        }


async def stress_test(notes: Dataset, delay_between_batches: float = 0) -> pd.DataFrame:
    """
    Invoke request to vLLM service for each note

    If `delay_between_batches` == -1, then run in series
    """
    async with aiohttp.ClientSession() as session:
        if delay_between_batches == -1:
            results = []
            for i, note in enumerate(notes):
                result = await send_request(session, note, i, delay_between_batches)
                results.append(result)
        else:
            results = []
            for i, note in enumerate(notes):
                result = send_request(session, note, i, delay_between_batches)
                results.append(result)
            results = await asyncio.gather(*results)

    return pd.DataFrame(results)


if __name__ == "__main__":
    n_requests = 100
    notes = load_notes()['train']['text']
    if len(notes) < n_requests:
        notes = (notes * math.ceil(n_requests/len(notes)))[:n_requests]
    else:
        notes = notes[:n_requests]

    print(f"Running stress test {n_requests} requests in series:")
    df = asyncio.run(stress_test(notes, -1))
    print(df.describe())

    delay = 0
    print(f"\nRunning stress test {n_requests} requests with {delay:.1f} seconds between requests:")
    df = asyncio.run(stress_test(notes, delay))
    print(df.describe())

    delay = 0.1
    print(f"\nRunning stress test {n_requests} requests with {delay:.1f} seconds between requests:")
    df = asyncio.run(stress_test(notes, delay))
    print(df.describe())

    delay = 0.2
    print(f"\nRunning stress test {n_requests} requests with {delay:.1f} seconds between requests:")
    df = asyncio.run(stress_test(notes, delay))
    print(df.describe())
