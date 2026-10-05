import ast
import csv
import os
from dataclasses import dataclass


SB_CSV = "StreamingBench/Real_Time_Visual_Understanding.csv"


@dataclass
class StreamingBenchQuestion:
    sample_id: str
    sample_index: int
    question_id: str
    task_type: str
    question: str
    timestamp: float
    timestamp_text: str
    answer: str
    options: list[str]


def timestamp_to_sec(timestamp: str) -> int:
    h, m, s = timestamp.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def streamingbench_csv_path(data_dir: str) -> str:
    path = os.path.join(data_dir, SB_CSV)
    if not os.path.exists(path):
        raise FileNotFoundError(f"StreamingBench CSV not found: {path}")
    return path


def streamingbench_video_path(data_dir: str, sample_id: str) -> str:
    path = os.path.join(data_dir, "videos", f"{sample_id}.mp4")
    if not os.path.exists(path):
        raise FileNotFoundError(f"StreamingBench video not found: {path}")
    return path


def parse_options(raw: str) -> list[str]:
    if raw.startswith("["):
        return list(ast.literal_eval(raw))
    return [raw]


def render_prompt(q: StreamingBenchQuestion) -> str:
    return (
        f"[t={q.timestamp_text}] {q.question}\n"
        f"Options: {' '.join(q.options)}\n"
        "Answer with the single option letter (A/B/C/D), then one short sentence."
    )


def list_sample_ids(data_dir: str) -> list[str]:
    csv_path = streamingbench_csv_path(data_dir)
    sample_ids = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            marker = "_sample_"
            if marker not in row["question_id"]:
                continue
            suffix = row["question_id"].split(marker, 1)[1]
            sample_num = suffix.split("_", 1)[0]
            sample_id = f"sample_{sample_num}"
            video_path = os.path.join(data_dir, "videos", f"{sample_id}.mp4")
            if os.path.exists(video_path):
                sample_ids.add(sample_id)
    return sorted(sample_ids, key=lambda x: int(x.split("_")[1]))


def load_questions(data_dir: str, sample_id: str) -> list[StreamingBenchQuestion]:
    csv_path = streamingbench_csv_path(data_dir)
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if f"_{sample_id}_" not in row["question_id"]:
                continue
            rows.append(row)

    rows.sort(key=lambda row: timestamp_to_sec(row["time_stamp"]))
    questions = []
    for index, row in enumerate(rows):
        questions.append(
            StreamingBenchQuestion(
                sample_id=sample_id,
                sample_index=index,
                question_id=row["question_id"],
                task_type=row["task_type"],
                question=row["question"],
                timestamp=float(timestamp_to_sec(row["time_stamp"])),
                timestamp_text=row["time_stamp"],
                answer=row["answer"].strip(),
                options=parse_options(row["options"]),
            )
        )
    if not questions:
        raise ValueError(f"No StreamingBench questions found for sample_id={sample_id}")
    return questions
