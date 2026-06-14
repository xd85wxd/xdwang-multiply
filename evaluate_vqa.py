import ijson
import random
import time
import csv
import gc
from pathlib import Path

from model import ask_vqa


QUESTION_FILE = \
    "VQA/v2_OpenEnded_mscoco_val2014_questions.json"

ANNOTATION_FILE = \
    "VQA/v2_mscoco_val2014_annotations.json"

IMAGE_DIR = \
    "VQA/val2014"

NUM_SAMPLES = 500

RANDOM_SEED = 42

RESULT_FILE = "vqa_results.csv"


print("Sampling questions...")

random.seed(RANDOM_SEED)

samples = []

with open(
    QUESTION_FILE,
    "rb"
) as f:

    for idx, q in enumerate(
        ijson.items(
            f,
            "questions.item"
        )
    ):

        if idx < NUM_SAMPLES:

            samples.append(q)

        else:

            j = random.randint(
                0,
                idx
            )

            if j < NUM_SAMPLES:

                samples[j] = q

print(
    "Sample Questions:",
    len(samples)
)

sample_qids = {
    q["question_id"]
    for q in samples
}

gc.collect()


print("Loading annotations...")

ann_map = {}

with open(
    ANNOTATION_FILE,
    "rb"
) as f:

    for ann in ijson.items(
        f,
        "annotations.item"
    ):

        qid = ann["question_id"]

        if qid in sample_qids:

            ann_map[qid] = ann

            if len(ann_map) == NUM_SAMPLES:

                break

gc.collect()

print(
    "Matched Annotations:",
    len(ann_map)
)

total_score = 0.0

start_time = time.time()

print("=" * 60)
print("Start Evaluation")
print("=" * 60)


csv_file = open(
    RESULT_FILE,
    "w",
    newline="",
    encoding="utf-8-sig"
)

writer = csv.writer(csv_file)

writer.writerow(
    [
        "question_id",
        "image",
        "question",
        "ground_truth",
        "prediction",
        "match_count",
        "vqa_score",
        "infer_time(s)"
    ]
)



for idx, q in enumerate(samples):

    question_id = q["question_id"]

    image_id = q["image_id"]

    question = q["question"]

    image_name = \
        f"COCO_val2014_{image_id:012d}.jpg"

    image_path = str(
        Path(IMAGE_DIR) / image_name
    )

    annotation = ann_map[
        question_id
    ]

    gt_answer = annotation[
        "multiple_choice_answer"
    ]

    answers = annotation[
        "answers"
    ]

    try:

        infer_start = time.time()

        pred_answer = ask_vqa(
            image_path,
            question
        )

        infer_time = (
            time.time() - infer_start
        )

        pred_answer = pred_answer.strip()

    except Exception as e:

        pred_answer = f"ERROR: {e}"

        infer_time = 0

    pred = pred_answer.lower().strip()

    match_count = 0

    for ans in answers:

        gt = ans[
            "answer"
        ].lower().strip()

        if pred == gt:
            match_count += 1

    score = min(
        match_count / 3.0,
        1.0
    )

    total_score += score


    writer.writerow(
        [
            question_id,
            image_name,
            question,
            gt_answer,
            pred_answer,
            match_count,
            round(score, 3),
            round(infer_time, 3)
        ]
    )

    current_acc = (
        total_score / (idx + 1)
    )

    print(
        f"[{idx+1}/{NUM_SAMPLES}] "
        f"VQA_ACC={current_acc:.4f}"
    )

    print("Q :", question)
    print("GT:", gt_answer)
    print("PR:", pred_answer)

    print(
        "Match:",
        match_count,
        "/10"
    )

    print(
        "Score:",
        round(score, 3)
    )

    print(
        f"Time: {infer_time:.2f}s"
    )

    print("-" * 60)


csv_file.close()



elapsed = time.time() - start_time

final_acc = (
    total_score / NUM_SAMPLES
)

print("=" * 60)
print("Finished")
print("=" * 60)

print(
    "Samples:",
    NUM_SAMPLES
)

print(
    "VQA Accuracy:",
    round(final_acc, 4)
)

print(
    "Total Time:",
    round(elapsed, 2),
    "s"
)

print(
    "Average Time:",
    round(
        elapsed / NUM_SAMPLES,
        2
    ),
    "s/sample"
)

print(
    "Saved:",
    RESULT_FILE
)