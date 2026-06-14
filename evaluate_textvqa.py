import time
import csv
import random
import gc
from datasets import load_dataset

from model import ask_textvqa  


NUM_SAMPLES = 200
RANDOM_SEED = 42
RESULT_FILE = "textvqa_results.csv"


print("Loading TextVQA dataset...")

ds = load_dataset(
    "Multimodal-Fatima/TextVQA_validation",
    split="validation"
)

print(ds)


random.seed(RANDOM_SEED)

indices = random.sample(range(len(ds)), NUM_SAMPLES)


csv_file = open(RESULT_FILE, "w", newline="", encoding="utf-8-sig")
writer = csv.writer(csv_file)

writer.writerow([
    "question_id",
    "image_id",
    "question",
    "ground_truth",
    "prediction",
    "match_count",
    "score",
    "time(s)"
])


total_score = 0.0
start_time = time.time()

print("=" * 60)
print("Start TextVQA Evaluation")
print("=" * 60)


for i, idx in enumerate(indices):

    sample = ds[idx]

    image = sample["image"]   
    question = sample["question"]
    answers = sample["answers"]
    question_id = sample["question_id"]
    image_id = sample["image_id"]

    try:
        t0 = time.time()
        pred = ask_textvqa(image, question)

        infer_time = time.time() - t0

    except Exception as e:
        pred = f"ERROR: {e}"
        infer_time = 0

    pred = pred.lower().strip()

    match_count = 0

    for a in answers:
        if pred == a.lower().strip():
            match_count += 1

    score = min(match_count / 3.0, 1.0)

    total_score += score

    writer.writerow([
        question_id,
        image_id,
        question,
        answers[0],
        pred,
        match_count,
        round(score, 3),
        round(infer_time, 3)
    ])

    print(
        f"[{i+1}/{NUM_SAMPLES}] "
        f"score={score:.3f} "
        f"match={match_count} "
        f"pred={pred} "
        f"time={infer_time:.2f}s"
    )

    print("-" * 50)

    del sample, image
    gc.collect()


elapsed = time.time() - start_time

print("=" * 60)
print("Finished")
print("=" * 60)

print("Samples:", NUM_SAMPLES)
print("TextVQA Accuracy:", round(total_score / NUM_SAMPLES, 4))
print("Total Time:", round(elapsed, 2), "s")
print("Avg Time:", round(elapsed / NUM_SAMPLES, 2), "s/sample")

csv_file.close()

print("Saved:", RESULT_FILE)