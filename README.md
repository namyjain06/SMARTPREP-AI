# SmartPrep AI

A dataset generation toolkit for educational machine learning tasks. This script programmatically creates three training datasets covering difficulty classification, student performance prediction, and automated answer evaluation.

---

## Project Structure

```
smartprep-ai/
└── dataset/
    └── generate_datasets.py
```

---

## What It Does

Running the script generates three CSV files inside the `dataset/` folder:

**difficulty_dataset.csv**
Contains 150 exam questions across 5 subjects (Mathematics, Physics, Computer Science, Chemistry, Biology), each labeled Easy, Medium, or Hard. Includes word count and character count as features. Used for multi-class difficulty classification.

**student_performance.csv**
Simulates 300 student records with subject-wise scores across all 5 subjects, along with study hours per day and attendance percentage. Scores are generated based on student type profiles (weak in math, weak in science, strong, average, etc.). Used for performance prediction or clustering.

**answer_evaluation.csv**
Contains 40 student answer samples across 5 essay prompts with reference answers and human-assigned scores (0-5). Used for automated short-answer grading, similar to the ASAP dataset format.

---

## Requirements

- Python 3.7 or above
- pandas
- numpy

Install dependencies:

```bash
pip install pandas numpy
```

---

## Usage

```bash
cd dataset
python generate_datasets.py
```

All three CSV files will be saved inside the `dataset/` folder automatically.

---

## Output Files

| File | Records | Task |
|---|---|---|
| difficulty_dataset.csv | 150 | Difficulty classification |
| student_performance.csv | 300 | Performance prediction |
| answer_evaluation.csv | 40 | Answer scoring |

---

## Subjects Covered

Mathematics, Physics, Computer Science, Chemistry, Biology

---

## Notes

- Random seeds are fixed (seed 42) so outputs are reproducible across runs.
- The student performance dataset uses randomized noise on base scores to simulate realistic variation.
- The answer evaluation dataset is hand-curated with reference answers and gold-standard scores.
