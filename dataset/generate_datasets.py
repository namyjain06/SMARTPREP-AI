"""
SmartPrep AI - Dataset Generation & Loading
Generates programmatic datasets for all three ML tasks.
"""

import pandas as pd
import numpy as np
import os
import random

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# 1. DIFFICULTY PREDICTION DATASET
# ─────────────────────────────────────────────

QUESTIONS = {
    "Mathematics": {
        "Easy": [
            "What is 2 + 2?",
            "Solve: x + 3 = 7",
            "What is the value of pi to 2 decimal places?",
            "Find the area of a square with side 4.",
            "What is 15% of 200?",
            "Simplify: 3x + 2x",
            "What is the square root of 144?",
            "Convert 0.5 to a fraction.",
            "What is the perimeter of a rectangle 3x5?",
            "Evaluate: 3^2"
        ],
        "Medium": [
            "Solve the quadratic equation x^2 - 5x + 6 = 0.",
            "Find the derivative of f(x) = 3x^2 + 2x.",
            "Calculate the determinant of a 2x2 matrix.",
            "Prove that the sum of angles in a triangle is 180 degrees.",
            "Integrate f(x) = 2x + 1 with respect to x.",
            "Find the inverse of a 2x2 matrix.",
            "Solve the system: 2x + y = 5, x - y = 1.",
            "What is the binomial expansion of (a+b)^3?",
            "Find the limit of (x^2-1)/(x-1) as x→1.",
            "Calculate the standard deviation of [2,4,4,4,5,5,7,9]."
        ],
        "Hard": [
            "Prove the Fundamental Theorem of Calculus.",
            "Solve the differential equation dy/dx = y*sin(x).",
            "Find eigenvalues of a 3x3 matrix.",
            "Evaluate the contour integral of 1/(z^2+1) over |z|=2.",
            "Prove Cauchy-Schwarz inequality in inner product spaces.",
            "Solve the partial differential equation uxx + uyy = 0.",
            "Derive the formula for Fourier series coefficients.",
            "Prove that sqrt(2) is irrational using contradiction.",
            "Apply Lagrange multipliers to optimize f(x,y) = x^2+y^2 subject to x+y=1.",
            "Prove that every finite integral domain is a field."
        ]
    },
    "Physics": {
        "Easy": [
            "What is Newton's First Law?",
            "Define velocity.",
            "What is the SI unit of force?",
            "State Ohm's Law.",
            "What is the speed of light in vacuum?",
            "Define kinetic energy.",
            "What is potential energy?",
            "State Archimedes' principle.",
            "What is the unit of electric current?",
            "Define frequency of a wave."
        ],
        "Medium": [
            "Derive the equation for projectile motion.",
            "Explain the working of a transformer.",
            "What is the Doppler effect? Give an example.",
            "Derive the expression for kinetic energy from work-energy theorem.",
            "Explain Snell's Law and derive it.",
            "What is simple harmonic motion? Write its differential equation.",
            "Explain the photoelectric effect.",
            "Derive expressions for electric field due to a point charge.",
            "Explain the concept of magnetic flux and Faraday's law.",
            "What is resonance in an LC circuit?"
        ],
        "Hard": [
            "Derive Maxwell's equations from first principles.",
            "Explain quantum tunneling and its applications.",
            "Derive the Schrodinger equation and explain its significance.",
            "Explain special theory of relativity and time dilation.",
            "Derive the expression for entropy in statistical mechanics.",
            "Explain the Heisenberg uncertainty principle mathematically.",
            "What is the Higgs mechanism? Explain spontaneous symmetry breaking.",
            "Derive the Navier-Stokes equations.",
            "Explain quantum entanglement and Bell's theorem.",
            "Describe the Standard Model of particle physics."
        ]
    },
    "Computer Science": {
        "Easy": [
            "What is an algorithm?",
            "Define a variable in programming.",
            "What is a loop? Name two types.",
            "What does CPU stand for?",
            "What is an array?",
            "Define a function in programming.",
            "What is a boolean data type?",
            "What is an if-else statement?",
            "Define RAM and ROM.",
            "What is a compiler?"
        ],
        "Medium": [
            "Explain the time complexity of bubble sort.",
            "What is a binary search tree? Explain insertion.",
            "Explain the concept of recursion with an example.",
            "What is dynamic programming? Explain with Fibonacci.",
            "Explain the difference between stack and queue.",
            "What is hashing? Explain collision resolution techniques.",
            "Explain BFS and DFS graph traversal algorithms.",
            "What is a deadlock? Explain necessary conditions.",
            "Explain the OSI model with all 7 layers.",
            "What is normalization in databases? Explain 1NF, 2NF, 3NF."
        ],
        "Hard": [
            "Prove the P vs NP problem statement and its implications.",
            "Explain and implement Dijkstra's algorithm with complexity analysis.",
            "Derive the time and space complexity of merge sort using recurrence.",
            "Explain the CAP theorem in distributed systems.",
            "Design a distributed key-value store with consistency guarantees.",
            "Explain Byzantine fault tolerance in consensus algorithms.",
            "Derive the complexity of matrix chain multiplication using DP.",
            "Explain the internals of a B+ tree and its use in databases.",
            "Describe the implementation of a garbage collector.",
            "Explain RAFT consensus algorithm for distributed systems."
        ]
    },
    "Chemistry": {
        "Easy": [
            "What is an atom?",
            "Define pH scale.",
            "What is a covalent bond?",
            "State the law of conservation of mass.",
            "What is the atomic number of Carbon?",
            "Define molar mass.",
            "What is oxidation?",
            "Name the three states of matter.",
            "What is a chemical equation?",
            "Define electronegativity."
        ],
        "Medium": [
            "Explain hybridization in carbon compounds.",
            "What is the ideal gas law? Derive it.",
            "Explain Le Chatelier's principle with examples.",
            "What is electrochemistry? Explain galvanic cells.",
            "Describe the mechanism of SN1 and SN2 reactions.",
            "Explain the concept of chemical equilibrium.",
            "What is titration? Explain acid-base titration.",
            "Derive the Henderson-Hasselbalch equation.",
            "Explain Hess's Law of heat summation.",
            "What are isomers? Explain structural isomerism."
        ],
        "Hard": [
            "Explain molecular orbital theory for benzene.",
            "Derive the rate equation for a complex reaction mechanism.",
            "Explain the quantum mechanical model of the hydrogen atom.",
            "Describe the mechanism of Diels-Alder reaction.",
            "Explain the Born-Haber cycle for ionic compounds.",
            "Describe transition state theory and Arrhenius equation derivation.",
            "Explain the synthesis of complex organic compounds using retrosynthesis.",
            "What is the Crystal Field Theory? Explain for octahedral complexes.",
            "Derive the Nernst equation and explain its applications.",
            "Explain the thermodynamic basis of spontaneity using Gibbs free energy."
        ]
    },
    "Biology": {
        "Easy": [
            "What is a cell?",
            "Define photosynthesis.",
            "What is DNA?",
            "Name the four blood groups.",
            "What is mitosis?",
            "Define respiration.",
            "What is an ecosystem?",
            "Name the organelles in a cell.",
            "What is the function of the heart?",
            "Define homeostasis."
        ],
        "Medium": [
            "Explain the process of protein synthesis.",
            "Describe the stages of mitosis in detail.",
            "Explain Mendel's laws of inheritance.",
            "What is the lac operon model?",
            "Explain the mechanism of enzyme action.",
            "Describe the structure and function of DNA.",
            "Explain the immune response to pathogens.",
            "What is the significance of the citric acid cycle?",
            "Describe the process of meiosis.",
            "Explain the concept of natural selection."
        ],
        "Hard": [
            "Explain the molecular basis of cancer development.",
            "Describe the CRISPR-Cas9 gene editing mechanism.",
            "Explain epigenetic regulation of gene expression.",
            "Describe the mechanism of signal transduction pathways.",
            "Explain the evolutionary theory of endosymbiosis.",
            "Describe the molecular mechanism of apoptosis.",
            "Explain the role of non-coding RNA in gene regulation.",
            "Describe the mechanism of viral replication in host cells.",
            "Explain the molecular basis of antibiotic resistance.",
            "Describe the structure and function of the ribosome in detail."
        ]
    }
}

def generate_difficulty_dataset(output_path="dataset/difficulty_dataset.csv"):
    rows = []
    label_map = {"Easy": 0, "Medium": 1, "Hard": 2}
    
    for subject, levels in QUESTIONS.items():
        for difficulty, questions in levels.items():
            for q in questions:
                rows.append({
                    "question": q,
                    "subject": subject,
                    "difficulty": difficulty,
                    "difficulty_label": label_map[difficulty],
                    "word_count": len(q.split()),
                    "char_count": len(q)
                })
    
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[✓] Difficulty dataset saved: {len(df)} records → {output_path}")
    return df


# ─────────────────────────────────────────────
# 2. STUDENT PERFORMANCE DATASET (WEAK TOPIC ANALYSIS)
# ─────────────────────────────────────────────

def generate_student_performance_dataset(n=300, output_path="dataset/student_performance.csv"):
    subjects = ["Mathematics", "Physics", "Computer Science", "Chemistry", "Biology"]
    
    rows = []
    for i in range(n):
        student_type = np.random.choice(["weak_math", "weak_science", "weak_cs", "strong", "average"], 
                                         p=[0.15, 0.20, 0.20, 0.25, 0.20])
        base = {
            "weak_math":    [30, 65, 70, 62, 68],
            "weak_science": [72, 35, 68, 38, 40],
            "weak_cs":      [60, 62, 32, 65, 63],
            "strong":       [80, 82, 85, 78, 83],
            "average":      [55, 58, 52, 56, 60],
        }[student_type]
        
        scores = [max(0, min(100, b + np.random.randint(-10, 11))) for b in base]
        row = {
            "student_id": f"S{i+1:04d}",
            "study_hours_per_day": round(np.random.uniform(1, 8), 1),
            "attendance_pct": round(np.random.uniform(50, 100), 1),
        }
        for subj, score in zip(subjects, scores):
            row[subj] = score
        row["avg_score"] = round(np.mean(scores), 2)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[✓] Student performance dataset saved: {len(df)} records → {output_path}")
    return df


# ─────────────────────────────────────────────
# 3. ANSWER EVALUATION DATASET (ASAP-style)
# ─────────────────────────────────────────────

ESSAY_DATA = [
    {
        "prompt": "Describe the water cycle.",
        "reference": "The water cycle describes how water evaporates from the surface of the earth, rises into the atmosphere, cools and condenses into rain or snow in clouds, and falls again to the surface as precipitation. The cycle involves evaporation, condensation, precipitation, and collection.",
        "student_answers": [
            ("Water evaporates from oceans, condenses to form clouds, and falls as rain or snow, then collects in water bodies and evaporates again.", 4),
            ("Water goes up and comes down as rain.", 1),
            ("The water cycle includes evaporation where water becomes vapor, condensation forming clouds, precipitation as rain or snow, and collection in rivers and lakes.", 5),
            ("Evaporation makes water vapor. Clouds form and it rains.", 2),
            ("Water evaporates due to solar energy, rises, cools to condense into clouds, precipitates as rain or snow, and collects in oceans and rivers starting the cycle again.", 5),
            ("Rain falls on land.", 0),
            ("The water cycle involves water changing states from liquid to gas through evaporation, forming clouds via condensation, and returning as precipitation.", 4),
            ("Hot sun makes water disappear into sky.", 1),
        ]
    },
    {
        "prompt": "Explain Newton's Second Law of Motion.",
        "reference": "Newton's Second Law of Motion states that the acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass. This is expressed as F = ma, where F is force, m is mass, and a is acceleration.",
        "student_answers": [
            ("F equals ma. Force equals mass times acceleration. More force means more acceleration if mass is same.", 3),
            ("Newton said things move when you push them.", 1),
            ("The second law states acceleration is proportional to net force and inversely proportional to mass, F=ma.", 5),
            ("Objects accelerate based on how hard you push.", 2),
            ("Force equals mass multiplied by acceleration. Larger force causes greater acceleration for the same mass.", 4),
            ("Newton made F=ma which means force is mass times acceleration showing relationship between force and motion.", 4),
            ("Things move.", 0),
            ("Second law: net force on body equals its mass times acceleration it acquires, F=ma, direction of acceleration same as force.", 5),
        ]
    },
    {
        "prompt": "What is photosynthesis?",
        "reference": "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy stored in glucose. The process uses carbon dioxide and water, releasing oxygen as a byproduct. The equation is 6CO2 + 6H2O + light → C6H12O6 + 6O2.",
        "student_answers": [
            ("Plants make food from sunlight, water, and CO2, releasing oxygen.", 3),
            ("Photosynthesis converts light energy to chemical energy in glucose using CO2 and water, releasing O2.", 5),
            ("Plants eat sunlight.", 0),
            ("Green plants use chlorophyll to absorb sunlight, convert CO2 and water into glucose and oxygen.", 4),
            ("It is how plants make food.", 1),
            ("Photosynthesis: 6CO2 + 6H2O + light produces glucose C6H12O6 and 6O2, occurs in chloroplasts.", 5),
            ("Plants absorb light and produce sugar.", 2),
            ("The process where plants use sunlight energy to combine carbon dioxide and water, forming glucose and releasing oxygen as byproduct.", 4),
        ]
    },
    {
        "prompt": "Describe the process of DNA replication.",
        "reference": "DNA replication is the process by which a DNA molecule makes a copy of itself. The double helix unwinds and each strand serves as a template for a new complementary strand. Key enzymes include helicase (unwinds DNA), primase (adds RNA primer), DNA polymerase (synthesizes new strand), and ligase (joins fragments).",
        "student_answers": [
            ("DNA copies itself by splitting and each half becoming a full copy.", 2),
            ("DNA replication uses helicase to unwind the double helix, then DNA polymerase synthesizes new complementary strands.", 4),
            ("DNA makes copies.", 0),
            ("During replication, helicase unwinds DNA, primase adds primers, polymerase adds nucleotides, and ligase joins Okazaki fragments.", 5),
            ("The double helix splits and each strand is used as template to form new complementary strand.", 3),
            ("DNA duplicates using enzymes like helicase and polymerase to create two identical daughter molecules.", 4),
            ("Cells copy DNA when dividing.", 1),
            ("Semi-conservative replication: helicase unwinds DNA, polymerase synthesizes complementary strand in 5' to 3' direction.", 5),
        ]
    },
    {
        "prompt": "Explain the concept of recursion in programming.",
        "reference": "Recursion is a programming technique where a function calls itself to solve a problem by breaking it into smaller subproblems of the same type. Every recursive function needs a base case (stopping condition) and a recursive case. Example: factorial(n) = n * factorial(n-1) with base case factorial(0) = 1.",
        "student_answers": [
            ("Recursion is when a function calls itself.", 2),
            ("Recursion: a function that calls itself with a smaller input until reaching the base case.", 4),
            ("A function calling itself repeatedly.", 1),
            ("Recursive functions call themselves with modified parameters, need base case to stop, used for tree traversal, factorial, fibonacci.", 5),
            ("Functions can call themselves to solve problems like factorial where n! = n * (n-1)!.", 3),
            ("Recursion solves problems by dividing into subproblems using self-referential function calls with base cases to prevent infinite loops.", 5),
            ("A loop that calls functions.", 0),
            ("Function that calls itself with base case like fibonacci(n) = fibonacci(n-1) + fibonacci(n-2).", 4),
        ]
    },
]

def generate_answer_evaluation_dataset(output_path="dataset/answer_evaluation.csv"):
    rows = []
    for idx, item in enumerate(ESSAY_DATA):
        for student_ans, score in item["student_answers"]:
            rows.append({
                "essay_id": idx + 1,
                "prompt": item["prompt"],
                "reference_answer": item["reference"],
                "student_answer": student_ans,
                "score": score,
                "max_score": 5
            })
    
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[✓] Answer evaluation dataset saved: {len(df)} records → {output_path}")
    return df


# ─────────────────────────────────────────────
# STUDY TOPICS (for planner)
# ─────────────────────────────────────────────

TOPIC_TREE = {
    "Mathematics": ["Algebra", "Calculus", "Linear Algebra", "Probability & Statistics", "Discrete Mathematics", "Differential Equations"],
    "Physics": ["Mechanics", "Thermodynamics", "Electromagnetism", "Optics", "Modern Physics", "Waves & Sound"],
    "Computer Science": ["Data Structures", "Algorithms", "Operating Systems", "DBMS", "Computer Networks", "OOP"],
    "Chemistry": ["Organic Chemistry", "Inorganic Chemistry", "Physical Chemistry", "Electrochemistry", "Thermochemistry", "Spectroscopy"],
    "Biology": ["Cell Biology", "Genetics", "Ecology", "Human Physiology", "Molecular Biology", "Evolution"]
}

if __name__ == "__main__":
    print("=== SmartPrep AI - Dataset Generator ===")
    generate_difficulty_dataset()
    generate_student_performance_dataset()
    generate_answer_evaluation_dataset()
    print("\n[✓] All datasets generated successfully.")
