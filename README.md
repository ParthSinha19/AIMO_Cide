**AI Engine for Solving IMO-Level Problems**

This project is an AI engine designed to solve **International Mathematical Olympiad (IMO)**-level problems. It combines **symbolic reasoning**, **neural-guided search**, and **verification** to tackle complex mathematical problems in domains like **number theory**, **geometry**, and **algebra**. The engine is modular, extensible, and designed to mimic human problem-solving strategies.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [How It Works](#how-it-works)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Examples](#examples)
8. [Future Work](#future-work)
9. [Contributing](#contributing)
10. [License](#license)

---

## **Project Overview**
The goal of this project is to build an AI engine capable of solving IMO-level problems by:
1. **Parsing** natural language problem statements into structured representations.
2. **Reasoning** using domain-specific modules (e.g., number theory, geometry).
3. **Verifying** solutions using automated theorem provers (ATPs) and symbolic libraries.

The engine is designed to be **domain-aware**, meaning it can apply the right tools and strategies for each problem type. It uses a combination of **neural-guided search** (to explore possible strategies) and **symbolic reasoning** (to apply mathematical rules and theorems).

---

## **Key Features**
- **Problem Parsing & Representation**:
  - Converts natural language problem statements into structured representations (e.g., JSON).
  - Extracts entities, relationships, and goals from the problem.

- **Domain-Specific Modules**:
  - **Number Theory & Combinatorics**: Solves problems involving LCM/GCD, modular arithmetic, and the Pigeonhole Principle.
  - **Geometry**: Applies theorems like Power of a Point, Ceva, and Menelaus.
  - **Algebra & Optimization**: Solves equations, maximizes/minimizes expressions, and handles inequalities.

- **Search & Reasoning Framework**:
  - Uses **Monte Carlo Tree Search (MCTS)** and **Reinforcement Learning (RL)** to explore proof paths.
  - Applies **rule-based deduction** to derive new information from constraints.

- **Verification & Correction**:
  - Validates each step of the reasoning process using **Automated Theorem Provers (ATPs)** like Lean4 and Coq.
  - Detects and corrects errors in the engine’s outputs.

- **Training & Data Pipeline**:
  - Generates synthetic problems for training.
  - Fine-tunes **GPT-Neo 1.3B** on annotated IMO problems.

---

## **Project Structure**
```
IMO-AI-Engine/
├── README.md
├── requirements.txt
├── src/
│   ├── number_theory.py       # Number theory and combinatorics solver
│   ├── geometry.py            # Geometry solver
│   ├── algebra.py             # Algebra and optimization solver
│   ├── search_reasoning.py    # Search and reasoning framework
│   ├── verification.py        # Verification and correction module
│   ├── data_pipeline.py       # Data generation and annotation
│   └── main.py                # Main script to run the engine
├── data/
│   ├── imo_problems/          # Past IMO problems
│   ├── synthetic_problems/    # Generated synthetic problems
│   └── annotations/           # Annotated structured representations
└── tests/                     # Unit tests for each module
```

---

## **How It Works**
1. **Problem Parsing**:
   - The engine uses a fine-tuned **GPT-Neo 1.3B** model to parse natural language problem statements into structured representations (e.g., JSON).

2. **Domain-Specific Reasoning**:
   - The problem is passed to the appropriate domain-specific module (e.g., number theory, geometry).
   - The module applies relevant algorithms and theorems to solve the problem.

3. **Search & Reasoning**:
   - The engine explores possible proof paths using **MCTS** and **RL**.
   - It applies **rule-based deduction** to derive new information from constraints.

4. **Verification**:
   - Each step of the reasoning process is verified using **ATPs** (e.g., Lean4, Coq).
   - Errors are detected and corrected using a feedback loop.

5. **Output**:
   - The engine outputs the solution in a structured format (e.g., equations, proofs).

---

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/IMO-AI-Engine.git
   cd IMO-AI-Engine
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download pre-trained models (if applicable):
   ```bash
   python src/data_pipeline.py --download-models
   ```

---

## **Usage**
1. Run the engine on a problem:
   ```bash
   python src/main.py --problem "Three airline companies depart every 100, 120, and 150 days. Find the greatest d such that there are d consecutive days without a flight."
   ```

2. Train the engine on new data:
   ```bash
   python src/data_pipeline.py --generate-synthetic --annotate
   python src/main.py --train
   ```

3. Test the engine:
   ```bash
   python -m pytest tests/
   ```

---

## **Examples**
### **Number Theory**
```python
nt_solver = NumberTheoryCombinatoricsSolver()
print("LCM of [100, 120, 150]:", nt_solver.lcm([100, 120, 150]))
# Output: 600
```

### **Geometry**
```python
geo_solver = GeometrySolver()
A = (0, 0)
B = (108, 0)
C = (0, 126)
print("Barycentric Coords:", geo_solver.barycentric_coords((39, 0), (A, B, C)))
# Output: [0.361, 0.0, 0.639]
```

### **Algebra**
```python
alg_solver = AlgebraOptimizationSolver()
x, y, z = symbols('x y z')
print("Solved System:", alg_solver.solve_system([x + y - 5, y - z - 1]))
# Output: {x: -z + 6, y: z + 1}
```

---

## **Future Work**
- **Expand Domains**: Add support for combinatorics, inequalities, and more.
- **Improve Creativity**: Incorporate more human-like insights into the reasoning process.
- **Real-World Applications**: Use the engine for educational tools and training.

---

## **Contributing**
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

**License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
This `README.md` provides a clear and concise overview of your project. Let me know if you need further assistance! 🚀# AIMO_Cide
