import random
import math
import re
import json
import logging
from collections import defaultdict
from math import sqrt, log
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("math_problem_solver")

def load_dataset(file_path: str) -> List[Dict[str, str]]:
    problems = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            problem_texts = re.findall(r'\[(.*?)\](.*?)(?=\[|$)', content, re.DOTALL)
            for problem_type, problem_content in problem_texts:
                parts = problem_content.split('Solution:', 1)
                if len(parts) == 2:
                    problems.append({'type': problem_type.strip(), 'problem': parts[0].strip(), 'solution': parts[1].strip()})
        logger.info(f"Successfully loaded {len(problems)} problems from {file_path}")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
    return problems

def split_dataset(problems: List[Dict[str, str]], test_size=0.2):
    return train_test_split(problems, test_size=test_size, random_state=42)

class FeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.fitted = False

    def fit_transform(self, problem_texts: List[str]):
        self.fitted = True
        return self.vectorizer.fit_transform(problem_texts)

    def transform(self, problem_texts: List[str]):
        if not self.fitted:
            logger.warning("Feature extractor used before fitting")
            return None
        return self.vectorizer.transform(problem_texts)

class ProblemClassifier:
    def __init__(self, training_data: List[Dict[str, str]] = None):
        self.problem_types = {
            'Geometry': r'(circle|triangle|radius|area|angle|polygon)',
            'Algebra': r'(polynomial|quadratic|equation|function|expression)',
            'Number Theory': r'(divisible|digits|sum|multiple|prime|modulo)',
            'Combinatorics': r'(ways|selected|order|permutation|combination)'
        }
        self.feature_extractor = FeatureExtractor()
        self.ml_classifier = MultinomialNB()
        self.ml_trained = False
        if training_data:
            self.learn_from_training_data(training_data)

    def learn_from_training_data(self, training_data: List[Dict[str, str]]):
        problem_texts = [item['problem'] for item in training_data]
        problem_labels = [item['type'] for item in training_data]
        features = self.feature_extractor.fit_transform(problem_texts)
        self.ml_classifier.fit(features, problem_labels)
        self.ml_trained = True

    def classify_problem(self, problem_text: str) -> str:
        if self.ml_trained:
            features = self.feature_extractor.transform([problem_text])
            return self.ml_classifier.predict(features)[0]
        for problem_type, pattern in self.problem_types.items():
            if re.search(pattern, problem_text, re.IGNORECASE):
                return problem_type
        return "Unknown"

class TransformerPlanner:
    def __init__(self, classifier=None):
        self.classifier = classifier or ProblemClassifier()

    def generate_strategies(self, problem: str) -> List[str]:
        problem_type = self.classifier.classify_problem(problem)
        strategies = {
            'Geometry': ['Use area = rs', 'Apply coordinate geometry'],
            'Algebra': ['Solve using quadratic formula', 'Use function transformations'],
            'Number Theory': ['Check divisibility rules', 'Apply modular arithmetic'],
            'Combinatorics': ['Use binomial theorem', 'Apply counting principles']
        }
        return strategies.get(problem_type, ["Try breaking the problem into smaller parts"])

class MCTSNode:
    def __init__(self, state: str, problem_type: str = None, parent=None):
        self.state = state
        self.problem_type = problem_type
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def select(self) -> 'MCTSNode':
        if not self.children:
            return self
        return max(self.children, key=lambda c: c.value / (c.visits + 1e-5) + sqrt(2 * log(self.visits + 1) / (c.visits + 1e-5)))

    def expand(self, planner: TransformerPlanner):
        strategies = planner.generate_strategies(self.state)
        for strategy in strategies:
            new_state = f"{self.state}\nApplied: {strategy}"
            child = MCTSNode(new_state, self.problem_type, self)    
            self.children.append(child)

    def simulate(self) -> float:
        return random.uniform(0.5, 0.9)

    def backpropagate(self, reward: float):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

class MCTS:
    def __init__(self, initial_state: str, classifier=None):
        self.planner = TransformerPlanner(classifier)
        problem_type = self.planner.classifier.classify_problem(initial_state)
        self.root = MCTSNode(initial_state, problem_type)

    def search(self, iterations: int = 50) -> MCTSNode:
        for _ in range(iterations):
            node = self.root.select()
            node.expand(self.planner)
            reward = node.simulate()
            node.backpropagate(reward)
        return max(self.root.children, key=lambda c: c.visits) if self.root.children else self.root

def solve_problem(problem_text: str, classifier: ProblemClassifier) -> str:
    mcts = MCTS(problem_text, classifier)
    best_node = mcts.search()
    return '\n'.join(best_node.state.split('\n')[1:])

if __name__ == "__main__":
    dataset_path = "IMO_High_Level_Problems.txt"
    dataset = load_dataset(dataset_path)
    train_set, test_set = split_dataset(dataset)
    classifier = ProblemClassifier(train_set)
    for i in range (1,10000,100):
      problem = random.choice(test_set)['problem']
      print(f"Solving Problem: {problem}\n")
      print("Solution Steps:")
      print(solve_problem(problem, classifier))

