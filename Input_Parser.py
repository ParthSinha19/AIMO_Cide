from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
import re
import json
from datasets import Dataset

class MathProblemSolver:
    def __init__(self, model_name="EleutherAI/gpt-neo-1.3B"):
        self.model_name = model_name
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPTNeoForCausalLM.from_pretrained(self.model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False

    def parse_problem(self, problem_text):
        """Extract key constraints and components from the problem"""
        if not self.model_loaded:
            return "default", {"error": "Model not loaded properly"}
        
        try:
            # Extract category
            category_match = re.search(r'\[(.*?)\]', problem_text)
            category = category_match.group(1) if category_match else "unknown"
            
            # Extract numerical values
            numbers = re.findall(r'\d+(?:\.\d+)?', problem_text)
            numerical_values = [float(n) for n in numbers]
            
            # Extract variables
            variables = re.findall(r'(?<![a-zA-Z])[a-zA-Z](?![a-zA-Z])', problem_text)
            variables = list(set(variables))
            
            # Extract relationships
            relationships = []
            if "=" in problem_text:
                relationships.extend(re.findall(r'[^=]+=\s*[^=]+', problem_text))
            if ">" in problem_text or "<" in problem_text:
                relationships.extend(re.findall(r'[^<>]+[<>]\s*[^<>]+', problem_text))
            
            # Extract constraints
            constraints = []
            constraint_keywords = ["must", "should", "only if", "if and only if", "such that", "where", "given that"]
            for keyword in constraint_keywords:
                if keyword in problem_text.lower():
                    constraint_match = re.search(f"{keyword}(.*?)(?:\.|$)", problem_text, re.IGNORECASE)
                    if constraint_match:
                        constraints.append(constraint_match.group(1).strip())
            
            # Extract goal
            goal = ""
            goal_keywords = ["find", "determine", "calculate", "express", "compute"]
            for keyword in goal_keywords:
                if keyword in problem_text.lower():
                    goal_match = re.search(f"{keyword}(.*?)(?:\.|$)", problem_text, re.IGNORECASE)
                    if goal_match:
                        goal = goal_match.group(1).strip()
                        break
            
            return category.lower(), {
                'numerical_values': numerical_values,
                'variables': variables,
                'relationships': relationships,
                'constraints': constraints,
                'goal': goal
            }
            
        except Exception as e:
            print(f"Error parsing problem: {e}")
            return "error", {"message": str(e)}

    def parse_and_store_problems(self):
        """Parse and store problem constraints from file"""
        try:
            with open('IMO_High_Level_Problems.txt', 'r') as file:
                problems = file.read().split('\n')
            
            parsed_results = []
            for problem in problems:
                if problem.strip():
                    category, constraints = self.parse_problem(problem)
                    parsed_results.append({
                        'original_problem': problem,
                        'category': category,
                        'constraints': constraints
                    })
            
            with open('problem_constraints.json', 'w') as outfile:
                json.dump(parsed_results, outfile, indent=4)
            
            return parsed_results
            
        except Exception as e:
            print(f"Error parsing and storing problems: {e}")
            return []

def test_parser():
    parser = MathProblemSolver()
    
    test_problems = [
        """[Geometry] A circle of radius 11 is inscribed in a right triangle with legs a and b. 
        If the hypotenuse of the triangle is c, express the area of the triangle in terms of r.""",
        
        """[Combinatorics] In how many ways can 3 students be selected from a group of 5 students 
        if order does not matter?""",
        
        """[Number Theory] Find the sum of the digits of the largest three-digit number that is 
        divisible by 7."""
    ]
    
    for problem in test_problems:
        print("\nProblem:", problem)
        category, constraints = parser.parse_problem(problem)
        print("Category:", category)
        print("Constraints:", json.dumps(constraints, indent=2))

if __name__ == "__main__":
    test_parser()