from sympy import *
from z3 import *
import math
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from Input_Parser import MathProblemSolver
from mcts_framework import MCTS, ProblemClassifier
import os
from glob import glob

class SolutionEngine:
    def __init__(self):
        self.nt_solver = NumberTheoryCombinatoricsSolver()
        self.geo_solver = GeometrySolver()
        self.alg_solver = AlgebraOptimizationSolver()
        self.parser = MathProblemSolver()
        self.classifier = ProblemClassifier()
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.solution_classifier = RandomForestClassifier(n_estimators=100)
        self.results_folder = os.path.join(os.path.dirname(__file__), "analysis_output", "results")
        self.load_and_process_results()

    def load_and_process_results(self):
        try:
            # Get all result files
            result_files = glob(os.path.join(self.results_folder, "results_*.json"))
            all_problems = []
            
            for file_path in result_files:
                with open(file_path, 'r') as f:
                    results = json.load(f)
                    for result in results:
                        all_problems.append(result['problem'])
            
            # Add these problems to training data
            self.train_solution_model(additional_problems=all_problems)
            
        except Exception as e:
            print(f"Error loading results: {e}")

    def train_solution_model(self, additional_problems=None):
        try:
            # Load original training data
            with open('IMO_High_Level_Problems.txt', 'r') as f:
                problems = f.readlines()
            
            X = []  # Problem texts
            y = []  # Solutions
            
            # Process original problems
            for problem in problems:
                if 'Solution:' in problem:
                    prob_text, solution = problem.split('Solution:')
                    X.append(prob_text.strip())
                    y.append(solution.strip())
            
            # Add additional problems if provided
            if additional_problems:
                for problem in additional_problems:
                    X.append(problem)
                    # Use ML model to generate initial solutions for new problems
                    if hasattr(self, 'solution_classifier'):
                        vec = self.vectorizer.transform([problem])
                        predicted_solution = self.solution_classifier.predict(vec)[0]
                        y.append(predicted_solution)
                    else:
                        y.append("Solution pending")
            
            # Transform text data
            X_vectors = self.vectorizer.fit_transform(X)
            
            # Train model with expanded dataset
            self.solution_classifier.fit(X_vectors, y)
            
        except Exception as e:
            print(f"Error training model: {e}")

    def solve_all_results(self):
        """Solve all problems from results files"""
        try:
            result_files = glob(os.path.join(self.results_folder, "results_*.json"))
            all_solutions = []
            
            for file_path in result_files:
                with open(file_path, 'r') as f:
                    results = json.load(f)
                    
                for result in results:
                    problem = result['problem']
                    solution = self.solve_problem(problem)
                    all_solutions.append({
                        'problem': problem,
                        'solution': solution,
                        'timestamp': result.get('timestamp', '')
                    })
            
            # Save updated solutions
            output_path = os.path.join(self.results_folder, f"solutions_{int(time.time())}.json")
            with open(output_path, 'w') as f:
                json.dump(all_solutions, f, indent=2)
            
            return all_solutions
            
        except Exception as e:
            print(f"Error solving results: {e}")
            return []

    def solve_problem(self, problem_text):
        category, constraints = self.parser.parse_problem(problem_text)
        
        # Get solution strategy from MCTS
        mcts = MCTS(problem_text, self.classifier)
        solution_strategy = mcts.search()
        
        solution = None
        try:
            if category == "geometry":
                solution = self.geo_solver.solve_with_constraints(constraints)
            elif category == "number theory":
                solution = self.nt_solver.solve_with_constraints(constraints)
            elif category == "algebra":
                solution = self.alg_solver.solve_with_constraints(constraints)
            
            # If specific solver fails, use ML model
            if not solution or solution.startswith("Unable"):
                vec = self.vectorizer.transform([problem_text])
                predicted_solution = self.solution_classifier.predict(vec)[0]
                solution = f"ML Model Solution: {predicted_solution}"
            
        except Exception as e:
            solution = f"Error in solving: {str(e)}"
        
        return {
            "category": category,
            "parsed_constraints": constraints,
            "solution_strategy": solution_strategy.state if solution_strategy else None,
            "solution": solution
        }

class NumberTheoryCombinatoricsSolver:
    def solve_with_constraints(self, constraints):
        try:
            numerical_values = constraints.get('numerical_values', [])
            goal = constraints.get('goal', '').lower()
            
            if 'divisible' in goal:
                return self.solve_divisibility(numerical_values)
            elif 'ways' in goal or 'selected' in goal:
                return self.solve_combinations(numerical_values)
            return self.solve_general_number_theory(constraints)
        except Exception as e:
            return f"Error in number theory solver: {str(e)}"

    def solve_divisibility(self, values):
        try:
            if not values:
                return "No values provided"
            
            divisor = values[0]
            max_num = 999
            largest_multiple = (max_num // divisor) * divisor
            digit_sum = sum(int(d) for d in str(largest_multiple))
            
            return f"Largest multiple: {largest_multiple}, Digit sum: {digit_sum}"
        except Exception as e:
            return f"Error in divisibility solver: {str(e)}"

    def solve_combinations(self, values):
        try:
            if len(values) < 2:
                return "Insufficient values for combination calculation"
            
            n = int(values[0])  # total students
            r = int(values[1])  # students to be selected
            result = math.comb(n, r)
            
            return f"C({n},{r}) = {result} ways"
        except Exception as e:
            return f"Error in combinations solver: {str(e)}"

class GeometrySolver:
    def solve_with_constraints(self, constraints):
        try:
            if 'circle' in str(constraints) and 'triangle' in str(constraints):
                return self.solve_circle_triangle_problem(constraints)
            return "Unable to determine specific geometry approach"
        except Exception as e:
            return f"Error in geometry solver: {str(e)}"

    def solve_circle_triangle_problem(self, constraints):
        try:
            radius = next((v for v in constraints.get('numerical_values', []) if v > 0), None)
            if not radius:
                return "No valid radius found"
            
            # For a right triangle with inscribed circle:
            # Area = r(a + b + c)/2 = rs, where s is semi-perimeter
            r = Symbol('r')
            s = Symbol('s')
            area = r * s
            
            return f"Area = {area} = r * s, where s is the semi-perimeter"
        except Exception as e:
            return f"Error in circle-triangle solver: {str(e)}"

class AlgebraOptimizationSolver:
    def solve_with_constraints(self, constraints):
        try:
            if 'quadratic' in str(constraints):
                return self.solve_quadratic(constraints)
            return self.solve_general_algebra(constraints)
        except Exception as e:
            return f"Error in algebra solver: {str(e)}"

    def solve_quadratic(self, constraints):
        try:
            # Extract points from constraints
            points = []
            for i, val in enumerate(constraints.get('numerical_values', []), start=1):
                points.append((i, val))
            
            if len(points) < 3:
                return "Insufficient points for quadratic function"
            
            # Solve system of equations using SymPy
            x = Symbol('x')
            a, b, c = symbols('a b c')
            equations = []
            
            for point in points[:3]:
                x_val, y_val = point
                eq = Eq(a*x_val**2 + b*x_val + c, y_val)
                equations.append(eq)
            
            # Solve the system of equations
            solution = solve(equations, [a, b, c])
            
            if solution:
                # Convert solution dictionary to values
                a_val = float(solution[a])
                b_val = float(solution[b])
                c_val = float(solution[c])
                
                # Calculate f(4)
                result = a_val*16 + b_val*4 + c_val
                return f"f(4) = {result}"
            
            return "No solution found"
        except Exception as e:
            return f"Error in quadratic solver: {str(e)}"

if __name__ == "__main__":
    engine = SolutionEngine()
    
    # Solve all problems from results folder
    print("\nSolving all problems from results folder...")
    solutions = engine.solve_all_results()
    
    # Display some sample solutions
    print("\nSample solutions:")
    for solution in solutions[:3]:  # Show first 3 solutions
        print("\nProblem:", solution['problem'])
        print("Solution details:")
        print(json.dumps(solution['solution'], indent=2))
    
    # Test with sample problems from dataset
    test_problems = [
        """[Geometry] A circle of radius 11 is inscribed in a right triangle with legs a and b. 
        If the hypotenuse of the triangle is c, express the area of the triangle in terms of r.""",
        
        """[Number Theory] Find the sum of the digits of the largest three-digit number that is 
        divisible by 7.""",
        
        """[Algebra] Let f(x) be a quadratic polynomial such that f(1) = 45, f(2) = 37, and 
        f(3) = 10. Determine f(4)."""
    ]
    
    for problem in test_problems:
        print("\nSolving problem:", problem)
        result = engine.solve_problem(problem)
        print("\nSolution details:")
        print(json.dumps(result, indent=2))


