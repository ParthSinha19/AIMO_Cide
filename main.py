import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Input_Parser import MathProblemSolver
from mcts_framework import MCTS, ProblemClassifier, load_dataset, split_dataset
from sde_sorta import NumberTheoryCombinatoricsSolver, GeometrySolver, AlgebraOptimizationSolver
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MathProblemDataset(Dataset):
    def __init__(self, problems, feature_dim=512):
        self.problems = problems
        self.feature_dim = feature_dim
        self.parser = MathProblemSolver()
    
    def __len__(self):
        return len(self.problems)
        
    def __getitem__(self, idx):
        problem = self.problems[idx]
        # Parse problem using updated Input_Parser
        category, constraints = self.parser.parse_problem(problem['problem'])
        
        # Create feature vector from constraints
        features = torch.zeros(self.feature_dim)
        for i, value in enumerate(constraints.get('numerical_values', [])):
            features[i % self.feature_dim] = value
        for var in constraints.get('variables', []):
            features[hash(var) % self.feature_dim] = 1.0
        
        return {
            'features': features,
            'category': category,
            'constraints': constraints,
            'problem': problem['problem'],
            'solution': problem.get('solution', '')
        }

class Generator(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=256, output_dim=512):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh()
        )
        
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

class AnalysisManager:
    def __init__(self, output_dir="analysis_output"):
        self.output_dir = os.path.join(os.path.dirname(__file__), output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []
        self.training_stats = {'d_loss': [], 'g_loss': [], 'epochs': []}
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "results"), exist_ok=True)
    
    def add_training_stats(self, epoch, d_loss, g_loss):
        self.training_stats['epochs'].append(epoch)
        self.training_stats['d_loss'].append(d_loss)
        self.training_stats['g_loss'].append(g_loss)
    
    def add_result(self, problem, result):
        self.results.append({
            'problem': problem,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
    
    def plot_training_progress(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_stats['epochs'], self.training_stats['d_loss'], label='Discriminator Loss')
        plt.plot(self.training_stats['epochs'], self.training_stats['g_loss'], label='Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "plots", f"training_progress_{self.timestamp}.png"))
        plt.close()
    
    def plot_category_distribution(self):
        categories = [r['result']['category'] for r in self.results]
        plt.figure(figsize=(10, 6))
        sns.countplot(x=categories)
        plt.title('Problem Category Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", f"category_distribution_{self.timestamp}.png"))
        plt.close()
    
    def plot_confidence_distribution(self):
        confidences = [r['result']['confidence'] for r in self.results]
        plt.figure(figsize=(10, 6))
        sns.histplot(confidences, bins=20)
        plt.title('Solution Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.savefig(os.path.join(self.output_dir, "plots", f"confidence_distribution_{self.timestamp}.png"))
        plt.close()
    
    def save_results(self):
        output_file = os.path.join(self.output_dir, "results", f"results_{self.timestamp}.json")
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

# Modify MathProblemGAN class to include analysis
class MathProblemGAN:
    def __init__(self, feature_dim=512, latent_dim=100, hidden_dim=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        
        # Initialize components
        self.generator = Generator(latent_dim, hidden_dim, feature_dim).to(self.device)
        self.discriminator = Discriminator(feature_dim, hidden_dim).to(self.device)
        self.parser = MathProblemSolver()
        self.mcts = None  # Will be initialized per problem
        self.nt_solver = NumberTheoryCombinatoricsSolver()
        self.geo_solver = GeometrySolver()
        self.alg_solver = AlgebraOptimizationSolver()
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.criterion = nn.BCELoss()
        self.analysis_manager = AnalysisManager()
    
    def train_gan(self, dataloader, epochs=100):
        logger.info("Starting GAN training...")
        
        for epoch in range(epochs):
            for batch in dataloader:
                # Get real problem features
                real_features = batch['features'].to(self.device)
                batch_size = real_features.size(0)
                
                # Labels for real and fake data
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # Train Discriminator
                self.d_optimizer.zero_grad()
                
                # Real data
                d_output_real = self.discriminator(real_features)
                d_loss_real = self.criterion(d_output_real, real_labels)
                
                # Fake data
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_features = self.generator(z)
                d_output_fake = self.discriminator(fake_features.detach())
                d_loss_fake = self.criterion(d_output_fake, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()
                
                # Train Generator
                self.g_optimizer.zero_grad()
                g_output = self.discriminator(fake_features)
                g_loss = self.criterion(g_output, real_labels)
                g_loss.backward()
                self.g_optimizer.step()
                
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
                self.analysis_manager.add_training_stats(epoch + 1, d_loss.item(), g_loss.item())
    
    def solve_problem(self, problem_text):
        """Updated pipeline for solving math problems"""
        logger.info("Starting problem solving pipeline...")
        
        # Use SolutionEngine for problem solving
        result = self.solution_engine.solve_problem(problem_text)
        
        # Generate solution features using GAN
        z = torch.randn(1, self.latent_dim).to(self.device)
        solution_features = self.generator(z)
        
        # Verify solution using discriminator
        confidence = self.discriminator(solution_features).item()
        
        # Combine results
        final_result = {
            'category': result['category'],
            'parsed_constraints': result['parsed_constraints'],
            'solution_strategy': result['solution_strategy'],
            'solution': result['solution'],
            'confidence': confidence
        }
        
        # Add to analysis manager
        self.analysis_manager.add_result(problem_text, final_result)
        
        return final_result

def main():
    # Load and prepare dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "IMO_High_Level_Problems.txt")
    problems = load_dataset(dataset_path)
    train_problems, test_problems = split_dataset(problems)
    
    # Create dataset and dataloader
    train_dataset = MathProblemDataset(train_problems)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize and train GAN
    math_gan = MathProblemGAN()
    math_gan.train_gan(train_dataloader, epochs=50)  # Reduced epochs for testing
    
    # Test the system with different problem types
    test_problems = [
        """[Geometry] A circle of radius 11 is inscribed in a right triangle with legs a and b. 
        If the hypotenuse of the triangle is c, express the area of the triangle in terms of r.""",
        
        """[Number Theory] Find the sum of the digits of the largest three-digit number that is 
        divisible by 7.""",
        
        """[Algebra] Solve the equation x^2 + 5x + 6 = 0."""
    ]
    
    for problem in test_problems:
        logger.info("\nSolving problem:")
        logger.info(problem)
        result = math_gan.solve_problem(problem)
        logger.info("\nSolution details:")
        logger.info(json.dumps(result, indent=2))
    
    # Generate and save analysis
    math_gan.analysis_manager.plot_training_progress()
    math_gan.analysis_manager.plot_category_distribution()
    math_gan.analysis_manager.save_results()

if __name__ == "__main__":
    main()