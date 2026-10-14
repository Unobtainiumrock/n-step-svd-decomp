#!/usr/bin/env python3
"""
Fixed Tone Calibration System - "Face ID for Tone" Experience
Learns individual user preferences through A/B comparisons in residual space
"""
import numpy as np
import json
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import h5py
from sklearn.linear_model import Ridge
from residualized_tone_space.core import ResidualToneCalculator

@dataclass
class ContentPair:
    """A pair of content for A/B comparison."""
    content_a_id: str
    content_b_id: str
    content_a_text: str
    content_b_text: str
    content_a_embedding: np.ndarray
    content_b_embedding: np.ndarray
    pair_id: str

@dataclass
class UserChoice:
    """User's choice between two content pieces."""
    pair_id: str
    chosen_content_id: str
    rejected_content_id: str
    chosen_embedding: np.ndarray
    rejected_embedding: np.ndarray
    confidence: Optional[float] = None

class ToneCalibrationSystem:
    """
    System for calibrating individual user tone preferences.
    Implements the 'Face ID for Tone' experience.
    """
    
    def __init__(self, embeddings_file: str = "data/embeddings/ugc-posts_COMPLETE_embeddings.h5"):
        """Initialize with pre-computed residual tone space components."""
        
        # Load residual tone space components
        self.calculator = self._load_residual_calculator()
        
        # Load content embeddings and metadata
        self.embeddings, self.content_ids, self.content_texts = self._load_content(embeddings_file)
        
        # Calculate residual vectors for all content
        print("🔄 Calculating residual vectors for all content...")
        self.residual_vectors = self.calculator.calculate_residual_vectors(self.embeddings)
        print(f"✅ Residual vectors calculated: {self.residual_vectors.shape}")
        
        # User preference learning state
        self.user_choices: List[UserChoice] = []
        self.user_preference_vector: Optional[np.ndarray] = None
        self.calibration_confidence: float = 0.0
        
    def _load_residual_calculator(self) -> ResidualToneCalculator:
        """Load the pre-computed residual tone space calculator."""
        projection_matrix = np.load("data/reports/optimal_160_projection_matrix.npy")
        calculator = ResidualToneCalculator(projection_matrix)
        
        # Load tone axes (optional - we'll learn user-specific preferences)
        tone_axes = np.load("data/reports/optimal_130_tone_axes.npy")
        tone_axis_names = [f"Tone_Axis_{i+1}" for i in range(len(tone_axes))]
        calculator.load_tone_axis_vectors(tone_axis_names, tone_axes)
        
        return calculator
    
    def _load_content(self, embeddings_file: str) -> Tuple[np.ndarray, List[str], List[str]]:
        """Load content embeddings and metadata."""
        with h5py.File(embeddings_file, 'r') as f:
            embeddings = f['embeddings'][:]
            content_ids = [id.decode('utf-8') for id in f['ids'][:]]
            
            # Try to load text content if available
            if 'texts' in f:
                content_texts = [text.decode('utf-8') for text in f['texts'][:]]
            else:
                # Fallback: use content IDs as text
                content_texts = [f"Content {id}" for id in content_ids]
        
        return embeddings, content_ids, content_texts
    
    def generate_calibration_pair(self, strategy: str = "diverse_residual") -> ContentPair:
        """Generate a pair of content for user comparison."""
        
        if strategy == "diverse_residual":
            return self._generate_diverse_residual_pair()
        elif strategy == "tone_axis_contrast":
            return self._generate_tone_axis_contrast_pair()
        elif strategy == "uncertainty_sampling":
            return self._generate_uncertainty_sampling_pair()
        else:  # random
            return self._generate_random_pair()
    
    def _generate_diverse_residual_pair(self) -> ContentPair:
        """Generate pair with maximum difference in residual space."""
        
        # Sample candidates
        n_candidates = min(1000, len(self.residual_vectors))
        candidate_indices = random.sample(range(len(self.residual_vectors)), n_candidates)
        candidate_residuals = self.residual_vectors[candidate_indices]
        
        # Find pair with maximum residual distance
        max_distance = 0
        best_pair = (0, 1)
        
        for i in range(len(candidate_residuals)):
            for j in range(i + 1, min(i + 50, len(candidate_residuals))):  # Limit comparisons
                distance = np.linalg.norm(candidate_residuals[i] - candidate_residuals[j])
                if distance > max_distance:
                    max_distance = distance
                    best_pair = (candidate_indices[i], candidate_indices[j])
        
        return self._create_content_pair(best_pair[0], best_pair[1])
    
    def _generate_tone_axis_contrast_pair(self) -> ContentPair:
        """Generate pair with contrast along existing tone axes."""
        
        # Choose a random tone axis
        tone_axis_names = self.calculator.get_tone_axis_names()
        if not tone_axis_names:
            return self._generate_random_pair()
        
        axis_name = random.choice(tone_axis_names)
        
        # Score all content on this axis
        all_scores = self.calculator.calculate_tone_scores_all_axes(self.embeddings)
        axis_scores = all_scores[axis_name]
        
        # Find high and low scoring content
        high_indices = np.argsort(axis_scores)[-100:]  # Top 100
        low_indices = np.argsort(axis_scores)[:100]    # Bottom 100
        
        high_idx = random.choice(high_indices)
        low_idx = random.choice(low_indices)
        
        return self._create_content_pair(high_idx, low_idx)
    
    def _generate_uncertainty_sampling_pair(self) -> ContentPair:
        """Generate pair focusing on areas of uncertainty in current model."""
        
        if self.user_preference_vector is None:
            return self._generate_diverse_residual_pair()
        
        # Score content with current preference vector
        scores = np.dot(self.residual_vectors, self.user_preference_vector)
        
        # Find content near the decision boundary (score ≈ 0)
        abs_scores = np.abs(scores)
        uncertain_indices = np.argsort(abs_scores)[:200]  # Most uncertain
        
        # Select two from uncertain region
        if len(uncertain_indices) >= 2:
            selected = random.sample(list(uncertain_indices), 2)
            return self._create_content_pair(selected[0], selected[1])
        else:
            return self._generate_random_pair()
    
    def _generate_random_pair(self) -> ContentPair:
        """Generate a random pair of content."""
        indices = random.sample(range(len(self.embeddings)), 2)
        return self._create_content_pair(indices[0], indices[1])
    
    def _create_content_pair(self, idx_a: int, idx_b: int) -> ContentPair:
        """Create a ContentPair object from two indices."""
        pair_id = f"pair_{len(self.user_choices)}_{random.randint(1000, 9999)}"
        
        return ContentPair(
            content_a_id=self.content_ids[idx_a],
            content_b_id=self.content_ids[idx_b],
            content_a_text=self.content_texts[idx_a],
            content_b_text=self.content_texts[idx_b],
            content_a_embedding=self.embeddings[idx_a],
            content_b_embedding=self.embeddings[idx_b],
            pair_id=pair_id
        )
    
    def record_user_choice(self, pair: ContentPair, chosen_content_id: str, confidence: float = 1.0):
        """Record user's choice and update preference model."""
        
        # Determine chosen and rejected
        if chosen_content_id == pair.content_a_id:
            chosen_embedding = pair.content_a_embedding
            rejected_embedding = pair.content_b_embedding
            rejected_content_id = pair.content_b_id
        else:
            chosen_embedding = pair.content_b_embedding
            rejected_embedding = pair.content_a_embedding
            rejected_content_id = pair.content_a_id
        
        # Create choice record
        choice = UserChoice(
            pair_id=pair.pair_id,
            chosen_content_id=chosen_content_id,
            rejected_content_id=rejected_content_id,
            chosen_embedding=chosen_embedding,
            rejected_embedding=rejected_embedding,
            confidence=confidence
        )
        
        self.user_choices.append(choice)
        
        # Update preference model
        self._update_preference_model()
        
        print(f"✅ Recorded choice {len(self.user_choices)}: {chosen_content_id}")
        print(f"   Calibration confidence: {self.calibration_confidence:.3f}")
    
    def _update_preference_model(self):
        """Update user preference model based on all choices."""
        
        if len(self.user_choices) < 1:
            self.calibration_confidence = 0.0
            return
        
        # Prepare training data in residual space
        X = []  # Difference vectors
        y = []  # Preference scores
        weights = []  # Confidence weights
        
        for choice in self.user_choices:
            # Get residual vectors
            chosen_residual = self.calculator.get_residual_embedding(choice.chosen_embedding)
            rejected_residual = self.calculator.get_residual_embedding(choice.rejected_embedding)
            
            # Difference vector (chosen - rejected)
            diff_vector = chosen_residual - rejected_residual
            X.append(diff_vector)
            y.append(1.0)  # Positive preference
            weights.append(choice.confidence)
        
        X = np.array(X)
        y = np.array(y)
        weights = np.array(weights)
        
        # Use Ridge regression instead of Logistic Regression
        # This handles the single-class problem better
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X, y, sample_weight=weights)
        
        # Extract preference vector
        self.user_preference_vector = ridge.coef_
        self.user_preference_vector = self.user_preference_vector / np.linalg.norm(self.user_preference_vector)
        
        # Calculate calibration confidence
        self._calculate_calibration_confidence(ridge, X, y)
    
    def _calculate_calibration_confidence(self, model, X, y):
        """Calculate how confident we are in the calibration."""
        
        # Use R² score as confidence measure
        if len(X) >= 2:
            score = model.score(X, y)
            score = max(0.0, score)  # Ensure non-negative
        else:
            score = 0.1
        
        # Number of choices factor
        choice_factor = min(1.0, len(self.user_choices) / 10.0)  # Saturate at 10 choices
        
        # Combine factors
        self.calibration_confidence = score * choice_factor
    
    def is_calibrated(self, min_choices: int = 5, min_confidence: float = 0.5) -> bool:
        """Check if user is sufficiently calibrated."""
        return (len(self.user_choices) >= min_choices and 
                self.calibration_confidence >= min_confidence)
    
    def score_content(self, content_embedding: np.ndarray) -> float:
        """Score new content with user's calibrated preferences."""
        
        if self.user_preference_vector is None:
            raise ValueError("User not calibrated yet")
        
        residual = self.calculator.get_residual_embedding(content_embedding)
        score = np.dot(residual, self.user_preference_vector)
        
        return float(score)
    
    def get_calibration_status(self) -> Dict:
        """Get current calibration status."""
        return {
            'choices_made': len(self.user_choices),
            'calibration_confidence': self.calibration_confidence,
            'is_calibrated': self.is_calibrated(),
            'has_preference_vector': self.user_preference_vector is not None,
            'next_strategy': self._get_next_strategy()
        }
    
    def _get_next_strategy(self) -> str:
        """Determine the best strategy for the next pair."""
        
        if len(self.user_choices) < 3:
            return "diverse_residual"  # Start with diverse examples
        elif len(self.user_choices) < 7:
            return "tone_axis_contrast"  # Explore known axes
        else:
            return "uncertainty_sampling"  # Focus on decision boundary
    
    def save_calibration(self, user_id: str, output_dir: str = "data/calibrations"):
        """Save user's calibration data."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        calibration_data = {
            'user_id': user_id,
            'choices': [
                {
                    'pair_id': choice.pair_id,
                    'chosen_content_id': choice.chosen_content_id,
                    'rejected_content_id': choice.rejected_content_id,
                    'confidence': choice.confidence
                }
                for choice in self.user_choices
            ],
            'preference_vector': self.user_preference_vector.tolist() if self.user_preference_vector is not None else None,
            'calibration_confidence': self.calibration_confidence,
            'status': self.get_calibration_status()
        }
        
        output_file = output_path / f"{user_id}_calibration.json"
        with open(output_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"💾 Calibration saved: {output_file}")
        return output_file

def demo_calibration_experience():
    """Demonstrate the calibration experience."""
    
    print("🎯 Tone Calibration System Demo")
    print("=" * 50)
    
    # Initialize system
    system = ToneCalibrationSystem()
    
    # Simulate calibration process
    user_id = "demo_user_001"
    
    while not system.is_calibrated():
        status = system.get_calibration_status()
        print(f"\n📊 Status: {status['choices_made']} choices, {status['calibration_confidence']:.3f} confidence")
        
        # Generate pair
        strategy = status['next_strategy']
        pair = system.generate_calibration_pair(strategy)
        
        print(f"\n🎯 Pair {status['choices_made'] + 1} (strategy: {strategy}):")
        print(f"   A: {pair.content_a_id}")
        print(f"   B: {pair.content_b_id}")
        
        # Simulate user choice (random for demo)
        chosen = random.choice([pair.content_a_id, pair.content_b_id])
        confidence = random.uniform(0.7, 1.0)
        
        system.record_user_choice(pair, chosen, confidence)
    
    print(f"\n🎉 Calibration Complete!")
    print(f"   Final confidence: {system.calibration_confidence:.3f}")
    
    # Save calibration
    calibration_file = system.save_calibration(user_id)
    
    # Test scoring
    print(f"\n🧪 Testing content scoring...")
    sample_embedding = system.embeddings[0]
    score = system.score_content(sample_embedding)
    print(f"   Sample content score: {score:.3f}")
    
    return system

if __name__ == "__main__":
    demo_calibration_experience()
