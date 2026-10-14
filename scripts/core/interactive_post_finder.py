#!/usr/bin/env python3
"""
Interactive Optimal Post Finder - Make real choices via CLI
Experience the convergent search process yourself!
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

from residualized_tone_space.core import ResidualToneCalculator

# Optional Supabase integration
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("💡 Supabase not installed. Install with: pip install supabase")

@dataclass
class PostCandidate:
    """Represents a post candidate in the search."""
    post_id: str
    embedding: np.ndarray
    topic_projection: np.ndarray
    residual_vector: np.ndarray
    content: str
    confidence_score: float = 0.0

@dataclass 
class SearchChoice:
    """Represents a user choice in the search."""
    round_number: int
    chosen_post: PostCandidate
    rejected_post: PostCandidate
    search_strategy: str
    confidence_gain: float
    choice_distance: float
    search_center_movement: float

class InteractiveOptimalPostFinder:
    """
    Interactive convergent search system where YOU make the choices
    """
    
    def __init__(self, embeddings_file: str = 'data/embeddings/ugc-posts_COMPLETE_embeddings.h5'):
        """Initialize the interactive optimal post finder."""

        # Load embeddings and posts
        with h5py.File(embeddings_file, 'r') as f:
            self.all_embeddings = f['embeddings'][:]
            self.all_ids = [id.decode('utf-8') for id in f['ids'][:]]
            
            # Try to load metadata for better content display
            self.metadata = []
            if 'metadata' in f:
                try:
                    for i in range(len(f['metadata'])):
                        metadata_json = f['metadata'][i].decode('utf-8')
                        self.metadata.append(json.loads(metadata_json))
                except:
                    self.metadata = [{} for _ in range(len(self.all_ids))]
            else:
                self.metadata = [{} for _ in range(len(self.all_ids))]
        
        # Load optimal components
        self.topic_projection_matrix = np.load('data/reports/optimal_160_projection_matrix.npy')
        self.tone_axes = np.load('data/reports/optimal_130_tone_axes.npy')
        
        # Initialize calculator
        self.calculator = ResidualToneCalculator(self.topic_projection_matrix)
        self.calculator.load_tone_axis_vectors(
            [f"Tone_Axis_{i+1}" for i in range(len(self.tone_axes))], 
            self.tone_axes
        )
        
        # Search state
        self.search_history: List[SearchChoice] = []
        self.current_best_candidate: Optional[PostCandidate] = None
        self.search_region_center: Optional[np.ndarray] = None
        self.search_radius = 1.0
        self.convergence_threshold = 0.95
        self.max_rounds = 50
        
        # Initialize Supabase connection if available
        self.supabase_client = None
        self.content_cache = {}  # Cache fetched content
        
        if SUPABASE_AVAILABLE:
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_ANON_KEY')
            
            if supabase_url and supabase_key:
                try:
                    self.supabase_client = create_client(supabase_url, supabase_key)
                    print(f"✅ Supabase connection established")
                except Exception as e:
                    print(f"⚠️  Supabase connection failed: {e}")
            else:
                print(f"💡 Supabase env vars not found (SUPABASE_URL, SUPABASE_ANON_KEY)")
        
        print(f"📊 Configuration:")
        print(f"  ├─ Dataset size: {len(self.all_embeddings):,} posts")
        print(f"  ├─ IDs loaded: {len(self.all_ids):,}")
        print(f"  ├─ Metadata loaded: {len(self.metadata):,}")
        print(f"  ├─ Supabase integration: {'✅ Connected' if self.supabase_client else '❌ Not available'}")
        print(f"  ├─ Topic dimensions: 160 (optimal)")
        print(f"  ├─ Tone dimensions: 130 (optimal)")
        print(f"  └─ Convergence threshold: {self.convergence_threshold}")
        print(f"✅ Initialization complete!")
    
    def fetch_post_content_from_supabase(self, post_id: str) -> Optional[str]:
        """Fetch actual post content from Supabase."""
        
        if not self.supabase_client:
            return None
        
        # Check cache first
        if post_id in self.content_cache:
            return self.content_cache[post_id]
        
        try:
            # Determine table and query based on ID format
            if post_id.endswith('_video'):
                # Video content table - strip _video suffix for Supabase lookup
                clean_id = post_id.replace('_video', '')
                table_name = 'ugc_videos'
                content_field = 'script'
            else:
                # Regular posts table - use ID as-is
                clean_id = post_id
                table_name = 'ugc_posts'
                content_field = 'post_text'
            
            # Query Supabase with cleaned ID
            print(f"🔍 Fetching content: {post_id} → table={table_name}, id={clean_id}")
            response = self.supabase_client.table(table_name).select(content_field).eq('id', int(clean_id)).execute()
            
            if response.data and len(response.data) > 0:
                content = response.data[0].get(content_field, '')
                if content:
                    # Cache the result
                    self.content_cache[post_id] = content
                    return content
            
            return None
            
        except Exception as e:
            print(f"⚠️  Supabase fetch error for {post_id}: {e}")
            return None
        
    def get_post_preview(self, post_id: str, post_idx: int) -> str:
        """Get a meaningful preview of the post for display."""
        
        try:
            # Try to get actual content from Supabase first
            actual_content = self.fetch_post_content_from_supabase(post_id)
            
            if actual_content:
                # Clean and truncate the actual content
                content = str(actual_content).strip().replace('\n', ' ').replace('\r', ' ')
                if len(content) > 150:
                    content = content[:150] + "..."
                
                # Add type indicator
                if post_id.endswith('_video'):
                    return f"🎥 VIDEO: \"{content}\""
                else:
                    return f"📝 TEXT: \"{content}\""
            
            # Fallback to metadata-based description
            if post_idx >= len(self.metadata):
                return f"Post {post_id} - No metadata available"
                
            metadata = self.metadata[post_idx]
            
            if not isinstance(metadata, dict):
                return f"Post {post_id} - Invalid metadata format"
            
            # Extract meaningful information from metadata
            post_type = metadata.get('post_type', 'unknown')
            content_type = metadata.get('embedded_content', 'unknown')
            
            # Get content length information
            text_length = metadata.get('post_text_length', 0)
            video_length = metadata.get('video_script_length', 0)
            
            # Get timestamps for context
            created_at = metadata.get('created_at', 0)
            posted_at = metadata.get('posted_at', 0)
            
            # Create descriptive preview
            description_parts = []
            
            # Content type and length
            if content_type == 'post_text' and text_length > 0:
                description_parts.append(f"{text_length} char text post")
            elif content_type == 'video_script' and video_length > 0:
                description_parts.append(f"{video_length} char video script")
            else:
                description_parts.append(f"{post_type} post")
            
            # Add timing context if available
            if created_at and posted_at:
                import datetime
                try:
                    posted_date = datetime.datetime.fromtimestamp(posted_at)
                    description_parts.append(f"posted {posted_date.strftime('%Y-%m-%d')}")
                except:
                    pass
            
            # Combine description
            description = " | ".join(description_parts)
            
            # Add ID suffix info and note about missing content
            if post_id.endswith('_video'):
                return f"🎥 VIDEO: {description} (content not cached)"
            else:
                return f"📝 TEXT: {description} (content not cached)"
            
        except Exception as e:
            return f"Post {post_id} - Error: {str(e)[:50]}"
    
    def display_post_choice(self, post_a: PostCandidate, post_b: PostCandidate, round_num: int, strategy: str):
        """Display two posts for user to choose between."""
        
        print(f"\n{'='*60}")
        print(f"🎯 ROUND {round_num} - {strategy}")
        print(f"{'='*60}")
        
        # Get post indices for content lookup with error handling
        try:
            idx_a = self.all_ids.index(post_a.post_id)
            idx_b = self.all_ids.index(post_b.post_id)
        except ValueError as e:
            print(f"❌ Error finding post indices: {e}")
            print(f"Post A ID: {post_a.post_id}, Post B ID: {post_b.post_id}")
            # Use fallback indices
            idx_a = int(post_a.post_id) if post_a.post_id.isdigit() else 0
            idx_b = int(post_b.post_id) if post_b.post_id.isdigit() else 1
        
        # Display posts with error handling
        print(f"\n📝 Choose your preferred post:")
        
        # Display Option A
        try:
            preview_a = self.get_post_preview(post_a.post_id, idx_a)
            print(f"\n[A] {preview_a}")
            print(f"    ID: {post_a.post_id}")
        except Exception as e:
            print(f"\n[A] Post {post_a.post_id}")
            print(f"    Content preview unavailable: {e}")
        
        # Display Option B  
        try:
            preview_b = self.get_post_preview(post_b.post_id, idx_b)
            print(f"\n[B] {preview_b}")
            print(f"    ID: {post_b.post_id}")
        except Exception as e:
            print(f"\n[B] Post {post_b.post_id}")
            print(f"    Content preview unavailable: {e}")
        
        # Show current search status
        if self.current_best_candidate:
            print(f"\n📊 Current Status:")
            print(f"  ├─ Current best: Post {self.current_best_candidate.post_id}")
            print(f"  ├─ Confidence: {self.current_best_candidate.confidence_score:.3f}")
            print(f"  ├─ Search radius: {self.search_radius:.3f}")
            print(f"  └─ Strategy: {strategy}")
        
        # Show what this stage is doing
        self._explain_current_strategy(strategy)
    
    def _explain_current_strategy(self, strategy: str):
        """Explain what the current strategy is trying to accomplish."""
        
        print(f"\n💡 What this stage does:")
        
        if strategy == "BROAD_EXPLOR":
            print(f"  └─ Finding general direction in content space")
            print(f"     These posts are very different - pick the general direction you prefer")
        elif strategy == "TOPIC_REFINE":
            print(f"  └─ Refining topic preferences (similar tone, different topics)")
            print(f"     These posts have similar style but different subject matter")
        elif strategy == "TONE_REFINE":
            print(f"  └─ Refining tone preferences (similar topic, different tones)")
            print(f"     These posts discuss similar topics but with different styles")
        elif strategy == "FINE_TUNING":
            print(f"  └─ Fine-tuning to exact optimal post")
            print(f"     These posts are very similar - pick the one that feels more 'you'")
    
    def get_user_choice(self, post_a: PostCandidate, post_b: PostCandidate) -> Tuple[PostCandidate, PostCandidate]:
        """Get user's choice between two posts."""
        
        while True:
            print(f"\n❓ Your choice:")
            choice = input("Enter 'A' or 'B' (or 'q' to quit): ").strip().upper()
            
            if choice == 'Q':
                print("👋 Exiting search...")
                exit(0)
            elif choice == 'A':
                return post_a, post_b
            elif choice == 'B':
                return post_b, post_a
            else:
                print("❌ Please enter 'A', 'B', or 'q'")
    
    def start_interactive_search(self):
        """Start the interactive search session."""
        
        print(f"\n🚀 STARTING INTERACTIVE SEARCH")
        print(f"=" * 50)
        print(f"Instructions:")
        print(f"  ├─ You'll see two posts at a time")
        print(f"  ├─ Choose A or B based on your preference") 
        print(f"  ├─ The system will learn and adapt")
        print(f"  └─ We'll converge to your optimal post!")
        
        # Reset search state
        self.search_history = []
        self.current_best_candidate = None
        self.search_region_center = None
        self.search_radius = 1.0
        
        # Generate initial diverse pair
        post_a, post_b = self._generate_diverse_pair()
        
        # Start the interactive loop
        try:
            while True:
                round_num = len(self.search_history) + 1
                strategy = self._get_current_strategy()
                
                print(f"\n🔄 Preparing round {round_num}...")
                
                # Show recently shown posts for debugging
                if len(self.search_history) > 0:
                    recent_posts = [choice.chosen_post.post_id for choice in self.search_history[-2:]]
                    recent_posts.extend([choice.rejected_post.post_id for choice in self.search_history[-2:]])
                    recent_posts = list(set(recent_posts))  # Remove duplicates
                    print(f"   (Avoiding recent posts: {recent_posts})")
                
                # Display choice to user
                try:
                    self.display_post_choice(post_a, post_b, round_num, strategy)
                except Exception as e:
                    print(f"❌ Error displaying posts: {e}")
                    print(f"Post A: {post_a.post_id}, Post B: {post_b.post_id}")
                    print(f"Continuing with basic display...")
                
                # Get user's choice
                try:
                    chosen, rejected = self.get_user_choice(post_a, post_b)
                except Exception as e:
                    print(f"❌ Error getting user choice: {e}")
                    break
                
                # Process the choice
                try:
                    result = self.record_choice(chosen, rejected)
                except Exception as e:
                    print(f"❌ Error processing choice: {e}")
                    print(f"Chosen: {chosen.post_id}, Rejected: {rejected.post_id}")
                    break
                
                if result['converged']:
                    self._show_convergence_success(result)
                    break
                elif result.get('reason') == 'max_rounds_reached':
                    self._show_max_rounds_reached(result)
                    break
                else:
                    try:
                        post_a, post_b = result['next_pair']
                        self._show_progress_update(result)
                    except Exception as e:
                        print(f"❌ Error generating next pair: {e}")
                        break
                        
        except KeyboardInterrupt:
            print(f"\n\n👋 Search interrupted by user. Goodbye!")
        except Exception as e:
            print(f"\n❌ Unexpected error in search loop: {e}")
            print(f"Search history: {len(self.search_history)} rounds completed")
    
    def record_choice(self, chosen_post: PostCandidate, rejected_post: PostCandidate) -> Dict:
        """Record user choice and update search state."""
        
        round_number = len(self.search_history) + 1
        strategy = self._get_current_strategy()
        
        # Calculate metrics
        choice_distance = np.linalg.norm(chosen_post.embedding - rejected_post.embedding)
        
        # Store previous state
        prev_candidate = self.current_best_candidate.post_id if self.current_best_candidate else "None"
        prev_confidence = self.current_best_candidate.confidence_score if self.current_best_candidate else 0.0
        prev_radius = self.search_radius
        
        # Calculate center movement before update
        prev_center = self.search_region_center.copy() if self.search_region_center is not None else None
        
        # Update search state
        self._update_search_state(chosen_post, rejected_post)
        
        # Calculate deltas
        center_movement = 0.0
        if prev_center is not None:
            center_movement = np.linalg.norm(self.search_region_center - prev_center)
        
        confidence_delta = self.current_best_candidate.confidence_score - prev_confidence
        radius_delta = self.search_radius - prev_radius
        
        # Record choice
        choice = SearchChoice(
            round_number=round_number,
            chosen_post=chosen_post,
            rejected_post=rejected_post,
            search_strategy=strategy,
            confidence_gain=confidence_delta,
            choice_distance=choice_distance,
            search_center_movement=center_movement
        )
        self.search_history.append(choice)
        
        # Check convergence
        converged, convergence_status = self._check_convergence_with_status()
        
        if converged:
            return {
                'converged': True,
                'optimal_post': self.current_best_candidate,
                'rounds_completed': round_number,
                'final_confidence': self.current_best_candidate.confidence_score
            }
        
        # Safety check
        if round_number >= self.max_rounds:
            return {
                'converged': False,
                'optimal_post': self.current_best_candidate,
                'rounds_completed': round_number,
                'final_confidence': self.current_best_candidate.confidence_score,
                'reason': 'max_rounds_reached'
            }
        
        return {
            'converged': False,
            'next_pair': self._generate_next_pair(),
            'rounds_completed': round_number,
            'current_confidence': self.current_best_candidate.confidence_score,
            'convergence_status': convergence_status
        }
    
    def _show_progress_update(self, result: Dict):
        """Show progress update after each choice."""
        
        print(f"\n📈 PROGRESS UPDATE:")
        print(f"  ├─ Round completed: {result['rounds_completed']}")
        print(f"  ├─ Current confidence: {result['current_confidence']:.3f}")
        print(f"  ├─ Status: {result['convergence_status']}")
        print(f"  └─ Search continuing...")
        
        input("\n⏸️  Press Enter to continue to next round...")
    
    def _show_convergence_success(self, result: Dict):
        """Show convergence success message."""
        
        print(f"\n🎉 CONVERGENCE ACHIEVED!")
        print(f"=" * 50)
        print(f"🏆 YOUR OPTIMAL POST: {result['optimal_post'].post_id}")
        print(f"📊 Final confidence: {result['final_confidence']:.3f}")
        print(f"🔢 Total rounds: {result['rounds_completed']}")
        
        # Show the optimal post content
        try:
            idx = self.all_ids.index(result['optimal_post'].post_id)
            content_preview = self.get_post_preview(result['optimal_post'].post_id, idx)
            print(f"\n📝 Your optimal post content:")
            print(f"    {content_preview}")
        except:
            pass
        
        # Show search journey
        print(f"\n📈 YOUR SEARCH JOURNEY:")
        for i, choice in enumerate(self.search_history, 1):
            confidence = choice.chosen_post.confidence_score if hasattr(choice.chosen_post, 'confidence_score') else 0
            print(f"  Round {i:2d}: Chose {choice.chosen_post.post_id} ({choice.search_strategy})")
        
        print(f"\n🎯 This post represents your optimal content preferences!")
        print(f"   It balances both topic (what to say) and tone (how to say it)")
        print(f"   based on your strategic choices through the search process.")
        
        # Comprehensive choice analysis
        self._analyze_user_choices()
        
        # Generate content recommendation
        self._generate_ideal_post_recommendation(result['optimal_post'])
    
    def _show_max_rounds_reached(self, result: Dict):
        """Show message when max rounds reached."""
        
        print(f"\n⏱️ MAXIMUM ROUNDS REACHED")
        print(f"=" * 40)
        print(f"🎯 Best candidate found: {result['optimal_post'].post_id}")
        print(f"📊 Final confidence: {result['final_confidence']:.3f}")
        print(f"🔢 Rounds completed: {result['rounds_completed']}")
        print(f"\nWhile we didn't reach full convergence, this is your best match!")
    
    def _check_convergence_with_status(self) -> Tuple[bool, str]:
        """Check convergence and return status string."""
        
        if self.current_best_candidate is None:
            return False, "no_candidate"
        
        confidence_check = self.current_best_candidate.confidence_score > self.convergence_threshold
        radius_check = self.search_radius < 0.2
        rounds_check = len(self.search_history) >= 5
        
        converged = confidence_check and radius_check and rounds_check
        
        # Create status message
        status_parts = []
        if not confidence_check:
            needed = self.convergence_threshold - self.current_best_candidate.confidence_score
            status_parts.append(f"confidence_needed={needed:.3f}")
        if not radius_check:
            needed = self.search_radius - 0.2
            status_parts.append(f"focus_needed={needed:.3f}")
        if not rounds_check:
            needed = 5 - len(self.search_history)
            status_parts.append(f"rounds_needed={needed}")
        
        if converged:
            return True, "ALL_CRITERIA_MET"
        else:
            return False, ", ".join(status_parts) if status_parts else "checking"
    
    # Include all the core methods from the final version
    def _update_search_state(self, chosen_post: PostCandidate, rejected_post: PostCandidate):
        """Update the search state based on user choice."""
        
        if self.current_best_candidate is None:
            self.current_best_candidate = chosen_post
            self.search_region_center = chosen_post.embedding.copy()
        else:
            direction = chosen_post.embedding - self.current_best_candidate.embedding
            step_size = 0.3
            self.search_region_center += step_size * direction
            self.current_best_candidate = self._find_closest_post(self.search_region_center)
        
        self.search_radius *= 0.8
        self.search_radius = max(0.1, self.search_radius)
        
        self.current_best_candidate.confidence_score = self._calculate_post_confidence(
            self.current_best_candidate
        )
    
    def _get_current_strategy(self) -> str:
        """Determine current search strategy based on round number."""
        
        round_num = len(self.search_history) + 1
        
        if round_num <= 2:
            return "BROAD_EXPLOR"
        elif round_num <= 6:
            return "TOPIC_REFINE" 
        elif round_num <= 10:
            return "TONE_REFINE"
        else:
            return "FINE_TUNING"
    
    def _generate_diverse_pair(self) -> Tuple[PostCandidate, PostCandidate]:
        """Generate initial diverse pair for broad exploration."""
        
        # Sample random posts that are far apart
        import random
        indices = random.sample(range(len(self.all_embeddings)), 100)
        embeddings_sample = self.all_embeddings[indices]
        
        # Find two most dissimilar posts
        similarities = cosine_similarity(embeddings_sample)
        min_sim_idx = np.unravel_index(np.argmin(similarities + np.eye(len(similarities))), 
                                       similarities.shape)
        
        idx_a, idx_b = indices[min_sim_idx[0]], indices[min_sim_idx[1]]
        
        post_a = self._create_post_candidate(idx_a)
        post_b = self._create_post_candidate(idx_b)
        
        return post_a, post_b
    
    def _generate_next_pair(self) -> Tuple[PostCandidate, PostCandidate]:
        """Generate next pair based on current strategy."""
        
        strategy = self._get_current_strategy()
        
        if strategy == "TOPIC_REFINE":
            return self._generate_topic_refinement_pair()
        elif strategy == "TONE_REFINE":
            return self._generate_tone_refinement_pair()
        elif strategy == "FINE_TUNING":
            return self._generate_fine_tuning_pair()
        else:
            return self._generate_diverse_pair()
    
    def _generate_topic_refinement_pair(self) -> Tuple[PostCandidate, PostCandidate]:
        """Generate pair for topic refinement (similar tone, different topics)."""
        
        current_residual = self.current_best_candidate.residual_vector
        all_residuals = self.calculator.calculate_residual_vectors(self.all_embeddings)
        residual_similarities = cosine_similarity([current_residual], all_residuals)[0]
        
        similar_tone_indices = np.where(residual_similarities > 0.7)[0]
        if len(similar_tone_indices) < 2:
            similar_tone_indices = np.argsort(residual_similarities)[-20:]
        
        topic_projections = self.all_embeddings[similar_tone_indices] @ self.topic_projection_matrix
        current_topic = self.current_best_candidate.topic_projection
        topic_similarities = cosine_similarity([current_topic], topic_projections)[0]
        
        # Avoid showing the same pairs repeatedly
        # Exclude recently shown posts
        recent_post_ids = {choice.chosen_post.post_id for choice in self.search_history[-3:]}
        recent_post_ids.update({choice.rejected_post.post_id for choice in self.search_history[-3:]})
        
        # Filter out recent posts
        available_indices = []
        available_similarities = []
        for i, sim in enumerate(topic_similarities):
            post_idx = similar_tone_indices[i]
            post_id = self.all_ids[post_idx]
            if post_id not in recent_post_ids:
                available_indices.append(i)
                available_similarities.append(sim)
        
        if len(available_indices) < 2:
            # Fallback: use all indices if we filtered too much
            available_indices = list(range(len(topic_similarities)))
            available_similarities = topic_similarities
        
        # Pick most similar and most different from available
        available_similarities = np.array(available_similarities)
        similar_idx = similar_tone_indices[available_indices[np.argmax(available_similarities)]]
        different_idx = similar_tone_indices[available_indices[np.argmin(available_similarities)]]
        
        post_a = self._create_post_candidate(similar_idx)
        post_b = self._create_post_candidate(different_idx)
        
        return post_a, post_b
    
    def _generate_tone_refinement_pair(self) -> Tuple[PostCandidate, PostCandidate]:
        """Generate pair for tone refinement (similar topic, different tones)."""
        
        current_topic = self.current_best_candidate.topic_projection
        all_topics = self.all_embeddings @ self.topic_projection_matrix
        topic_similarities = cosine_similarity([current_topic], all_topics)[0]
        
        similar_topic_indices = np.where(topic_similarities > 0.7)[0]
        if len(similar_topic_indices) < 2:
            similar_topic_indices = np.argsort(topic_similarities)[-20:]
        
        residuals = self.calculator.calculate_residual_vectors(self.all_embeddings[similar_topic_indices])
        current_residual = self.current_best_candidate.residual_vector
        residual_similarities = cosine_similarity([current_residual], residuals)[0]
        
        # Avoid showing the same pairs repeatedly
        # Exclude recently shown posts
        recent_post_ids = {choice.chosen_post.post_id for choice in self.search_history[-3:]}
        recent_post_ids.update({choice.rejected_post.post_id for choice in self.search_history[-3:]})
        
        # Filter out recent posts
        available_indices = []
        available_similarities = []
        for i, sim in enumerate(residual_similarities):
            post_idx = similar_topic_indices[i]
            post_id = self.all_ids[post_idx]
            if post_id not in recent_post_ids:
                available_indices.append(i)
                available_similarities.append(sim)
        
        if len(available_indices) < 2:
            # Fallback: use all indices if we filtered too much
            available_indices = list(range(len(residual_similarities)))
            available_similarities = residual_similarities
        
        # Pick most similar and most different from available
        available_similarities = np.array(available_similarities)
        similar_idx = similar_topic_indices[available_indices[np.argmax(available_similarities)]]
        different_idx = similar_topic_indices[available_indices[np.argmin(available_similarities)]]
        
        post_a = self._create_post_candidate(similar_idx)
        post_b = self._create_post_candidate(different_idx)
        
        return post_a, post_b
    
    def _generate_fine_tuning_pair(self) -> Tuple[PostCandidate, PostCandidate]:
        """Generate pair for fine-tuning around best candidate."""
        
        similarities = cosine_similarity([self.search_region_center], self.all_embeddings)[0]
        closest_indices = np.argsort(similarities)[-10:]
        
        # Exclude current best and recently shown posts
        recent_post_ids = {choice.chosen_post.post_id for choice in self.search_history[-3:]}
        recent_post_ids.update({choice.rejected_post.post_id for choice in self.search_history[-3:]})
        recent_post_ids.add(self.current_best_candidate.post_id)
        
        available_indices = []
        for idx in closest_indices:
            post_id = self.all_ids[idx]
            if post_id not in recent_post_ids:
                available_indices.append(idx)
        
        if len(available_indices) < 2:
            # Fallback: use all closest if we filtered too much
            available_indices = closest_indices[-4:]  # Get more options
        
        # Pick two different posts from available
        idx_a, idx_b = available_indices[-1], available_indices[-2]
        
        post_a = self._create_post_candidate(idx_a)
        post_b = self._create_post_candidate(idx_b)
        
        return post_a, post_b
    
    def _create_post_candidate(self, post_idx: int) -> PostCandidate:
        """Create a PostCandidate from post index."""
        
        embedding = self.all_embeddings[post_idx]
        topic_projection = embedding @ self.topic_projection_matrix
        residual_vector = self.calculator.get_residual_embedding(embedding)
        post_id = self.all_ids[post_idx]
        
        return PostCandidate(
            post_id=post_id,
            embedding=embedding,
            topic_projection=topic_projection,
            residual_vector=residual_vector,
            content=f"Post {post_id}",
            confidence_score=0.0
        )
    
    def _find_closest_post(self, target_embedding: np.ndarray) -> PostCandidate:
        """Find the closest actual post to target embedding."""
        
        similarities = cosine_similarity([target_embedding], self.all_embeddings)[0]
        closest_idx = np.argmax(similarities)
        
        return self._create_post_candidate(closest_idx)
    
    def _calculate_post_confidence(self, post: PostCandidate) -> float:
        """Calculate confidence that this post is optimal."""
        
        if len(self.search_history) == 0:
            return 0.0
        
        rounds_completed = len(self.search_history)
        base_confidence = min(0.9, rounds_completed * 0.1)
        radius_bonus = (1.0 - self.search_radius) * 0.3
        
        return min(1.0, base_confidence + radius_bonus)
    
    def _analyze_user_choices(self):
        """Analyze all user choices to understand preferences."""
        
        print(f"\n📊 COMPREHENSIVE CHOICE ANALYSIS")
        print(f"=" * 50)
        
        if not self.search_history:
            print("  └─ No choices to analyze")
            return
        
        # Analyze topic preferences
        topic_movements = []
        tone_movements = []
        choice_patterns = {'text_posts': 0, 'video_posts': 0, 'mixed': 0}
        
        for choice in self.search_history:
            chosen_id = choice.chosen_post.post_id
            rejected_id = choice.rejected_post.post_id
            
            # Track content type preferences
            if chosen_id.endswith('_video'):
                choice_patterns['video_posts'] += 1
            else:
                choice_patterns['text_posts'] += 1
            
            # Calculate topic and tone movements
            try:
                chosen_idx = self.all_ids.index(chosen_id)
                rejected_idx = self.all_ids.index(rejected_id)
                
                chosen_embedding = self.all_embeddings[chosen_idx]
                rejected_embedding = self.all_embeddings[rejected_idx]
                
                # Topic space movement
                chosen_topic = chosen_embedding @ self.topic_projection_matrix
                rejected_topic = rejected_embedding @ self.topic_projection_matrix
                topic_movement = np.linalg.norm(chosen_topic - rejected_topic)
                topic_movements.append(topic_movement)
                
                # Tone space movement (residuals)
                chosen_residual = self.calculator.get_residual_embedding(chosen_embedding)
                rejected_residual = self.calculator.get_residual_embedding(rejected_embedding)
                tone_movement = np.linalg.norm(chosen_residual - rejected_residual)
                tone_movements.append(tone_movement)
                
            except Exception as e:
                continue
        
        # Summary statistics
        print(f"🎯 PREFERENCE PATTERNS:")
        print(f"  ├─ Text posts chosen: {choice_patterns['text_posts']}")
        print(f"  ├─ Video posts chosen: {choice_patterns['video_posts']}")
        print(f"  └─ Content preference: {'Text-focused' if choice_patterns['text_posts'] > choice_patterns['video_posts'] else 'Video-focused' if choice_patterns['video_posts'] > choice_patterns['text_posts'] else 'Balanced'}")
        
        if topic_movements:
            print(f"\n📚 TOPIC SPACE ANALYSIS:")
            print(f"  ├─ Average topic movement: {np.mean(topic_movements):.3f}")
            print(f"  ├─ Topic exploration range: {np.max(topic_movements):.3f}")
            print(f"  └─ Topic consistency: {'High' if np.std(topic_movements) < 0.5 else 'Moderate' if np.std(topic_movements) < 1.0 else 'Low'}")
        
        if tone_movements:
            print(f"\n🎵 TONE SPACE ANALYSIS:")
            print(f"  ├─ Average tone movement: {np.mean(tone_movements):.3f}")
            print(f"  ├─ Tone exploration range: {np.max(tone_movements):.3f}")
            print(f"  └─ Tone consistency: {'High' if np.std(tone_movements) < 0.3 else 'Moderate' if np.std(tone_movements) < 0.6 else 'Low'}")
        
        # Store analysis for content generation
        self.choice_analysis = {
            'content_preferences': choice_patterns,
            'topic_movements': topic_movements,
            'tone_movements': tone_movements,
            'total_choices': len(self.search_history),
            'exploration_breadth': {
                'topic': np.std(topic_movements) if topic_movements else 0,
                'tone': np.std(tone_movements) if tone_movements else 0
            }
        }
    
    def _generate_ideal_post_recommendation(self, optimal_post: PostCandidate):
        """Generate content recommendation based on choice analysis."""
        
        print(f"\n🎨 IDEAL POST GENERATION")
        print(f"=" * 40)
        
        # Get the actual content of the optimal post
        try:
            optimal_idx = self.all_ids.index(optimal_post.post_id)
            optimal_content = self.fetch_post_content_from_supabase(optimal_post.post_id)
            
            print(f"🏆 OPTIMAL POST ANALYSIS:")
            print(f"  ├─ Post ID: {optimal_post.post_id}")
            print(f"  ├─ Content type: {'Video' if optimal_post.post_id.endswith('_video') else 'Text'}")
            
            if optimal_content:
                print(f"  └─ Content preview: \"{optimal_content[:100]}...\"")
            else:
                print(f"  └─ Content: Available in Supabase")
            
        except Exception as e:
            print(f"  └─ Error accessing optimal post: {e}")
        
        # Choice pattern summary
        if hasattr(self, 'choice_analysis'):
            analysis = self.choice_analysis
            
            print(f"\n📊 YOUR CONTENT PROFILE:")
            print(f"  ├─ Preferred format: {'Video content' if analysis['content_preferences']['video_posts'] > analysis['content_preferences']['text_posts'] else 'Text content'}")
            print(f"  ├─ Topic exploration: {analysis['exploration_breadth']['topic']:.3f} (breadth)")
            print(f"  ├─ Tone exploration: {analysis['exploration_breadth']['tone']:.3f} (breadth)")
            print(f"  └─ Decision consistency: {'High' if len(set(choice.chosen_post.post_id for choice in self.search_history[-3:])) == 1 else 'Moderate'}")
        
        # Content generation prompt
        print(f"\n🤖 CONTENT GENERATION PROMPT:")
        print(f"=" * 40)
        print(f"Based on your choices, here's what an LLM could use to generate your ideal post:")
        
        prompt_elements = []
        prompt_elements.append(f"User converged to Post {optimal_post.post_id}")
        
        if hasattr(self, 'choice_analysis'):
            analysis = self.choice_analysis
            if analysis['content_preferences']['video_posts'] > analysis['content_preferences']['text_posts']:
                prompt_elements.append("Prefers video/script content format")
            else:
                prompt_elements.append("Prefers text post format")
            
            if analysis['exploration_breadth']['topic'] > 1.0:
                prompt_elements.append("Explores diverse topics")
            else:
                prompt_elements.append("Focused on specific topic areas")
                
            if analysis['exploration_breadth']['tone'] > 0.5:
                prompt_elements.append("Varies tone/style significantly")
            else:
                prompt_elements.append("Consistent tone/style preference")
        
        # Show example prompt for LLM
        print(f"\n💭 EXAMPLE LLM PROMPT:")
        print(f'"""')
        print(f'Generate a LinkedIn post that matches this user\'s discovered preferences:')
        print(f'')
        for element in prompt_elements:
            print(f'- {element}')
        print(f'')
        print(f'User\'s optimal post example: "{optimal_content[:200] if "optimal_content" in locals() and optimal_content else "See Supabase for full content"}..."')
        print(f'')
        print(f'Create a new post that captures the same topic focus and tone style.')
        print(f'"""')
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"  ├─ Use this analysis to prompt an LLM (GPT-4, Claude, etc.)")
        print(f"  ├─ Generate content that matches discovered preferences")
        print(f"  ├─ Apply tone calibration for final refinement")
        print(f"  └─ Present generated content to user for approval")

def main():
    """Main interactive function."""
    
    try:
        # Initialize the interactive system
        print("📊 Loading embeddings and components...")
        finder = InteractiveOptimalPostFinder()
        
        print(f"\n🚀 Ready to start interactive search!")
        input("Press Enter when ready to begin...")
        
        # Start the interactive search
        finder.start_interactive_search()
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Search interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Initialization error: {e}")
        print(f"Please check that all required files are available:")
        print(f"  ├─ data/embeddings/ugc-posts_COMPLETE_embeddings.h5")
        print(f"  ├─ data/reports/optimal_160_projection_matrix.npy")
        print(f"  └─ data/reports/optimal_130_tone_axes.npy")

if __name__ == "__main__":
    main() 