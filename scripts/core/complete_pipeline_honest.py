#!/usr/bin/env python3
"""
Complete Seamless Pipeline: Interactive Search → Choice Analysis → LLM Generation
Uses HONEST tone alignments instead of fake style parameters
"""

import json
import os
import tempfile
from pathlib import Path
from scripts.core.interactive_post_finder import InteractiveOptimalPostFinder

class SeamlessTonePipeline:
    """Complete pipeline with automatic data handoff between stages."""
    
    def __init__(self):
        self.session_data = {}
        self.temp_dir = Path(tempfile.mkdtemp(prefix="tone_pipeline_"))
        print(f"📁 Session data directory: {self.temp_dir}")
    
    def stage_1_interactive_search(self) -> dict:
        """Stage 1: Run interactive search and capture results."""
        
        print("🎯 STAGE 1: INTERACTIVE CONVERGENT SEARCH")
        print("=" * 60)
        
        # Initialize interactive finder
        finder = InteractiveOptimalPostFinder()
        
        # Start interactive search
        finder.start_interactive_search()
        
        # After convergence, extract session data
        session_data = {
            'optimal_post_id': finder.current_best_candidate.post_id,
            'final_confidence': finder.current_best_candidate.confidence_score,
            'total_rounds': len(finder.search_history),
            'choice_history': [
                {
                    'round': choice.round_number,
                    'chosen_post_id': choice.chosen_post.post_id,
                    'rejected_post_id': choice.rejected_post.post_id,
                    'strategy': choice.search_strategy,
                    'confidence_gain': choice.confidence_gain,
                    'choice_distance': choice.choice_distance,
                    'search_center_movement': choice.search_center_movement
                }
                for choice in finder.search_history
            ],
            'choice_analysis': getattr(finder, 'choice_analysis', {}),
            'user_preference_vector': finder.current_best_candidate.residual_vector.tolist()
        }
        
        # Save session data for next stage
        session_file = self.temp_dir / "stage1_results.json"
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"\n✅ STAGE 1 COMPLETE")
        print(f"  ├─ Optimal post: {session_data['optimal_post_id']}")
        print(f"  ├─ Confidence: {session_data['final_confidence']:.3f}")
        print(f"  └─ Data saved: {session_file}")
        
        self.session_data['stage1'] = session_data
        return session_data
    
    def stage_2_semantic_extraction(self, stage1_data: dict) -> dict:
        """Stage 2: Extract HONEST semantic tone data from choice patterns."""
        
        print(f"\n🔍 STAGE 2: HONEST SEMANTIC TONE EXTRACTION")
        print("=" * 60)
        
        # Load the interactive finder to access the calculator
        finder = InteractiveOptimalPostFinder()
        
        # Reconstruct user preference vector from stage 1
        import numpy as np
        user_preference_vector = np.array(stage1_data['user_preference_vector'])
        
        # Calculate HONEST tone axis alignments (no fake interpretations)
        tone_alignments = {}
        if hasattr(finder.calculator, 'tone_axis_vectors') and finder.calculator.tone_axis_vectors:
            for axis_name, axis_vector in finder.calculator.tone_axis_vectors.items():
                alignment = float(np.dot(user_preference_vector, axis_vector))
                tone_alignments[axis_name] = alignment
        
        # Extract HONEST semantic parameters (no fake style parameters)
        semantic_data = {
            'preference_vector_norm': float(np.linalg.norm(user_preference_vector)),
            'tone_axis_alignments': tone_alignments,
            'top_aligned_axes': sorted(tone_alignments.items(), key=lambda x: abs(x[1]), reverse=True)[:10],
            'choice_patterns': stage1_data['choice_analysis'],
            'optimal_post_content': self._fetch_optimal_post_content(stage1_data['optimal_post_id']),
            'content_preferences': {
                'text_vs_video': stage1_data['choice_analysis'].get('content_preferences', {}),
                'exploration_breadth': stage1_data['choice_analysis'].get('exploration_breadth', {}),
                'decision_consistency': stage1_data['choice_analysis'].get('decision_consistency', {})
            }
        }
        
        # Save semantic data
        semantic_file = self.temp_dir / "stage2_semantic_data.json"
        with open(semantic_file, 'w') as f:
            json.dump(semantic_data, f, indent=2)
        
        print(f"✅ HONEST SEMANTIC EXTRACTION COMPLETE")
        print(f"  ├─ Tone axes analyzed: {len(tone_alignments)}")
        print(f"  ├─ Top alignment: {semantic_data['top_aligned_axes'][0] if semantic_data['top_aligned_axes'] else 'None'}")
        print(f"  ├─ Content preferences: {list(semantic_data['content_preferences'].keys())}")
        print(f"  └─ Data saved: {semantic_file}")
        
        self.session_data['stage2'] = semantic_data
        return semantic_data
    
    def stage_3_llm_generation(self, stage1_data: dict, stage2_data: dict, content_topic: str = "", user_goals: str = "") -> str:
        """Stage 3: Generate content using LLM with HONEST semantic tone context."""
        
        print(f"\n🤖 STAGE 3: LLM CONTENT GENERATION")
        print("=" * 60)
        
        # Build comprehensive LLM prompt with HONEST data
        llm_prompt = self._build_honest_llm_prompt(stage1_data, stage2_data, content_topic, user_goals)
        
        # Save prompt for inspection
        prompt_file = self.temp_dir / "stage3_llm_prompt.txt"
        with open(prompt_file, 'w') as f:
            f.write(llm_prompt)
        
        print(f"📝 HONEST LLM PROMPT GENERATED:")
        print(f"  ├─ Includes choice journey: ✅")
        print(f"  ├─ Includes raw tone alignments: ✅")
        print(f"  ├─ Includes content preferences: ✅")
        print(f"  ├─ Includes optimal post example: ✅")
        print(f"  └─ Prompt saved: {prompt_file}")
        
        # Generate content with OpenAI
        generated_content = self._generate_with_openai(llm_prompt)
        
        # Save generated content
        content_file = self.temp_dir / "stage3_generated_content.txt"
        with open(content_file, 'w') as f:
            f.write(generated_content)
        
        print(f"\n✅ CONTENT GENERATION COMPLETE")
        print(f"  └─ Content saved: {content_file}")
        
        return generated_content
    
    def _build_honest_llm_prompt(self, stage1_data: dict, stage2_data: dict, content_topic: str, user_goals: str) -> str:
        """Build comprehensive LLM prompt with HONEST context (no fake interpretations)."""
        
        prompt = f"""# Generate Ideal LinkedIn Post

## User Discovery Results
Through convergent search in residualized tone space, we discovered:

### Optimal Post Convergence
- Converged to: Post {stage1_data['optimal_post_id']}
- Final confidence: {stage1_data['final_confidence']:.3f}
- Total rounds: {stage1_data['total_rounds']}

### Choice Journey Analysis
{chr(10).join(f"Round {choice['round']}: Chose {choice['chosen_post_id']} ({choice['strategy']})" for choice in stage1_data['choice_history'])}

### Content Preferences (Discovered)
- Format preference: {'Text-focused' if stage1_data['choice_analysis'].get('content_preferences', {}).get('text_posts', 0) > stage1_data['choice_analysis'].get('content_preferences', {}).get('video_posts', 0) else 'Video-focused'}
- Topic exploration: {stage1_data['choice_analysis'].get('exploration_breadth', {}).get('topic', 0):.3f} ({'Diverse' if stage1_data['choice_analysis'].get('exploration_breadth', {}).get('topic', 0) > 1.0 else 'Focused'})
- Tone consistency: {stage1_data['choice_analysis'].get('exploration_breadth', {}).get('tone', 0):.3f} ({'Variable' if stage1_data['choice_analysis'].get('exploration_breadth', {}).get('tone', 0) > 0.5 else 'Consistent'})

### Raw Tone Axis Alignments (Mathematical Dimensions)
These represent the user's alignment with discovered semantic dimensions:
{chr(10).join(f"- {axis}: {score:.3f}" for axis, score in stage2_data['top_aligned_axes'][:5])}

Note: These are mathematical dimensions discovered through unsupervised learning. 
The LLM should interpret what these alignments likely represent based on the 
user's choice patterns and content preferences.

### Optimal Post Example
"{stage2_data.get('optimal_post_content', 'Content not available')[:300]}..."

## Generation Task
Content topic: {content_topic or 'User-specified topic'}
User goals: {user_goals or 'Professional engagement and thought leadership'}

Create a NEW LinkedIn post that:
1. Matches the discovered topic preferences from the choice journey
2. Aligns with the discovered tone dimensions (interpret the mathematical alignments)
3. Follows the preferred format (text-focused based on choices)
4. Incorporates the user's goals and specified topic

Generate content that feels authentic to this user's discovered voice and preferences.
The tone axis alignments represent mathematical dimensions of the user's preferences - 
interpret these to understand their communication style.
"""
        
        return prompt
    
    def _fetch_optimal_post_content(self, post_id: str) -> str:
        """Fetch the actual content of the optimal post."""
        
        try:
            # Use the same Supabase logic as interactive finder
            finder = InteractiveOptimalPostFinder()
            content = finder.fetch_post_content_from_supabase(post_id)
            return content if content else "Content not available"
        except:
            return "Content not available"
    
    def _generate_with_openai(self, prompt: str) -> str:
        """Generate content using OpenAI API."""
        
        try:
            import openai
            
            openai_key = os.getenv('OPENAI_API_KEY')
            if not openai_key:
                return "❌ OpenAI API key not found. Add to .env file."
            
            client = openai.OpenAI(api_key=openai_key)
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert LinkedIn content creator who understands tone preferences discovered through residualized tone space analysis. You can interpret mathematical tone alignments to understand user communication style."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Content generation failed: {e}"
    
    def run_complete_pipeline(self, content_topic: str = "", user_goals: str = ""):
        """Run the complete seamless pipeline."""
        
        print("🚀 COMPLETE SEAMLESS TONE PIPELINE (HONEST VERSION)")
        print("=" * 70)
        print("Automatic flow: Search → Analysis → Extraction → Generation")
        print("Uses HONEST tone alignments (no fake style parameters)")
        print("=" * 70)
        
        # Stage 1: Interactive search (user-driven)
        stage1_data = self.stage_1_interactive_search()
        
        # Automatic transition to Stage 2
        print(f"\n🔄 AUTOMATIC TRANSITION TO STAGE 2...")
        input("Press Enter to continue to semantic extraction...")
        
        # Stage 2: Semantic extraction (automatic)
        stage2_data = self.stage_2_semantic_extraction(stage1_data)
        
        # Automatic transition to Stage 3
        print(f"\n🔄 AUTOMATIC TRANSITION TO STAGE 3...")
        
        # Get user input for content generation
        if not content_topic:
            content_topic = input("Enter content topic (or press Enter for general): ").strip()
        if not user_goals:
            user_goals = input("Enter your goals (or press Enter for default): ").strip()
        
        # Stage 3: LLM generation (automatic)
        generated_content = self.stage_3_llm_generation(stage1_data, stage2_data, content_topic, user_goals)
        
        # Final results
        self._show_final_results(stage1_data, stage2_data, generated_content)
        
        return {
            'stage1': stage1_data,
            'stage2': stage2_data,
            'generated_content': generated_content,
            'session_files': list(self.temp_dir.glob("*"))
        }
    
    def _show_final_results(self, stage1_data: dict, stage2_data: dict, generated_content: str):
        """Show comprehensive final results."""
        
        print(f"\n🎉 COMPLETE PIPELINE RESULTS (HONEST VERSION)")
        print("=" * 60)
        
        print(f"🎯 DISCOVERY SUMMARY:")
        print(f"  ├─ Optimal post found: {stage1_data['optimal_post_id']}")
        print(f"  ├─ Convergence confidence: {stage1_data['final_confidence']:.3f}")
        print(f"  ├─ Search rounds: {stage1_data['total_rounds']}")
        print(f"  └─ Content preference: {'Text' if stage1_data['choice_analysis'].get('content_preferences', {}).get('text_posts', 0) > 0 else 'Mixed'}")
        
        print(f"\n🎨 GENERATED IDEAL POST:")
        print("=" * 40)
        print(generated_content)
        
        print(f"\n📁 SESSION DATA SAVED:")
        for file_path in self.temp_dir.glob("*"):
            print(f"  ├─ {file_path.name}")
        
        print(f"\n🔄 NEXT STEPS:")
        print(f"  ├─ Review generated content above")
        print(f"  ├─ Edit/refine as needed")
        print(f"  ├─ Use tone calibration API for further refinement")
        print(f"  └─ Publish optimized content")

def main():
    """Run the complete seamless pipeline."""
    
    try:
        # Initialize pipeline
        pipeline = SeamlessTonePipeline()
        
        # Run complete flow
        results = pipeline.run_complete_pipeline()
        
        print(f"\n✅ COMPLETE PIPELINE FINISHED!")
        
    except KeyboardInterrupt:
        print(f"\n👋 Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")

if __name__ == "__main__":
    main()
