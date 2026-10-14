#!/usr/bin/env python3
"""
FastAPI endpoints for Tone Calibration System
Provides the backend for the "Face ID for Tone" experience
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utilities.tone_calibration_system_fixed import ToneCalibrationSystem
import uvicorn

# Global system instance
calibration_system = None

app = FastAPI(
    title="Tone Calibration API",
    description="Face ID for Tone - Learn user preferences through A/B comparisons",
    version="1.0.0"
)

# Pydantic models
class ContentPairResponse(BaseModel):
    pair_id: str
    content_a: Dict[str, str]  # {id, text}
    content_b: Dict[str, str]  # {id, text}
    strategy_used: str
    choices_made: int
    calibration_progress: float

class UserChoiceRequest(BaseModel):
    pair_id: str
    chosen_content_id: str
    confidence: Optional[float] = 1.0

class CalibrationStatusResponse(BaseModel):
    choices_made: int
    calibration_confidence: float
    is_calibrated: bool
    next_strategy: str
    progress_percentage: float

class ContentScoreRequest(BaseModel):
    content_text: str

class ContentScoreResponse(BaseModel):
    tone_score: float
    confidence: float
    is_calibrated: bool

class ToneProfileRequest(BaseModel):
    user_id: str
    conversation_history: Optional[List[str]] = []
    user_goals: Optional[str] = ""
    context_data: Optional[Dict] = {}

class ToneProfileResponse(BaseModel):
    tone_profile: Dict[str, Any]
    llm_prompt_context: str
    semantic_tone_data: Dict[str, Any]
    is_calibrated: bool

class LLMPromptRequest(BaseModel):
    user_id: str
    content_topic: str
    conversation_history: Optional[List[str]] = []
    user_goals: Optional[str] = ""
    additional_context: Optional[Dict] = {}

class LLMPromptResponse(BaseModel):
    structured_prompt: str
    tone_context: Dict[str, Any]
    topic_guidance: str
    style_parameters: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Initialize the calibration system on startup."""
    global calibration_system
    print("🚀 Initializing Tone Calibration System...")
    calibration_system = ToneCalibrationSystem()
    print("✅ System initialized!")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Tone Calibration API", "status": "ready"}

@app.post("/calibration/start")
async def start_calibration():
    """Start a new calibration session."""
    # In production, you'd create per-user sessions
    global calibration_system
    
    # Reset system for new user (in production, create new instance per user)
    calibration_system.user_choices = []
    calibration_system.user_preference_vector = None
    calibration_system.calibration_confidence = 0.0
    
    return {"message": "Calibration started", "session_id": str(uuid.uuid4())}

@app.get("/calibration/next-pair")
async def get_next_pair() -> ContentPairResponse:
    """Get the next pair of content for user comparison."""
    
    if calibration_system is None:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    # Check if already calibrated
    if calibration_system.is_calibrated():
        raise HTTPException(status_code=400, detail="User already calibrated")
    
    # Generate pair using optimal strategy
    status = calibration_system.get_calibration_status()
    strategy = status['next_strategy']
    pair = calibration_system.generate_calibration_pair(strategy)
    
    # Calculate progress
    progress = min(100, (len(calibration_system.user_choices) / 10.0) * 100)
    
    return ContentPairResponse(
        pair_id=pair.pair_id,
        content_a={"id": pair.content_a_id, "text": pair.content_a_text[:200] + "..."},
        content_b={"id": pair.content_b_id, "text": pair.content_b_text[:200] + "..."},
        strategy_used=strategy,
        choices_made=len(calibration_system.user_choices),
        calibration_progress=progress
    )

@app.post("/calibration/record-choice")
async def record_choice(choice: UserChoiceRequest):
    """Record user's choice between two content pieces."""
    
    if calibration_system is None:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    # Find the pair (in production, store pairs in session)
    # For demo, we'll create a simple lookup
    try:
        # This is simplified - in production you'd store pairs properly
        pair = calibration_system.generate_calibration_pair("random")  # Placeholder
        pair.pair_id = choice.pair_id
        
        # Record the choice
        calibration_system.record_user_choice(
            pair, 
            choice.chosen_content_id, 
            choice.confidence
        )
        
        # Get updated status
        status = calibration_system.get_calibration_status()
        
        return {
            "message": "Choice recorded",
            "choices_made": status['choices_made'],
            "calibration_confidence": status['calibration_confidence'],
            "is_calibrated": status['is_calibrated']
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error recording choice: {str(e)}")

@app.get("/calibration/status")
async def get_calibration_status() -> CalibrationStatusResponse:
    """Get current calibration status."""
    
    if calibration_system is None:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    status = calibration_system.get_calibration_status()
    
    # Calculate progress percentage
    progress = min(100, (status['choices_made'] / 10.0) * 100)
    
    return CalibrationStatusResponse(
        choices_made=status['choices_made'],
        calibration_confidence=status['calibration_confidence'],
        is_calibrated=status['is_calibrated'],
        next_strategy=status['next_strategy'],
        progress_percentage=progress
    )

@app.post("/content/score")
async def score_content(request: ContentScoreRequest) -> ContentScoreResponse:
    """Score content with user's calibrated preferences."""
    
    if calibration_system is None:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    if not calibration_system.is_calibrated():
        raise HTTPException(status_code=400, detail="User not calibrated yet")
    
    try:
        # In production, you'd generate embedding for the text
        # For demo, we'll use a random embedding
        import numpy as np
        dummy_embedding = np.random.randn(1536).astype(np.float32)
        
        score = calibration_system.score_content(dummy_embedding)
        
        return ContentScoreResponse(
            tone_score=score,
            confidence=calibration_system.calibration_confidence,
            is_calibrated=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring content: {str(e)}")

@app.post("/calibration/save/{user_id}")
async def save_calibration(user_id: str):
    """Save user's calibration data."""
    
    if calibration_system is None:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        calibration_file = calibration_system.save_calibration(user_id)
        return {
            "message": "Calibration saved",
            "user_id": user_id,
            "file": str(calibration_file)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving calibration: {str(e)}")

@app.post("/tone/extract-profile")
async def extract_tone_profile(request: ToneProfileRequest) -> ToneProfileResponse:
    """Extract comprehensive tone profile for LLM prompting."""
    
    if calibration_system is None:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    if not calibration_system.is_calibrated():
        raise HTTPException(status_code=400, detail="User not calibrated yet")
    
    try:
        # Extract semantic tone data from calibration
        tone_profile = {
            'calibration_confidence': calibration_system.calibration_confidence,
            'total_choices': len(calibration_system.user_choices),
            'preference_strength': float(np.linalg.norm(calibration_system.user_preference_vector)) if calibration_system.user_preference_vector is not None else 0.0
        }
        
        # Analyze choice patterns
        choice_analysis = _analyze_choice_patterns(calibration_system.user_choices)
        
        # Generate LLM context
        llm_context = _generate_llm_tone_context(
            tone_profile, 
            choice_analysis, 
            request.conversation_history,
            request.user_goals
        )
        
        # Extract semantic tone data
        semantic_data = _extract_semantic_tone_data(calibration_system)
        
        return ToneProfileResponse(
            tone_profile=tone_profile,
            llm_prompt_context=llm_context,
            semantic_tone_data=semantic_data,
            is_calibrated=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting tone profile: {str(e)}")

@app.post("/llm/generate-prompt")
async def generate_llm_prompt(request: LLMPromptRequest) -> LLMPromptResponse:
    """Generate structured LLM prompt with tone context."""
    
    if calibration_system is None:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    if not calibration_system.is_calibrated():
        raise HTTPException(status_code=400, detail="User not calibrated yet")
    
    try:
        # Extract tone context
        tone_context = _extract_semantic_tone_data(calibration_system)
        
        # Generate structured prompt
        structured_prompt = f"""
# Content Generation Task

## User Tone Profile
{_generate_llm_tone_context({}, {}, request.conversation_history, request.user_goals)}

## Content Requirements
- Topic: {request.content_topic}
- User Goals: {request.user_goals or 'Professional engagement'}
- Context: {request.additional_context}

## Tone Calibration Data
{json.dumps(tone_context, indent=2)}

## Instructions
Generate LinkedIn content that matches the user's discovered tone preferences while addressing the specified topic and goals.
"""
        
        # Style parameters for fine-tuning
        style_params = {
            'formality_level': _estimate_formality_level(calibration_system),
            'enthusiasm_level': _estimate_enthusiasm_level(calibration_system),
            'technical_depth': _estimate_technical_depth(calibration_system),
            'personal_voice': _estimate_personal_voice(calibration_system)
        }
        
        return LLMPromptResponse(
            structured_prompt=structured_prompt,
            tone_context=tone_context,
            topic_guidance=request.content_topic,
            style_parameters=style_params
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating prompt: {str(e)}")

# Helper functions for tone analysis
def _analyze_choice_patterns(user_choices):
    """Analyze patterns in user choices."""
    
    patterns = {
        'consistency_score': 0.0,
        'preference_strength': 0.0,
        'choice_confidence_avg': 0.0
    }
    
    if user_choices:
        confidences = [choice.confidence for choice in user_choices if choice.confidence]
        patterns['choice_confidence_avg'] = np.mean(confidences) if confidences else 0.0
    
    return patterns

def _generate_llm_tone_context(tone_profile, choice_analysis, conversation_history, user_goals):
    """Generate LLM-ready tone context."""
    
    context = f"""
User's Tone Calibration Results:
- Calibration confidence: {tone_profile.get('calibration_confidence', 0):.3f}
- Total choices analyzed: {tone_profile.get('total_choices', 0)}
- Preference strength: {tone_profile.get('preference_strength', 0):.3f}

User Goals: {user_goals or 'Not specified'}

Conversation Context:
{chr(10).join(f'- {msg}' for msg in conversation_history[-5:]) if conversation_history else '- No conversation history'}

Tone Guidance: Use the calibrated tone preferences to match the user's discovered style.
"""
    
    return context

def _extract_semantic_tone_data(calibration_system):
    """Extract semantic tone data for LLM context."""
    
    if not calibration_system.user_preference_vector is not None:
        return {"error": "No preference vector available"}
    
    # Calculate tone axis alignments
    tone_alignments = {}
    if hasattr(calibration_system.calculator, 'tone_axis_vectors') and calibration_system.calculator.tone_axis_vectors:
        for axis_name, axis_vector in calibration_system.calculator.tone_axis_vectors.items():
            alignment = float(np.dot(calibration_system.user_preference_vector, axis_vector))
            tone_alignments[axis_name] = alignment
    
    return {
        'preference_vector_norm': float(np.linalg.norm(calibration_system.user_preference_vector)),
        'tone_axis_alignments': tone_alignments,
        'top_aligned_axes': sorted(tone_alignments.items(), key=lambda x: abs(x[1]), reverse=True)[:5] if tone_alignments else []
    }

def _estimate_formality_level(calibration_system):
    """Estimate user's formality preference."""
    # This would analyze the residual patterns to estimate formality
    return "moderate"  # Placeholder

def _estimate_enthusiasm_level(calibration_system):
    """Estimate user's enthusiasm preference."""
    # This would analyze the residual patterns to estimate enthusiasm
    return "moderate"  # Placeholder

def _estimate_technical_depth(calibration_system):
    """Estimate user's technical depth preference."""
    # This would analyze the topic patterns to estimate technical level
    return "moderate"  # Placeholder

def _estimate_personal_voice(calibration_system):
    """Estimate user's personal voice preference."""
    # This would analyze the tone patterns to estimate personal vs professional
    return "professional"  # Placeholder

# Development endpoints
@app.get("/debug/system-info")
async def get_system_info():
    """Get system information for debugging."""
    
    if calibration_system is None:
        return {"status": "not_initialized"}
    
    return {
        "embeddings_loaded": len(calibration_system.embeddings),
        "content_ids_loaded": len(calibration_system.content_ids),
        "residual_vectors_shape": calibration_system.residual_vectors.shape,
        "tone_axes_loaded": len(calibration_system.calculator.get_tone_axis_names()),
        "current_user_choices": len(calibration_system.user_choices)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
