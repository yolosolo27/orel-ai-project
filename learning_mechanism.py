"""
Advanced Learning Mechanism for Or'el

This module enables Or'el to continuously learn and improve by:
1. Tracking user interactions and preferences
2. Analyzing feedback patterns
3. Fine-tuning responses based on historical interactions
4. Adapting to user's communication style
5. Implementing various AI model types for specialized learning

The learning system integrates:
- Foundation models (GPT-4o, Claude, etc.)
- Memory systems for long and short-term memory
- User preference tracking
- Specialized domain adaptation
- Continuous improvement mechanisms
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
from sqlalchemy import desc

from app import db
from models import UserInteraction, LearningSession, ModelPerformance
from openai_utils import generate_embedding, analyze_text_with_gpt4, classify_text, OPENAI_AVAILABLE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningMechanism:
    """Advanced learning system for Or'el that tracks interactions, user preferences,
    and continuously improves responses through various AI models"""
    
    def __init__(self):
        """Initialize the learning mechanism with models and tracking systems"""
        self.version = "1.0.0"
        self.learning_rate = 0.01
        self.memory_decay = 0.95  # Rate at which older memories fade
        self.learning_initialized = False
        self.models = {}
        self.specialized_domains = [
            "programming", "science", "mathematics", "creative", 
            "personal_assistance", "research", "education"
        ]
        self.active_session = None
        self.performance_metrics = {
            "response_quality": 0.0,
            "learning_progress": 0.0,
            "adaptation_speed": 0.0
        }
        
        try:
            # Initialize model connections
            self._initialize_models()
            self.learning_initialized = True
            logger.info(f"Learning Mechanism v{self.version} initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing learning mechanism: {e}")
            
    def _initialize_models(self):
        """Initialize connections to various AI models"""
        # Primary foundation model - GPT-4o
        self.models["foundation"] = {
            "name": "gpt-4o", 
            "provider": "openai",
            "capabilities": ["text", "code", "reasoning", "multimodal"],
            "performance": 0.95
        }
        
        # Domain-specific models
        self.models["code"] = {
            "name": "gpt-4o",  # Using GPT-4o for code generation
            "provider": "openai",
            "capabilities": ["code_generation", "debugging", "explanation"],
            "performance": 0.92
        }
        
        self.models["vision"] = {
            "name": "gpt-4o",  # GPT-4o handles vision tasks
            "provider": "openai",
            "capabilities": ["image_analysis", "object_detection", "visual_reasoning"],
            "performance": 0.90
        }
        
        self.models["embedding"] = {
            "name": "text-embedding-3-large",
            "provider": "openai",
            "capabilities": ["text_embedding", "semantic_similarity"],
            "performance": 0.93
        }
        
        logger.info(f"Initialized {len(self.models)} AI models for learning")
        
    def start_learning_session(self, user_id: int, context: Optional[Dict[str, Any]] = None) -> str:
        """Start a new learning session for tracking interactions and improvements"""
        timestamp = datetime.now()
        session_id = f"session_{timestamp.strftime('%Y%m%d%H%M%S')}_{user_id}"
        
        # Create session record in database
        new_session = LearningSession(
            session_id=session_id,
            user_id=user_id,
            start_time=timestamp,
            context=json.dumps(context) if context else None,
            active=True
        )
        
        try:
            db.session.add(new_session)
            db.session.commit()
            self.active_session = session_id
            logger.info(f"Started learning session: {session_id}")
            return session_id
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to start learning session: {e}")
            return None
        
    def end_learning_session(self, session_id: str) -> bool:
        """End an active learning session and process collected data"""
        try:
            session = LearningSession.query.filter_by(session_id=session_id).first()
            if not session:
                logger.warning(f"Session {session_id} not found")
                return False
                
            session.end_time = datetime.now()
            session.active = False
            session.performance_data = json.dumps(self.performance_metrics)
            
            # Process the session data to extract learning
            interactions = UserInteraction.query.filter_by(session_id=session_id).all()
            
            if interactions:
                # Analyze the session for insights
                self._process_session_insights(interactions)
                
            db.session.commit()
            
            if session_id == self.active_session:
                self.active_session = None
                
            logger.info(f"Ended learning session: {session_id}")
            return True
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to end learning session: {e}")
            return False
    
    def track_interaction(self, user_id: int, input_text: str, response_text: str, 
                         interaction_type: str = "conversation", 
                         metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Track a user interaction for learning purposes
        
        Args:
            user_id: ID of the user
            input_text: User's input text
            response_text: Or'el's response text
            interaction_type: Type of interaction (conversation, code, image, etc.)
            metadata: Additional context or data about the interaction
            
        Returns:
            interaction_id: ID of the tracked interaction
        """
        try:
            # Generate embedding for semantic analysis
            embedding = generate_embedding(input_text)
            
            # Create interaction record
            interaction = UserInteraction(
                user_id=user_id,
                session_id=self.active_session,
                input_text=input_text,
                response_text=response_text,
                interaction_type=interaction_type,
                timestamp=datetime.now(),
                embedding=json.dumps(embedding) if embedding else None,
                interaction_data=json.dumps(metadata) if metadata else None
            )
            
            db.session.add(interaction)
            db.session.commit()
            
            # Update performance metrics
            self._update_performance_metrics(interaction.id)
            
            logger.info(f"Tracked interaction {interaction.id} for learning")
            return interaction.id
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to track interaction: {e}")
            return None
    
    def get_learning_insights(self, user_id: int) -> Dict[str, Any]:
        """
        Get learning insights for a specific user
        
        Args:
            user_id: ID of the user
            
        Returns:
            insights: Dict containing learning insights for the user
        """
        try:
            # Get recent interactions
            recent_interactions = UserInteraction.query.filter_by(user_id=user_id)\
                .order_by(desc(UserInteraction.timestamp)).limit(100).all()
            
            if not recent_interactions:
                return {
                    "status": "success",
                    "user_id": user_id,
                    "insights": "Not enough interactions to generate insights"
                }
            
            # Extract common topics and interests
            topics = self._extract_topics([i.input_text for i in recent_interactions])
            
            # Analyze interaction patterns
            patterns = self._analyze_interaction_patterns(recent_interactions)
            
            # Calculate response quality over time
            quality_trend = self._calculate_quality_trend(user_id)
            
            return {
                "status": "success",
                "user_id": user_id,
                "interaction_count": len(recent_interactions),
                "common_topics": topics,
                "interaction_patterns": patterns,
                "quality_trend": quality_trend,
                "learning_progress": self.performance_metrics["learning_progress"]
            }
        except Exception as e:
            logger.error(f"Failed to get learning insights: {e}")
            return {
                "status": "error",
                "message": f"Failed to get learning insights: {str(e)}"
            }
    
    def improve_from_feedback(self, interaction_id: int, feedback_score: float, 
                             feedback_text: Optional[str] = None) -> bool:
        """
        Improve responses based on explicit user feedback
        
        Args:
            interaction_id: ID of the interaction receiving feedback
            feedback_score: Numerical score (0-1) representing feedback quality
            feedback_text: Optional text explaining the feedback
            
        Returns:
            success: Whether the feedback was successfully processed
        """
        try:
            interaction = UserInteraction.query.get(interaction_id)
            if not interaction:
                logger.warning(f"Interaction {interaction_id} not found")
                return False
            
            # Update interaction with feedback
            interaction.feedback_score = feedback_score
            interaction.feedback_text = feedback_text
            interaction.feedback_timestamp = datetime.now()
            
            # Calculate adjustment to learning parameters
            adjustment = (feedback_score - 0.5) * 2 * self.learning_rate
            
            # Adjust learning parameters based on feedback
            self.learning_rate = max(0.001, min(0.1, self.learning_rate + adjustment))
            
            # Record model performance
            model_name = "gpt-4o"  # Default model
            if interaction.interaction_data:
                metadata = json.loads(interaction.interaction_data)
                if "model" in metadata:
                    model_name = metadata["model"]
            
            # Add performance record
            performance = ModelPerformance(
                model_name=model_name,
                interaction_id=interaction_id,
                score=feedback_score,
                timestamp=datetime.now()
            )
            
            db.session.add(performance)
            db.session.commit()
            
            logger.info(f"Processed feedback for interaction {interaction_id}")
            return True
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to process feedback: {e}")
            return False
    
    def get_personalized_parameters(self, user_id: int) -> Dict[str, Any]:
        """
        Get personalized parameters for a specific user based on learning
        
        Args:
            user_id: ID of the user
            
        Returns:
            parameters: Dict containing personalized parameters
        """
        try:
            # Get user's interaction history
            interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            
            if not interactions or len(interactions) < 5:
                # Not enough data for personalization
                return self._get_default_parameters()
            
            # Calculate personalized parameters
            verbosity = self._calculate_verbosity(interactions)
            technical_level = self._calculate_technical_level(interactions)
            creative_level = self._calculate_creative_level(interactions)
            preferred_domains = self._identify_preferred_domains(interactions)
            
            return {
                "status": "success",
                "user_id": user_id,
                "parameters": {
                    "verbosity": verbosity,
                    "technical_level": technical_level,
                    "creative_level": creative_level,
                    "preferred_domains": preferred_domains,
                    "interaction_count": len(interactions)
                }
            }
        except Exception as e:
            logger.error(f"Failed to get personalized parameters: {e}")
            return self._get_default_parameters()
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters when personalization is not possible"""
        return {
            "status": "default",
            "parameters": {
                "verbosity": 0.5,  # Medium verbosity
                "technical_level": 0.5,  # Medium technical level
                "creative_level": 0.5,  # Medium creativity
                "preferred_domains": ["general"],
                "interaction_count": 0
            }
        }
    
    def _calculate_verbosity(self, interactions: List[UserInteraction]) -> float:
        """Calculate user's preferred verbosity based on their interactions"""
        # Analyze length of successful interactions (high feedback)
        positive_interactions = [i for i in interactions if getattr(i, "feedback_score", 0) > 0.7]
        if not positive_interactions:
            return 0.5  # Default to medium verbosity
        
        # Calculate average response length for positive feedback
        avg_length = sum(len(i.response_text) for i in positive_interactions) / len(positive_interactions)
        
        # Normalize to 0-1 scale (0 = concise, 1 = verbose)
        # Assuming average lengths between 100-1000 characters
        verbosity = min(1.0, max(0.0, (avg_length - 100) / 900))
        
        return verbosity
    
    def _calculate_technical_level(self, interactions: List[UserInteraction]) -> float:
        """Calculate user's technical sophistication level"""
        # Extract all user inputs
        inputs = [i.input_text for i in interactions]
        
        # Combine inputs for analysis
        combined_text = " ".join(inputs)
        
        # Count technical indicators (code snippets, technical terms)
        code_indicators = ["```", "function", "class", "def ", "import ", "const ", "var ", "let "]
        technical_terms = ["algorithm", "framework", "library", "middleware", "API", "function", "database"]
        
        code_count = sum(combined_text.count(indicator) for indicator in code_indicators)
        term_count = sum(combined_text.count(term) for term in technical_terms)
        
        # Calculate normalized technical score
        total_words = len(combined_text.split())
        if total_words == 0:
            return 0.5
            
        technical_score = min(1.0, (code_count * 3 + term_count) / (total_words / 10))
        
        return technical_score
    
    def _calculate_creative_level(self, interactions: List[UserInteraction]) -> float:
        """Calculate user's preference for creative vs. factual responses"""
        # Count creative keywords in positive feedback
        creative_keywords = ["creative", "imagine", "story", "generate", "design", "art"]
        factual_keywords = ["facts", "accurate", "precisely", "specifically", "exactly"]
        
        positive_feedback = [i.feedback_text for i in interactions 
                            if hasattr(i, "feedback_text") and i.feedback_text 
                            and getattr(i, "feedback_score", 0) > 0.7]
        
        if not positive_feedback:
            return 0.5  # Default to balanced
            
        combined_feedback = " ".join(positive_feedback).lower()
        
        creative_count = sum(combined_feedback.count(kw) for kw in creative_keywords)
        factual_count = sum(combined_feedback.count(kw) for kw in factual_keywords)
        
        total = creative_count + factual_count
        if total == 0:
            return 0.5
            
        creative_score = creative_count / total if total > 0 else 0.5
        
        return creative_score
    
    def _identify_preferred_domains(self, interactions: List[UserInteraction]) -> List[str]:
        """Identify user's preferred knowledge domains"""
        domains = {domain: 0 for domain in self.specialized_domains}
        domains["general"] = 0  # Add general domain
        
        # Classify each interaction into domains
        for interaction in interactions:
            if not interaction.input_text:
                continue
                
            # Simple keyword-based classification
            text = interaction.input_text.lower()
            
            # Count domain indicators
            if any(kw in text for kw in ["code", "program", "function", "bug", "error"]):
                domains["programming"] += 1
                
            if any(kw in text for kw in ["science", "experiment", "theory", "hypothesis"]):
                domains["science"] += 1
                
            if any(kw in text for kw in ["math", "equation", "calculate", "formula"]):
                domains["mathematics"] += 1
                
            if any(kw in text for kw in ["create", "design", "imagine", "story", "art"]):
                domains["creative"] += 1
                
            if any(kw in text for kw in ["reminder", "schedule", "organize", "plan"]):
                domains["personal_assistance"] += 1
                
            if any(kw in text for kw in ["research", "study", "analyze", "investigate"]):
                domains["research"] += 1
                
            if any(kw in text for kw in ["learn", "teach", "explain", "understand"]):
                domains["education"] += 1
                
            # Increment general counter for all interactions
            domains["general"] += 0.5
        
        # Get top 3 domains (or fewer if not enough data)
        sorted_domains = sorted(domains.items(), key=lambda x: x[1], reverse=True)
        top_domains = [domain for domain, count in sorted_domains if count > 0][:3]
        
        # Always include general domain if no others are found
        if not top_domains:
            top_domains = ["general"]
            
        return top_domains
    
    def _extract_topics(self, texts: List[str]) -> List[str]:
        """Extract common topics from a list of texts"""
        # Combine texts for analysis
        combined_text = " ".join(texts)
        
        # If text is too short, return empty list
        if len(combined_text) < 50:
            return []
            
        # Use GPT-4 to extract topics
        prompt = (
            "Extract the top 5 topics or subjects from the following text. "
            "Respond with only a JSON array of strings, each representing a topic.\n\n"
            f"Text: {combined_text[:2000]}..."  # Limit text length for API
        )
        
        try:
            result = analyze_text_with_gpt4(prompt)
            topics = json.loads(result) if result else []
            return topics[:5]  # Ensure we return at most 5 topics
        except Exception as e:
            logger.error(f"Failed to extract topics: {e}")
            # Fallback to keyword frequency
            words = combined_text.lower().split()
            common_words = set(["the", "and", "a", "to", "of", "is", "in", "that", "it", "for"])
            word_counts = {}
            for word in words:
                if word not in common_words and len(word) > 3:
                    word_counts[word] = word_counts.get(word, 0) + 1
                    
            # Get top 5 words by frequency
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            return [word for word, _ in top_words]
    
    def _analyze_interaction_patterns(self, interactions: List[UserInteraction]) -> Dict[str, Any]:
        """Analyze patterns in user interactions"""
        if not interactions:
            return {"patterns": []}
            
        # Analyze time patterns
        timestamps = [i.timestamp for i in interactions]
        time_diffs = []
        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i-1]).total_seconds()
            if diff < 24 * 60 * 60:  # Only count differences less than a day
                time_diffs.append(diff)
                
        avg_response_time = sum(time_diffs) / len(time_diffs) if time_diffs else 0
        
        # Analyze interaction types
        types = {}
        for i in interactions:
            types[i.interaction_type] = types.get(i.interaction_type, 0) + 1
            
        primary_type = max(types.items(), key=lambda x: x[1])[0] if types else "conversation"
        
        # Average length of user inputs
        avg_input_length = sum(len(i.input_text) for i in interactions) / len(interactions)
        
        return {
            "avg_response_time_seconds": avg_response_time,
            "primary_interaction_type": primary_type,
            "avg_input_length": avg_input_length,
            "interaction_types": types
        }
    
    def _calculate_quality_trend(self, user_id: int) -> List[float]:
        """Calculate trend in response quality over time"""
        # Get feedback scores in chronological order
        scores = UserInteraction.query.with_entities(UserInteraction.feedback_score)\
            .filter(UserInteraction.user_id == user_id)\
            .filter(UserInteraction.feedback_score != None)\
            .order_by(UserInteraction.timestamp).all()
            
        scores = [s[0] for s in scores if s[0] is not None]
        
        if not scores:
            return [0.5]  # Default if no scores
            
        # Apply moving average to smooth the trend
        window_size = min(5, len(scores))
        smoothed = []
        for i in range(len(scores)):
            start = max(0, i - window_size + 1)
            smoothed.append(sum(scores[start:i+1]) / (i - start + 1))
            
        # Return last 10 points for trend visualization
        return smoothed[-10:]
    
    def _update_performance_metrics(self, interaction_id: int) -> None:
        """Update performance metrics based on recent interactions"""
        # Get recent performance records
        recent_performance = ModelPerformance.query.order_by(
            desc(ModelPerformance.timestamp)).limit(20).all()
            
        if recent_performance:
            # Calculate average score
            avg_score = sum(p.score for p in recent_performance) / len(recent_performance)
            
            # Update quality metric - weighted average with previous value
            self.performance_metrics["response_quality"] = (
                self.performance_metrics["response_quality"] * 0.7 + avg_score * 0.3
            )
            
            # Update learning progress - how much improvement over time
            first_five = [p.score for p in recent_performance[-5:]]
            last_five = [p.score for p in recent_performance[:5]]
            
            if first_five and last_five:
                avg_first = sum(first_five) / len(first_five)
                avg_last = sum(last_five) / len(last_five)
                improvement = max(0, avg_last - avg_first)
                
                self.performance_metrics["learning_progress"] = (
                    self.performance_metrics["learning_progress"] * 0.8 + improvement * 0.2
                )
    
    def _process_session_insights(self, interactions: List[UserInteraction]) -> None:
        """Process insights from a completed learning session"""
        if not interactions:
            return
            
        # Calculate overall session statistics
        feedback_scores = [i.feedback_score for i in interactions if i.feedback_score is not None]
        
        if feedback_scores:
            avg_score = sum(feedback_scores) / len(feedback_scores)
            
            # Check if feedback improved over the session
            if len(feedback_scores) >= 3:
                first_third = feedback_scores[:len(feedback_scores)//3]
                last_third = feedback_scores[-len(feedback_scores)//3:]
                
                first_avg = sum(first_third) / len(first_third)
                last_avg = sum(last_third) / len(last_third)
                
                improvement = last_avg - first_avg
                
                # Update adaptation speed metric
                if improvement > 0:
                    self.performance_metrics["adaptation_speed"] = (
                        self.performance_metrics["adaptation_speed"] * 0.7 + improvement * 0.3
                    )
        
        # All updates to the learning mechanism from session insights
        logger.info(f"Processed insights from session with {len(interactions)} interactions")


class ModelRegistry:
    """Registry for managing AI models and their capabilities"""
    
    def __init__(self):
        """Initialize the model registry"""
        self.models = {}
        self.capabilities = {}
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize the available AI models"""
        # Base LLM models
        self.register_model(
            "gpt-4o",
            "openai",
            ["text", "chat", "code", "reasoning", "multimodal"],
            {"temperature": 0.7, "max_tokens": 4000}
        )
        
        # Embedding models
        self.register_model(
            "text-embedding-3-large",
            "openai",
            ["embedding", "similarity", "clustering"],
            {"dimensions": 3072}
        )
        
        # Image generation models
        self.register_model(
            "dall-e-3",
            "openai",
            ["image_generation", "creative"],
            {"quality": "hd", "style": "vivid"}
        )
        
        # Audio and speech models
        self.register_model(
            "whisper-1",
            "openai",
            ["speech_to_text", "transcription", "translation"],
            {"response_format": "text"}
        )
        
        # Index capabilities
        self._index_capabilities()
        
        logger.info(f"Model registry initialized with {len(self.models)} models")
        
    def register_model(self, name, provider, capabilities, params=None):
        """Register a model with its capabilities"""
        self.models[name] = {
            "provider": provider,
            "capabilities": capabilities,
            "params": params or {},
            "registered_at": datetime.now()
        }
        
    def _index_capabilities(self):
        """Create an index of capabilities to models"""
        self.capabilities = {}
        for model_name, model_info in self.models.items():
            for capability in model_info["capabilities"]:
                if capability not in self.capabilities:
                    self.capabilities[capability] = []
                self.capabilities[capability].append(model_name)
                
    def get_model_for_capability(self, capability):
        """Find the best model for a given capability"""
        if capability not in self.capabilities:
            return None
            
        models = self.capabilities[capability]
        if not models:
            return None
            
        # For now, just return the first model with this capability
        # Future: could implement more sophisticated selection
        return models[0]
        
    def get_model_info(self, model_name):
        """Get information about a specific model"""
        return self.models.get(model_name)


# Initialize the learning mechanism as a singleton
learning_mechanism = LearningMechanism()

# Initialize the model registry as a singleton
model_registry = ModelRegistry()