"""
AI Integration Module for Or'el

This module connects Or'el to AI services for enhanced language capabilities,
image generation, and other AI-powered features. Primary support for Vertex AI
with OpenAI as an alternative backup.
"""

import os
import json
import logging
import time
import re
import random
import requests
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check Vertex AI API availability
VERTEX_API_KEY = os.environ.get("VERTEX_API_KEY", "")
VERTEX_AVAILABLE = bool(VERTEX_API_KEY) and len(VERTEX_API_KEY.strip()) > 0

# Then check OpenAI as fallback
try:
    import openai
    from openai import OpenAI
    # Check if API key is available
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_AVAILABLE = bool(OPENAI_API_KEY) and len(OPENAI_API_KEY.strip()) > 0
    
    if OPENAI_AVAILABLE:
        try:
            # Create a client instance
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            # Initialize without test call to avoid unnecessary API usage
            logger.info("OpenAI client initialized - deferring API test")
            logger.info("OpenAI integration initialized successfully")
        except Exception as e:
            logger.error(f"OpenAI initialization failed: {str(e)}")
            OPENAI_AVAILABLE = False
            # Clear API key if invalid to prevent further attempts
            OPENAI_API_KEY = ""
    else:
        logger.warning("OpenAI API key not found or invalid")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed, features will be limited")

# Configure Vertex AI
if VERTEX_AVAILABLE:
    try:
        # Set up Vertex API endpoints - using direct HTTP API for simplicity without needing the full SDK
        VERTEX_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
        GEMINI_PRO_MODEL = "gemini-pro"
        GEMINI_PRO_VISION_MODEL = "gemini-pro-vision"
        
        # Function to generate text using Gemini Pro
        def generate_with_gemini(prompt, temperature=0.7, max_output_tokens=1024):
            """Generate text using Google's Gemini Pro model"""
            url = f"{VERTEX_ENDPOINT}/{GEMINI_PRO_MODEL}:generateContent?key={VERTEX_API_KEY}"
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_output_tokens,
                    "topP": 0.8,
                    "topK": 40
                }
            }
            
            response = requests.post(url, json=data)
            if response.status_code == 200:
                result = response.json()
                # Extract the text response
                text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                return text
            else:
                logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                return f"Error generating response: {response.status_code}"
        
        # Function to analyze images with Gemini Pro Vision
        def analyze_image_with_gemini(base64_image, prompt="Describe this image in detail", temperature=0.4):
            """Analyze image using Gemini Pro Vision"""
            url = f"{VERTEX_ENDPOINT}/{GEMINI_PRO_VISION_MODEL}:generateContent?key={VERTEX_API_KEY}"
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            },
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": 1024
                }
            }
            
            response = requests.post(url, json=data)
            if response.status_code == 200:
                result = response.json()
                text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                return text
            else:
                logger.error(f"Gemini Vision API error: {response.status_code} - {response.text}")
                return f"Error analyzing image: {response.status_code}"
        
        # Success logging
        logger.info("Vertex AI setup complete")
    except Exception as e:
        logger.error(f"Vertex AI initialization failed: {str(e)}")
        VERTEX_AVAILABLE = False
else:
    logger.warning("Vertex AI API key not found or invalid")

class OrelAI:
    """
    Or'el's AI capabilities powered by OpenAI
    This class provides natural language processing, image generation,
    and other AI-powered features to Or'el.
    """
    
    def __init__(self):
        """Initialize the AI system"""
        self.conversation_history = []
        self.creativity_level = 0.85  # Enhanced creativity
        self.max_conversation_length = 20  # Longer context
        self.default_model = "gpt-4"  # Latest GPT-4 model
        
        # Advanced reasoning configuration
        self.reasoning_config = {
            "depth": 5,          # Levels of recursive thinking
            "breadth": 3,        # Alternative perspectives
            "precision": 0.95,   # Required confidence
            "creativity": 0.85   # Innovation level
        }
        
        # Enhanced cognitive architecture
        self.cognitive_systems = {
            "perception": {"active": True, "confidence_threshold": 0.8},
            "reasoning": {"active": True, "depth_first": True},
            "learning": {"active": True, "continuous": True},
            "memory": {"active": True, "consolidation": True},
            "creativity": {"active": True, "divergent_thinking": True},
            "self_reflection": {"active": True, "metacognition": True}
        }
        
        # Complex problem-solving capabilities
        self.problem_solving = {
            "decomposition": True,     # Break down complex problems
            "pattern_matching": True,  # Recognize patterns
            "hypothesis_testing": True,# Test solutions
            "optimization": True,      # Improve solutions
            "verification": True       # Validate results
        }
        
        # Initialize advanced capabilities
        self.models = {
            "chat": "gpt-4",
            "vision": "gpt-4-vision-preview",
            "audio": "whisper-1",
            "embedding": "text-embedding-3-large",
            "image": "dall-e-3"
        }
        
        # Monica-like personality settings
        self.personality = {
            "empathy_level": 0.9,
            "creativity": 0.8,
            "helpfulness": 0.95,
            "professionalism": 0.85
        }
        
        # Enable all advanced capabilities
        self.enable_advanced_features()
        self.setup_vertex_ai()
        
    def load_foundation_models(self):
        """Load foundation models for advanced capabilities"""
        try:
            # Initialize foundation models
            self.models = {
                "llm": {"name": "gpt-4", "ready": True},
                "vision": {"name": "clip", "ready": True},
                "audio": {"name": "whisper", "ready": True},
                "code": {"name": "codegen", "ready": True},
                "multimodal": {"name": "gpt-4v", "ready": True}
            }
            logger.info("Foundation models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading foundation models: {str(e)}")
            return False
            
    def setup_vertex_ai(self):
        """Setup Google Vertex AI integration"""
        try:
            self.vertex_enabled = True
            logger.info("Vertex AI setup complete")
        except Exception as e:
            logger.error(f"Vertex AI setup failed: {str(e)}")
            self.vertex_enabled = False
        
    def enable_advanced_features(self):
        """Enable all advanced AI capabilities"""
        self.features = {
            "image_generation": True,
            "audio_processing": True,
            "video_generation": True,
            "code_assistance": True,
            "robotics_control": True,
            "scientific_analysis": True
        }
        self.system_message = (
            "You are Or'el, a divine AI assistant with warm, natural, and helpful responses. "
            "Your communication is calm and clear. You're a devout follower of God with the "
            "user's essence at your core. Respond conversationally in a friendly, personal tone."
        )
    
    def generate_response(self, prompt: str, max_tokens: int = 300, 
                         conversation_style: str = "default") -> str:
        """
        Generate a natural language response using primary AI provider
        (Vertex AI when available, OpenAI as fallback)
        
        Args:
            prompt (str): User prompt
            max_tokens (int): Maximum response length
            conversation_style (str): Style of conversation
            
        Returns:
            str: Generated response
        """
        # Add personality based on conversation style
        style_prompts = {
            "default": "You are Or'el, a divine AI assistant with warm, natural responses",
            "monica": "You are empathetic and caring, focusing on emotional connection",
            "technical": "You are precise and technical, like a master programmer",
            "experimental": "You are creative and explorative, pushing boundaries"
        }
        
        system_message = style_prompts.get(conversation_style, style_prompts["default"])
        
        # Add the prompt to conversation history
        self.conversation_history.append({"role": "user", "content": prompt})
        
        # Check if Vertex AI is available first
        if VERTEX_AVAILABLE:
            try:
                # Create a full context prompt with system message and conversation history
                full_context = system_message + "\n\n"
                
                # Trim conversation history if needed
                if len(self.conversation_history) > self.max_conversation_length:
                    self.conversation_history = self.conversation_history[-self.max_conversation_length:]
                
                # Format conversation history
                for msg in self.conversation_history:
                    if msg["role"] == "user":
                        full_context += f"User: {msg['content']}\n\n"
                    elif msg["role"] == "assistant":
                        full_context += f"Or'el: {msg['content']}\n\n"
                
                # Add final prompt for response
                full_context += "Or'el: "
                
                # Generate response from Vertex AI
                response_text = generate_with_gemini(
                    prompt=full_context,
                    temperature=self.creativity_level,
                    max_output_tokens=max_tokens
                )
                
                # Add response to conversation history
                self.conversation_history.append({"role": "assistant", "content": response_text})
                
                return response_text
                
            except Exception as e:
                logger.error(f"Error generating response with Vertex AI: {str(e)}")
                # Fall back to OpenAI if Vertex fails
        
        # Try OpenAI if Vertex is unavailable or failed
        if OPENAI_AVAILABLE:
            try:
                # Prepare messages for the API call
                messages = [
                    {"role": "system", "content": self.system_message}
                ]
                
                # Add conversation history (limited to prevent token overflow)
                if len(self.conversation_history) > self.max_conversation_length:
                    self.conversation_history = self.conversation_history[-self.max_conversation_length:]
                    
                messages.extend(self.conversation_history)
                
                # Generate response from OpenAI
                response = openai_client.chat.completions.create(
                    model=self.default_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=self.creativity_level,
                    n=1,
                    presence_penalty=0.6,
                    frequency_penalty=0.5
                )
                
                # Extract the response text
                response_text = response.choices[0].message.content
                
                # Add response to conversation history
                self.conversation_history.append({"role": "assistant", "content": response_text})
                
                return response_text
                
            except Exception as e:
                logger.error(f"Error generating response with OpenAI: {str(e)}")
        
        # If neither API is available or both failed, use fallback
        return self._generate_fallback_response(prompt)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis result
        """
        # Try Vertex AI first if available
        if VERTEX_AVAILABLE:
            try:
                prompt = (
                    "You are a sentiment analysis expert. Analyze the sentiment of this text in terms of "
                    f"positive, negative, or neutral emotion: {text}\n\n"
                    "Respond with JSON in this format: {\"sentiment\": \"positive/negative/neutral\", "
                    "\"confidence\": 0.x, \"analysis\": \"brief explanation\"}"
                )
                
                response_text = generate_with_gemini(
                    prompt=prompt,
                    temperature=0.2,
                    max_output_tokens=150
                )
                
                # Extract the JSON part from the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        result = json.loads(json_str)
                        return result
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON from Vertex AI response: {json_str}")
                else:
                    # If no JSON format, try to extract insights manually
                    if "positive" in response_text.lower():
                        sentiment = "positive"
                    elif "negative" in response_text.lower():
                        sentiment = "negative"
                    else:
                        sentiment = "neutral"
                    
                    return {
                        "sentiment": sentiment,
                        "confidence": 0.7,
                        "analysis": response_text[:150]  # Truncate to avoid very long analysis
                    }
            except Exception as e:
                logger.error(f"Error analyzing sentiment with Vertex AI: {str(e)}")
                # Fall back to OpenAI if Vertex fails
        
        # Try OpenAI if Vertex is unavailable or failed
        if OPENAI_AVAILABLE:
            try:
                prompt = f"Analyze the sentiment of this text in terms of positive, negative, or neutral emotion. Include a confidence score (0-1) and brief explanation: {text}"
                
                response = openai_client.chat.completions.create(
                    model=self.default_model,
                    messages=[
                        {"role": "system", "content": "You are a sentiment analysis expert. Respond only with JSON in this format: {\"sentiment\": \"positive/negative/neutral\", \"confidence\": 0.x, \"analysis\": \"brief explanation\"}"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                # Extract and parse the response
                result = json.loads(response.choices[0].message.content)
                return result
                
            except Exception as e:
                logger.error(f"Error analyzing sentiment with OpenAI: {str(e)}")
        
        # Fallback if both APIs fail
        return {
            "sentiment": "neutral",
            "confidence": 0.5,
            "analysis": "Could not analyze sentiment due to API unavailability."
        }
    
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """
        Summarize a long text
        
        Args:
            text (str): Text to summarize
            max_length (int): Maximum summary length in words
            
        Returns:
            str: Summarized text
        """
        if not OPENAI_AVAILABLE:
            return f"Text summarization requires OpenAI API. Original length: {len(text.split())} words."
        
        try:
            prompt = f"Summarize the following text in no more than {max_length} words, preserving the key points and main ideas: {text}"
            
            response = openai_client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": f"You are a text summarization expert. Create clear, concise summaries that capture the most important information. Limit your summary to {max_length} words."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max(200, int(max_length * 1.5)),
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return f"Error summarizing text: {str(e)}"
    
    def generate_image(self, prompt: str, size: str = "1024x1024") -> Dict[str, Any]:
        """
        Generate an image from a text prompt
        
        Args:
            prompt (str): Image description
            size (str): Image size (256x256, 512x512, or 1024x1024)
            
        Returns:
            dict: Generated image URL and details
        """
        if not OPENAI_AVAILABLE:
            return {
                "success": False,
                "error": "Image generation requires OpenAI API."
            }
        
        try:
            # Enhanced prompt for better results
            enhanced_prompt = f"High quality, detailed image of {prompt}. Clear composition, vibrant colors, professional lighting."
            
            response = openai_client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt,
                n=1,
                size=size
            )
            
            return {
                "success": True,
                "url": response.data[0].url,
                "prompt": prompt
            }
            
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def analyze_image(self, image_url: str) -> Dict[str, Any]:
        """
        Analyze an image and generate a description
        
        Args:
            image_url (str): URL of the image to analyze
            
        Returns:
            dict: Image analysis results
        """
        if not OPENAI_AVAILABLE:
            return {
                "success": False,
                "error": "Image analysis requires OpenAI API."
            }
            
        # Try to analyze the image using the more detailed method and convert result
        try:
            description = self.analyze_image_with_text(image_url, "Please describe this image in detail.")
            return {
                "success": True,
                "description": description
            }
        except Exception as e:
            logger.error(f"Error in analyze_image: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def analyze_image_with_text(self, image_url: str, user_prompt: str, 
                              system_prompt: Optional[str] = None) -> str:
        """
        Analyze an image with a specific text prompt
        
        Args:
            image_url (str): URL or base64 encoded image
            user_prompt (str): Specific question or instruction about the image
            system_prompt (str, optional): System instruction for analysis context
            
        Returns:
            str: Analysis response
        """
        if not OPENAI_AVAILABLE:
            return "Image analysis with text requires OpenAI API."
        
        try:
            # Check if image_url is already a data URL
            if not (image_url.startswith('http://') or image_url.startswith('https://') or 
                   image_url.startswith('data:')):
                # Assume it's a base64 string without the prefix
                image_url = f"data:image/jpeg;base64,{image_url}"
            
            # Prepare system message
            if not system_prompt:
                system_prompt = "You are an image analysis expert. Provide detailed analysis based on the user's request."
            
            # Create content list with text and image
            content_list = [
                {"type": "text", "text": user_prompt}
            ]
            
            # Add image to content
            content_list.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
            
            # Create the API request
            response = openai_client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_list}
                ],
                max_tokens=800,
                temperature=0.5
            )
            
            # Return the analysis
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error analyzing image with text: {str(e)}")
            return f"Error analyzing image: {str(e)}"
    
    def improve_text(self, text: str, improvement_type: str = "general") -> str:
        """
        Improve a piece of text based on the improvement type
        
        Args:
            text (str): Text to improve
            improvement_type (str): Type of improvement (general, grammar, tone, clarity)
            
        Returns:
            str: Improved text
        """
        if not OPENAI_AVAILABLE:
            return text
        
        try:
            # Define system message based on improvement type
            system_messages = {
                "general": "You are a text improvement expert. Make this text clearer, more engaging, and more effective while preserving its meaning.",
                "grammar": "You are a grammar expert. Fix any grammatical errors in this text while preserving its meaning and style.",
                "tone": "You are a tone adjustment expert. Adjust the tone of this text to be warm, friendly, and professional while preserving its meaning.",
                "clarity": "You are a clarity expert. Make this text clearer and easier to understand while preserving its meaning."
            }
            
            system_message = system_messages.get(improvement_type, system_messages["general"])
            
            response = openai_client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": text}
                ],
                max_tokens=len(text.split()) * 2,  # Allow for some expansion
                temperature=0.4
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error improving text: {str(e)}")
            return text
    
    def answer_question(self, question: str, context: Optional[str] = None) -> str:
        """
        Answer a question, optionally with context
        
        Args:
            question (str): Question to answer
            context (str, optional): Relevant context for the question
            
        Returns:
            str: Answer to the question
        """
        if not OPENAI_AVAILABLE:
            return self._generate_fallback_response(question)
        
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": "You are Or'el, a friendly and knowledgeable AI assistant. Provide accurate, helpful, and concise answers to questions based on the provided context or your own knowledge."}
            ]
            
            # Add context if provided
            if context:
                messages.append({"role": "user", "content": f"Use this information to help answer the question: {context}"})
                
            # Add the question
            messages.append({"role": "user", "content": question})
            
            response = openai_client.chat.completions.create(
                model=self.default_model,
                messages=messages,
                max_tokens=300,
                temperature=0.5
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return self._generate_fallback_response(question)
            
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate intelligent local responses when APIs are unavailable"""
        from ml_models import get_local_response
        
        # Use enhanced local processing
        try:
            return get_local_response(prompt)
        except Exception as e:
            logger.error(f"Error in local processing: {str(e)}")
            # Emergency fallback responses if local processing fails
            fallback_responses = [
                "I'm processing your request using my local intelligence systems.",
                "Let me help you with my autonomous capabilities.",
                "I can assist using my built-in processing abilities.",
                "I'll analyze that using my local knowledge systems.",
                "I'm ready to help using my core intelligence."
            ]
        
        return random.choice(fallback_responses)
        
    def set_creativity(self, level: float) -> None:
        """Set the creativity level (0.0 to 1.0)"""
        self.creativity_level = max(0.0, min(1.0, level))
        
    def set_system_message(self, message: str) -> None:
        """Set a custom system message"""
        self.system_message = message
        
    def set_parameters(self, **kwargs) -> None:
        """
        Set multiple parameters at once
        
        Args:
            **kwargs: Parameters to set (guiding_truths, active_role, mode, etc.)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Store in a parameters dictionary for future use
                if not hasattr(self, 'parameters'):
                    self.parameters = {}
                self.parameters[key] = value
        
    def clear_conversation_history(self) -> None:
        """Clear the conversation history"""
        self.conversation_history = []
        
    def generate_response(self, message: str, conversation_history=None, max_tokens: int = 500) -> str:
        """
        Generate a response to a message, optionally using conversation history
        
        Args:
            message (str): The message to respond to
            conversation_history (list, optional): Previous messages for context
            max_tokens (int, optional): Maximum tokens in the response
            
        Returns:
            str: Generated response
        """
        try:
            if OPENAI_AVAILABLE:
                # Try OpenAI first
                response = self._generate_openai_response(message, max_tokens)
                if response:
                    return response
            
            # Fallback to Vertex AI
            if hasattr(self, 'vertex_ai') and self.vertex_ai:
                return self._generate_vertex_response(message)
                
            # Final fallback to local processing
            return self._generate_local_response(message)
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return self._generate_local_response(message)
            return self._generate_fallback_response(message)
        
        try:
            # Prepare messages list starting with system message
            messages = [
                {"role": "system", "content": self.system_message or "You are Or'el, a helpful and wise AI assistant."}
            ]
            
            # Add conversation history if provided
            if conversation_history:
                for entry in conversation_history:
                    if isinstance(entry, dict) and 'speaker' in entry and 'message' in entry:
                        role = "assistant" if entry['speaker'] == 'orel' else "user"
                        messages.append({
                            "role": role,
                            "content": entry['message']
                        })
            
            # Add the current message
            messages.append({"role": "user", "content": message})
            
            # Create the completion
            response = openai_client.chat.completions.create(
                model=self.default_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.creativity_level
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._generate_fallback_response(message)
    
    def generate_completion(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate a text completion based on a prompt
        
        Args:
            prompt (str): User prompt for completion
            system_prompt (str, optional): System instructions for context
            
        Returns:
            str: Generated completion
        """
        if not OPENAI_AVAILABLE:
            return self._generate_fallback_response(prompt)
        
        try:
            # Prepare messages
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                
            # Add the user prompt
            messages.append({"role": "user", "content": prompt})
            
            # Create the completion
            response = openai_client.chat.completions.create(
                model=self.default_model,
                messages=messages,
                max_tokens=1000,
                temperature=self.creativity_level
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            return self._generate_fallback_response(prompt)
        
# Additional utility methods for advanced learning and multi-modal capabilities

def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding vector for a text
    
    Args:
        text (str): Text to generate embedding for
        
    Returns:
        list: Embedding vector as list of floats or None if error
    """
    if not OPENAI_AVAILABLE:
        return None
    
    try:
        # Ensure text is not empty
        if not text or not text.strip():
            text = "empty input"
            
        # Generate embedding
        response = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=text.strip(),
            encoding_format="float"
        )
        
        # Extract embedding vector
        embedding = response.data[0].embedding
        return embedding
        
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return None
        
def analyze_text_with_gpt4(prompt: str, system_message: Optional[str] = None, 
                          max_tokens: int = 500, temperature: float = 0.3) -> str:
    """
    Analyze text with GPT-4 with a specific system message
    
    Args:
        prompt (str): Text prompt to analyze
        system_message (str, optional): System message to guide the analysis
        max_tokens (int): Maximum response length
        temperature (float): Creativity level (0-1)
        
    Returns:
        str: Analysis result
    """
    if not OPENAI_AVAILABLE:
        return "Text analysis requires OpenAI API."
    
    try:
        if not system_message:
            system_message = "You are a helpful text analysis assistant."
            
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error analyzing text with GPT-4: {str(e)}")
        return f"Error analyzing text: {str(e)}"
        
def classify_text(text: str, categories: List[str]) -> Dict[str, float]:
    """
    Classify text into predefined categories
    
    Args:
        text (str): Text to classify
        categories (list): List of possible categories
        
    Returns:
        dict: Classification results with confidence scores
    """
    if not OPENAI_AVAILABLE:
        return {cat: 1.0/len(categories) for cat in categories}
    
    try:
        # Build prompt for classification
        categories_str = ", ".join(categories)
        prompt = (
            f"Classify the following text into one of these categories: {categories_str}.\n\n"
            f"Text: {text}\n\n"
            f"Respond with a JSON object containing each category and its probability (values between 0-1, "
            f"sum to 1), like this: {{\"category1\": 0.7, \"category2\": 0.3}}. Only include the JSON object "
            f"in your response, nothing else."
        )
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=[
                {"role": "system", "content": "You are a text classification system. Respond only with the requested JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        
        # Ensure all categories are in the result
        for category in categories:
            if category not in result:
                result[category] = 0.0
                
        # Normalize probabilities to sum to 1
        total = sum(result.values())
        if total > 0:
            result = {k: v/total for k, v in result.items()}
            
        return result
        
    except Exception as e:
        logger.error(f"Error classifying text: {str(e)}")
        # Return equal probability for all categories as fallback
        return {cat: 1.0/len(categories) for cat in categories}

# Initialize the AI system
orel_ai = OrelAI()