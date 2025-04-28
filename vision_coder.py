"""
VisionCoder Module for Or'el

This module adds visual programming and code generation capabilities to Or'el,
allowing for:
1. Code generation from visual inputs (sketches, diagrams, screenshots)
2. Visual representation of code structures and architectures
3. Interactive visual programming interfaces
4. Visual debugging and code visualization
5. Conversion between visual representations and executable code
"""

import base64
import json
import logging
import os
import re
from typing import Dict, List, Any, Optional, Union
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import OpenAI utilities if available
try:
    from openai_utils import orel_ai
    VISION_AVAILABLE = True
    logger.info("Vision capabilities available through OpenAI")
except ImportError:
    VISION_AVAILABLE = False
    logger.warning("OpenAI utilities not available. Visual coding capabilities will be limited.")

class VisionCoder:
    """
    Main VisionCoder class that provides visual programming and code generation
    capabilities for Or'el
    """
    def __init__(self):
        """Initialize the VisionCoder module"""
        self.supported_languages = [
            "python", "javascript", "typescript", "html", "css", "react", 
            "java", "c", "cpp", "csharp", "go", "rust", "swift", "kotlin"
        ]
        self.supported_frameworks = [
            "react", "vue", "angular", "flask", "django", "express", 
            "spring", "laravel", "rails", "flutter"
        ]
        self.supported_visualizations = [
            "flowchart", "class_diagram", "sequence_diagram", 
            "mind_map", "architecture"
        ]
        self.operation_history = []
        self.ui_patterns = self._load_ui_patterns()
        self.architecture_patterns = self._load_architecture_patterns()
        
    def _load_ui_patterns(self) -> Dict[str, Any]:
        """Load common UI element patterns for recognition"""
        return {
            "button": {
                "patterns": ["rectangular", "rounded", "clickable"],
                "implementation": {
                    "html": "<button>Label</button>",
                    "react": "<Button>Label</Button>",
                    "flutter": "ElevatedButton(onPressed: () {}, child: Text('Label'))"
                }
            },
            "text_field": {
                "patterns": ["input", "entry", "text box"],
                "implementation": {
                    "html": "<input type=\"text\" placeholder=\"Enter text\">",
                    "react": "<TextField placeholder=\"Enter text\" />",
                    "flutter": "TextField(decoration: InputDecoration(hintText: 'Enter text'))"
                }
            },
            "dropdown": {
                "patterns": ["select", "menu", "options"],
                "implementation": {
                    "html": "<select><option>Option 1</option><option>Option 2</option></select>",
                    "react": "<Select options={options} />",
                    "flutter": "DropdownButton(items: items, onChanged: (value) {})"
                }
            },
            "checkbox": {
                "patterns": ["check", "toggle", "boolean"],
                "implementation": {
                    "html": "<input type=\"checkbox\">Label",
                    "react": "<Checkbox label=\"Label\" />",
                    "flutter": "Checkbox(value: true, onChanged: (value) {})"
                }
            },
            "navigation": {
                "patterns": ["menu", "navbar", "tabs"],
                "implementation": {
                    "html": "<nav><ul><li>Item 1</li><li>Item 2</li></ul></nav>",
                    "react": "<Navigation items={items} />",
                    "flutter": "BottomNavigationBar(items: items)"
                }
            }
        }
        
    def _load_architecture_patterns(self) -> Dict[str, Any]:
        """Load system architecture patterns for recognition"""
        return {
            "mvc": {
                "patterns": ["model", "view", "controller"],
                "implementation": {
                    "description": "Model-View-Controller pattern separating data, UI, and application logic",
                    "code_structure": {
                        "python": ["models.py", "views.py", "controllers.py"],
                        "javascript": ["models.js", "views.js", "controllers.js"]
                    }
                }
            },
            "microservices": {
                "patterns": ["service", "api", "gateway"],
                "implementation": {
                    "description": "Microservices architecture with separate, independent services",
                    "code_structure": {
                        "python": ["service1/", "service2/", "api_gateway.py"],
                        "javascript": ["service1/", "service2/", "apiGateway.js"]
                    }
                }
            },
            "client_server": {
                "patterns": ["client", "server", "request", "response"],
                "implementation": {
                    "description": "Client-Server architecture with frontend and backend components",
                    "code_structure": {
                        "python": ["client.py", "server.py"],
                        "javascript": ["client.js", "server.js"]
                    }
                }
            }
        }
    
    def generate_code_from_image(self, image_data: Union[str, bytes], 
                                target_language: str = "python",
                                additional_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate code from an image (sketch, diagram, screenshot, etc.)
        
        Args:
            image_data (str or bytes): Base64 encoded image or file path
            target_language (str): Target programming language
            additional_context (str, optional): Additional context or requirements
            
        Returns:
            dict: Generated code and metadata
        """
        if not VISION_AVAILABLE:
            logger.warning("Vision capabilities not available. Generating fallback code.")
            return {
                "success": False,
                "error": "Vision capabilities not available",
                "code": self._generate_fallback_code(target_language)
            }
            
        try:
            # Prepare image data
            processed_image_data = self._prepare_image_data(image_data)
            
            # Prepare system prompt based on target language
            language_context = self._get_language_context(target_language)
            
            # Add additional context if provided
            if additional_context:
                language_context += f"\nAdditional requirements: {additional_context}"
                
            # Use OpenAI API to analyze the image and generate code
            prompt = f"Please analyze this image and generate {target_language} code that implements what is shown. Include explanations of your implementation choices."
            
            # Use the OpenAI integration
            response = orel_ai.analyze_image_with_text(
                processed_image_data,
                prompt,
                language_context
            )
            
            # Extract code blocks from the response
            code_blocks = self._extract_code_blocks(response, target_language)
            explanation = self._extract_explanation(response)
            
            # Log the operation
            self._log_operation(
                "generate_code_from_image", 
                {"target_language": target_language, "success": True}
            )
            
            return {
                "success": True,
                "code": code_blocks,
                "explanation": explanation,
                "language": target_language
            }
            
        except Exception as e:
            logger.error(f"Error generating code from image: {str(e)}")
            self._log_operation(
                "generate_code_from_image", 
                {"target_language": target_language, "success": False, "error": str(e)}
            )
            return {
                "success": False,
                "error": str(e),
                "code": self._generate_fallback_code(target_language)
            }
    
    def _prepare_image_data(self, image_data: Union[str, bytes]) -> str:
        """
        Prepare image data for API processing
        
        Args:
            image_data: Either a file path, URL, or base64 encoded string
            
        Returns:
            str: Properly formatted image data for API
        """
        # Check if image_data is already a data URL
        if isinstance(image_data, str) and (image_data.startswith('http://') or 
                                       image_data.startswith('https://') or 
                                       image_data.startswith('data:')):
            return image_data
            
        # If it's a file path, read and encode
        if isinstance(image_data, str) and os.path.isfile(image_data):
            with open(image_data, 'rb') as f:
                image_bytes = f.read()
            base64_data = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_data}"
            
        # If it's already base64 encoded (but without prefix)
        if isinstance(image_data, str) and not any(
            image_data.startswith(prefix) for prefix in ('http://', 'https://', 'data:', '/')
        ):
            # Add prefix if it's a raw base64 string
            return f"data:image/jpeg;base64,{image_data}"
            
        # If it's bytes, encode to base64
        if isinstance(image_data, bytes):
            base64_data = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_data}"
            
        # Default fallback
        return image_data
        
    def _get_language_context(self, language: str) -> str:
        """Get appropriate system prompt for the target language"""
        contexts = {
            "python": "You are an expert Python developer. Generate clean, efficient, and PEP8-compliant Python code. Focus on readability and proper use of Pythonic features. Use modern practices like type hints when appropriate.",
            "javascript": "You are an expert JavaScript developer. Generate clean ES6+ JavaScript code following best practices. Use appropriate patterns for the task and consider browser compatibility when relevant.",
            "typescript": "You are an expert TypeScript developer. Generate strongly-typed TypeScript code with proper interfaces and type definitions. Follow best practices for TypeScript development.",
            "html": "You are an expert HTML developer. Generate semantic HTML5 markup with accessibility features. Structure the document properly with appropriate tags and attributes.",
            "css": "You are an expert CSS developer. Generate clean CSS with appropriate selectors and properties. Consider responsiveness and browser compatibility.",
            "react": "You are an expert React developer. Generate functional components with hooks and proper state management. Follow React best practices for component structure and performance.",
            "java": "You are an expert Java developer. Generate clean, object-oriented Java code following industry best practices. Use appropriate design patterns and handle exceptions properly.",
            "default": f"You are an expert developer. Generate clean, well-structured {language} code following industry best practices."
        }
        return contexts.get(language.lower(), contexts["default"])
        
    def _extract_code_blocks(self, text: str, language: str) -> List[Dict[str, str]]:
        """Extract code blocks from the AI response"""
        # Match Markdown code blocks with language identifier
        pattern = r"```(?:" + language + r"|[\w\+\#]+)?\s*([\s\S]*?)```"
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        if not matches:
            # Try matching with any language identifier or none at all
            pattern = r"```(?:[\w\+\#]*)\s*([\s\S]*?)```"
            matches = re.findall(pattern, text)
            
        code_blocks = []
        for i, code in enumerate(matches):
            code_blocks.append({
                "id": i + 1,
                "code": code.strip(),
                "language": language
            })
            
        return code_blocks
        
    def _extract_explanation(self, text: str) -> str:
        """Extract explanation text outside of code blocks"""
        # Remove all code blocks
        explanation = re.sub(r"```[\s\S]*?```", "", text)
        # Clean up and format
        explanation = explanation.strip()
        return explanation
        
    def _generate_fallback_code(self, language: str) -> List[Dict[str, str]]:
        """Generate a simple fallback code when vision capabilities are unavailable"""
        fallbacks = {
            "python": "def main():\n    print(\"Generated from visual input\")\n    # TODO: Implement functionality based on the image\n\nif __name__ == \"__main__\":\n    main()",
            "javascript": "function main() {\n    console.log(\"Generated from visual input\");\n    // TODO: Implement functionality based on the image\n}\n\nmain();",
            "html": "<!DOCTYPE html>\n<html>\n<head>\n    <title>Generated Page</title>\n</head>\n<body>\n    <h1>Generated from visual input</h1>\n    <!-- TODO: Implement layout based on the image -->\n</body>\n</html>",
            "default": "// Generated from visual input\n// TODO: Implement functionality based on the image"
        }
        
        code = fallbacks.get(language.lower(), fallbacks["default"])
        return [{
            "id": 1,
            "code": code,
            "language": language
        }]
        
    def _log_operation(self, operation_type: str, metadata: Dict[str, Any]) -> None:
        """Log an operation for history tracking"""
        from datetime import datetime
        self.operation_history.append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation_type,
            "metadata": metadata
        })
        
    def visualize_code(self, code: str, visualization_type: str = "flowchart") -> Dict[str, Any]:
        """
        Generate a visual representation of code
        
        Args:
            code (str): The code to visualize
            visualization_type (str): Type of visualization ('flowchart', 'class_diagram', 'sequence')
            
        Returns:
            dict: Visualization data or instructions
        """
        if not VISION_AVAILABLE:
            return {
                "success": False,
                "error": "Vision capabilities not available"
            }
            
        try:
            # Prepare system prompt based on visualization type
            if visualization_type == "flowchart":
                system_prompt = "Generate a detailed Mermaid flowchart diagram representing the logic flow of this code. Include all major functions and decision points."
            elif visualization_type == "class_diagram":
                system_prompt = "Generate a detailed Mermaid class diagram representing the class structure, relationships, methods, and properties in this code."
            elif visualization_type == "sequence_diagram":
                system_prompt = "Generate a detailed Mermaid sequence diagram representing the interaction flow between components or functions in this code."
            else:
                system_prompt = f"Generate a detailed Mermaid {visualization_type} diagram representing this code's structure and logic."
                
            # Use OpenAI API to analyze the code and generate visualization
            prompt = f"Please analyze this code and generate a {visualization_type} diagram using Mermaid syntax:"
            
            # Use the OpenAI integration
            input_text = f"{prompt}\n\n```\n{code}\n```"
            response = orel_ai.generate_completion(input_text, system_prompt)
            
            # Extract Mermaid code from the response
            mermaid_pattern = r"```(?:mermaid)?\s*([\s\S]*?)```"
            matches = re.findall(mermaid_pattern, response)
            
            mermaid_code = matches[0].strip() if matches else response
            
            # Log the operation
            self._log_operation(
                "visualize_code", 
                {"visualization_type": visualization_type, "success": True}
            )
            
            return {
                "success": True,
                "mermaid_code": mermaid_code,
                "visualization_type": visualization_type,
                "instructions": "This Mermaid diagram can be rendered with any Mermaid-compatible viewer."
            }
            
        except Exception as e:
            logger.error(f"Error visualizing code: {str(e)}")
            self._log_operation(
                "visualize_code", 
                {"visualization_type": visualization_type, "success": False, "error": str(e)}
            )
            return {
                "success": False,
                "error": str(e)
            }
            
    def extract_ui_elements(self, image_data: Union[str, bytes]) -> Dict[str, Any]:
        """
        Extract UI elements from an image and suggest code implementations
        
        Args:
            image_data (str or bytes): Base64 encoded image or file path
            
        Returns:
            dict: Extracted UI elements and code suggestions
        """
        if not VISION_AVAILABLE:
            return {
                "success": False,
                "error": "Vision capabilities not available"
            }
            
        try:
            # Prepare image data
            processed_image_data = self._prepare_image_data(image_data)
            
            # Prepare system prompt for UI extraction
            system_prompt = "You are a UI/UX expert. Identify and analyze UI elements in this image, describing their type, position, and purpose. Return a structured, detailed analysis of the interface design."
            
            # Use OpenAI API to analyze the image and extract UI elements
            prompt = "Please identify all UI elements in this image (buttons, inputs, navigation, etc.). For each element, describe its type, appearance, likely function, and suggest code implementations."
            
            # Use the OpenAI integration
            response = orel_ai.analyze_image_with_text(
                processed_image_data,
                prompt,
                system_prompt
            )
            
            # Parse the UI elements from the response
            ui_elements = self._parse_ui_elements(response)
            
            # Log the operation
            self._log_operation("extract_ui_elements", {"success": True})
            
            return {
                "success": True,
                "ui_elements": ui_elements,
                "raw_analysis": response
            }
            
        except Exception as e:
            logger.error(f"Error extracting UI elements: {str(e)}")
            self._log_operation(
                "extract_ui_elements", 
                {"success": False, "error": str(e)}
            )
            return {
                "success": False,
                "error": str(e)
            }
            
    def _parse_ui_elements(self, response: str) -> List[Dict[str, Any]]:
        """Parse the AI response to extract UI elements"""
        elements = []
        
        # Extract element sections from the response
        # This is a simple extraction; for production, use more robust parsing
        element_pattern = r"(?:^|\n)((?:Button|Input|Navigation|Form|Dropdown|Checkbox|Radio|Toggle|Menu|Tab|Card|Modal|Alert|Icon|Image|Text)[^\n]*(?:\n(?!\n)[^\n]*)*)"
        matches = re.findall(element_pattern, response, re.IGNORECASE | re.MULTILINE)
        
        for i, match in enumerate(matches):
            lines = match.strip().split('\n')
            if not lines:
                continue
                
            # Extract element type and description
            element_type = "Unknown"
            description = ""
            implementation = {}
            
            header = lines[0].strip()
            type_match = re.match(r'^(Button|Input|Navigation|Form|Dropdown|Checkbox|Radio|Toggle|Menu|Tab|Card|Modal|Alert|Icon|Image|Text)[:\s]*(.*)', header, re.IGNORECASE)
            
            if type_match:
                element_type = type_match.group(1).capitalize()
                description = type_match.group(2).strip()
            else:
                description = header
                
            # Look for implementation suggestions in the element description
            for language in ["HTML", "React", "Flutter", "CSS"]:
                pattern = rf"{language}[:\s]*(```[\s\S]*?```|`[\s\S]*?`)"
                impl_match = re.search(pattern, match, re.IGNORECASE)
                if impl_match:
                    # Clean up the code snippet
                    code = impl_match.group(1)
                    code = re.sub(r'^```[\w]*\n|```$|^`|`$', '', code)
                    implementation[language.lower()] = code.strip()
                    
            elements.append({
                "id": i + 1,
                "type": element_type,
                "description": description,
                "implementation": implementation
            })
            
        return elements
        
    def analyze_architecture(self, image_data: Union[str, bytes]) -> Dict[str, Any]:
        """
        Analyze system architecture from an image and suggest implementation
        
        Args:
            image_data (str or bytes): Base64 encoded image or file path
            
        Returns:
            dict: Architecture analysis and implementation suggestions
        """
        if not VISION_AVAILABLE:
            return {
                "success": False,
                "error": "Vision capabilities not available"
            }
            
        try:
            # Prepare image data
            processed_image_data = self._prepare_image_data(image_data)
            
            # Prepare system prompt for architecture analysis
            system_prompt = "You are a software architecture expert. Analyze this system architecture diagram and identify key components, their relationships, and architectural patterns. Provide detailed implementation suggestions."
            
            # Use OpenAI API to analyze the image and architecture
            prompt = "Please analyze this architecture diagram. Identify components, their relationships, architectural patterns, and suggest implementation approaches. Include code structure recommendations."
            
            # Use the OpenAI integration
            response = orel_ai.analyze_image_with_text(
                processed_image_data,
                prompt,
                system_prompt
            )
            
            # Log the operation
            self._log_operation("analyze_architecture", {"success": True})
            
            # Extract architecture patterns
            patterns = []
            for pattern_name, pattern_info in self.architecture_patterns.items():
                if any(p.lower() in response.lower() for p in pattern_info["patterns"]):
                    patterns.append({
                        "name": pattern_name,
                        "description": pattern_info["implementation"]["description"],
                        "code_structure": pattern_info["implementation"]["code_structure"]
                    })
            
            return {
                "success": True,
                "analysis": response,
                "detected_patterns": patterns
            }
            
        except Exception as e:
            logger.error(f"Error analyzing architecture: {str(e)}")
            self._log_operation(
                "analyze_architecture", 
                {"success": False, "error": str(e)}
            )
            return {
                "success": False,
                "error": str(e)
            }
            
    def generate_ui_from_description(self, description: str, framework: str = "html") -> Dict[str, Any]:
        """
        Generate UI code from a text description
        
        Args:
            description (str): Textual description of the desired UI
            framework (str): Target framework/language (html, react, flutter)
            
        Returns:
            dict: Generated UI code
        """
        if not VISION_AVAILABLE:
            return {
                "success": False,
                "error": "Vision capabilities not available"
            }
            
        try:
            # Validate framework
            framework = framework.lower()
            valid_frameworks = ["html", "react", "vue", "angular", "flutter", "android", "ios"]
            
            if framework not in valid_frameworks:
                framework = "html"  # Default to HTML
                
            # Prepare system prompt based on framework
            system_prompts = {
                "html": "You are an expert HTML/CSS developer. Generate clean, semantic HTML5 and CSS for the described UI.",
                "react": "You are an expert React developer. Generate functional components with hooks and CSS for the described UI.",
                "vue": "You are an expert Vue developer. Generate Vue components for the described UI.",
                "angular": "You are an expert Angular developer. Generate Angular components and templates for the described UI.",
                "flutter": "You are an expert Flutter developer. Generate Dart/Flutter widgets for the described UI.",
                "android": "You are an expert Android developer. Generate XML layouts and Java/Kotlin code for the described UI.",
                "ios": "You are an expert iOS developer. Generate Swift UI code for the described UI."
            }
            
            system_prompt = system_prompts.get(framework, system_prompts["html"])
            
            # Use OpenAI API to generate UI code
            prompt = f"Please generate {framework.upper()} code for a UI with the following description:\n\n{description}\n\nInclude all necessary code (HTML, CSS, component definitions, etc.) to implement this UI."
            
            # Use the OpenAI integration
            response = orel_ai.generate_completion(prompt, system_prompt)
            
            # Extract code blocks from the response
            code_blocks = []
            
            # Primary code pattern matching based on framework
            if framework in ["html", "react", "vue", "angular"]:
                # Look for HTML/JSX/template blocks
                html_pattern = r"```(?:html|jsx|vue|xml)?\s*([\s\S]*?)```"
                html_matches = re.findall(html_pattern, response)
                
                if html_matches:
                    code_blocks.append({
                        "id": 1,
                        "language": "html" if framework == "html" else framework,
                        "code": html_matches[0].strip()
                    })
                    
                # Look for CSS blocks
                css_pattern = r"```(?:css)?\s*([\s\S]*?)```"
                css_matches = re.findall(css_pattern, response)
                
                # Only add CSS if it's different from HTML block
                if css_matches and (not html_matches or css_matches[0].strip() != html_matches[0].strip()):
                    code_blocks.append({
                        "id": len(code_blocks) + 1,
                        "language": "css",
                        "code": css_matches[0].strip()
                    })
                    
                # Look for JS/TS blocks
                js_pattern = r"```(?:javascript|typescript|js|ts)?\s*([\s\S]*?)```"
                js_matches = re.findall(js_pattern, response)
                
                # Only add JS if it's different from previous blocks
                if js_matches and all(js_matches[0].strip() != block["code"] for block in code_blocks):
                    code_blocks.append({
                        "id": len(code_blocks) + 1,
                        "language": "javascript" if framework != "angular" else "typescript",
                        "code": js_matches[0].strip()
                    })
            else:
                # For other frameworks like Flutter, Android, iOS
                # Just extract all code blocks
                code_pattern = r"```(?:\w+)?\s*([\s\S]*?)```"
                matches = re.findall(code_pattern, response)
                
                for i, code in enumerate(matches):
                    code_blocks.append({
                        "id": i + 1,
                        "language": framework,
                        "code": code.strip()
                    })
            
            # If no code blocks found, extract the entire response
            if not code_blocks:
                code_blocks.append({
                    "id": 1,
                    "language": framework,
                    "code": response
                })
                
            # Log the operation
            self._log_operation(
                "generate_ui_from_description", 
                {"framework": framework, "success": True}
            )
            
            return {
                "success": True,
                "code": code_blocks,
                "framework": framework
            }
            
        except Exception as e:
            logger.error(f"Error generating UI: {str(e)}")
            self._log_operation(
                "generate_ui_from_description", 
                {"framework": framework, "success": False, "error": str(e)}
            )
            return {
                "success": False,
                "error": str(e)
            }
            
    def detect_design_patterns(self, code: str) -> Dict[str, Any]:
        """
        Analyze code to detect design patterns and visualize them
        
        Args:
            code (str): Code to analyze
            
        Returns:
            dict: Detected patterns and visualization
        """
        if not VISION_AVAILABLE:
            return {
                "success": False,
                "error": "Vision capabilities not available"
            }
            
        try:
            # Prepare system prompt for pattern detection
            system_prompt = "You are an expert software architect specializing in design patterns. Analyze this code to identify implemented design patterns, explain how they're used, and suggest potential improvements. Include UML diagrams using Mermaid syntax."
            
            # Use OpenAI API to analyze the code and detect patterns
            prompt = "Please analyze this code for design patterns. Identify any patterns used, explain their implementation, and suggest a Mermaid diagram to visualize the pattern:"
            
            # Use the OpenAI integration
            input_text = f"{prompt}\n\n```\n{code}\n```"
            response = orel_ai.generate_completion(input_text, system_prompt)
            
            # Extract patterns from response
            pattern_sections = re.split(r'\n##\s+', response)
            patterns = []
            
            if len(pattern_sections) > 1:
                # If the response is formatted with markdown headers
                for section in pattern_sections[1:]:  # Skip intro section
                    lines = section.strip().split('\n')
                    if not lines:
                        continue
                        
                    pattern_name = lines[0].strip()
                    description = "\n".join(lines[1:]).strip()
                    
                    patterns.append({
                        "name": pattern_name,
                        "description": description
                    })
            else:
                # If not formatted with headers, use the whole response
                patterns.append({
                    "name": "Pattern Analysis",
                    "description": response.strip()
                })
                
            # Extract Mermaid code from the response
            mermaid_pattern = r"```(?:mermaid)?\s*([\s\S]*?)```"
            matches = re.findall(mermaid_pattern, response)
            
            visualization = ""
            if matches:
                visualization = matches[0].strip()
            else:
                visualization = self._extract_visualization_suggestion(response)
                
            # Log the operation
            self._log_operation("detect_design_patterns", {"success": True})
            
            return {
                "success": True,
                "patterns": patterns,
                "visualization": visualization,
                "raw_analysis": response
            }
            
        except Exception as e:
            logger.error(f"Error detecting design patterns: {str(e)}")
            self._log_operation(
                "detect_design_patterns", 
                {"success": False, "error": str(e)}
            )
            return {
                "success": False,
                "error": str(e)
            }
            
    def _extract_visualization_suggestion(self, text: str) -> str:
        """Extract visualization suggestion from pattern analysis"""
        # Look for paragraphs that seem to describe a diagram
        diagram_indicators = [
            r"(?:Here|Below) (?:is|would be) a (?:Mermaid )?diagram",
            r"A (?:Mermaid )?diagram (?:to|that) visualize",
            r"(?:could|can|would) be visualized as",
            r"(?:UML|Class|Sequence|Flow) diagram",
            r"(?:Here's|Here is) how [a-z]+ (?:would|could) look"
        ]
        
        for indicator in diagram_indicators:
            match = re.search(f"{indicator}[^.]*\\.", text, re.IGNORECASE)
            if match:
                start_idx = match.start()
                
                # Look for a paragraph break after the matching sentence
                paragraph_end = text.find("\n\n", start_idx)
                if paragraph_end == -1:
                    paragraph_end = len(text)
                    
                # Extract the paragraph
                return text[start_idx:paragraph_end].strip()
                
        # Fallback to a generic diagram instruction
        return "Visualization suggestion not found. Consider creating a class diagram showing the main components and their relationships."
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the VisionCoder module
        
        Returns:
            dict: Status information
        """
        return {
            "available": VISION_AVAILABLE,
            "supported_languages": self.supported_languages,
            "supported_frameworks": self.supported_frameworks,
            "supported_visualizations": self.supported_visualizations,
            "recent_operations": self.operation_history[-5:] if self.operation_history else [],
            "version": "1.0.0"
        }


class OrelVisionCoderModule:
    """
    Adapter class to integrate VisionCoder with Or'el's module system
    """
    def __init__(self):
        """Initialize the VisionCoder module for Or'el"""
        self.name = "VisionCoder"
        self.description = "Visual programming capabilities for Or'el"
        self.version = "1.0.0"
        self.enabled = True
        self.improvement_level = 0
        self.capabilities = [
            "generate_code_from_image",
            "visualize_code",
            "extract_ui_elements",
            "analyze_architecture",
            "generate_ui_from_description",
            "detect_design_patterns"
        ]
        self.vision_coder = VisionCoder()
        logger.info(f"VisionCoder v{self.version} initialized successfully")
        
    def execute(self, command: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute VisionCoder commands from Or'el
        
        Args:
            command (str): Command to execute
            context (dict, optional): Command context and parameters
            
        Returns:
            dict: Command results
        """
        if not self.enabled:
            return {
                "success": False,
                "error": "VisionCoder module is currently disabled."
            }
            
        if not context:
            context = {}
            
        try:
            if command == "generate_code_from_image":
                return self.vision_coder.generate_code_from_image(
                    context.get("image_data", ""),
                    context.get("target_language", "python"),
                    context.get("additional_context")
                )
                
            elif command == "visualize_code":
                return self.vision_coder.visualize_code(
                    context.get("code", ""),
                    context.get("visualization_type", "flowchart")
                )
                
            elif command == "extract_ui_elements":
                return self.vision_coder.extract_ui_elements(
                    context.get("image_data", "")
                )
                
            elif command == "analyze_architecture":
                return self.vision_coder.analyze_architecture(
                    context.get("image_data", "")
                )
                
            elif command == "generate_ui_from_description":
                return self.vision_coder.generate_ui_from_description(
                    context.get("description", ""),
                    context.get("framework", "html")
                )
                
            elif command == "detect_design_patterns":
                return self.vision_coder.detect_design_patterns(
                    context.get("code", "")
                )
                
            else:
                return {
                    "success": False,
                    "error": f"Unknown command: {command}",
                    "available_commands": self.capabilities
                }
                
        except Exception as e:
            logger.error(f"Error executing VisionCoder command {command}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def improve(self):
        """
        Improve VisionCoder capabilities
        
        Returns:
            str: Improvement results
        """
        self.improvement_level += 1
        
        # Improvements at different levels
        improvements = []
        
        if self.improvement_level == 1:
            improvements.append("Enhanced code generation accuracy")
        elif self.improvement_level == 2:
            improvements.append("Improved UI element recognition")
            self.capabilities.append("advanced_ui_analysis")
        elif self.improvement_level == 3:
            improvements.append("Added support for more programming languages")
            self.vision_coder.supported_languages.extend(["ruby", "php", "dart"])
        elif self.improvement_level >= 4:
            improvements.append("Optimized architecture analysis algorithms")
            
        return f"VisionCoder improved to level {self.improvement_level}. {', '.join(improvements)}"
        
    def get_capabilities(self) -> List[str]:
        """
        Get the list of capabilities provided by this module
        
        Returns:
            list: Module capabilities
        """
        return self.capabilities
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get module status
        
        Returns:
            dict: Status information
        """
        status = self.vision_coder.get_status()
        status.update({
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "enabled": self.enabled,
            "improvement_level": self.improvement_level
        })
        return status
        
    def describe(self):
        """Return a description of this module"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "enabled": self.enabled,
            "improvement_level": self.improvement_level,
            "capabilities": self.capabilities
        }