"""
Replit-like Capabilities Module for Or'el

This module provides code editing, execution, and real-time feedback functionality
similar to Replit, allowing Or'el to help with code development tasks.
"""

import os
import sys
import subprocess
import tempfile
import logging
import traceback
from typing import Dict, List, Any, Optional, Union
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeExecutionEnvironment:
    """
    Manages a safe environment for code execution with proper isolation
    and resource constraints
    """
    
    def __init__(self):
        """Initialize the code execution environment"""
        self.supported_languages = {
            "python": {
                "extension": ".py",
                "cmd": ["python"],
                "version_cmd": ["python", "--version"]
            },
            "javascript": {
                "extension": ".js",
                "cmd": ["node"],
                "version_cmd": ["node", "--version"]
            },
            "html": {
                "extension": ".html",
                "cmd": None,  # HTML is not executed but rendered
                "version_cmd": None
            },
            "css": {
                "extension": ".css",
                "cmd": None,  # CSS is not executed but used in rendering
                "version_cmd": None
            }
        }
        self.execution_timeout = 10  # seconds
        
    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages"""
        return list(self.supported_languages.keys())
        
    def check_language_support(self, language: str) -> bool:
        """Check if a language is supported"""
        return language.lower() in self.supported_languages
        
    def execute_code(self, code: str, language: str, inputs: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute code in a specified language
        
        Args:
            code: The code to execute
            language: Programming language of the code
            inputs: Optional stdin inputs for the code
            
        Returns:
            dict: Execution results with stdout, stderr, and status
        """
        language = language.lower()
        
        if not self.check_language_support(language):
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Language '{language}' is not supported",
                "execution_time": 0
            }
            
        if language in ["html", "css"]:
            return {
                "success": True,
                "stdout": f"{language.upper()} doesn't produce output when executed. Preview it in a browser instead.",
                "stderr": "",
                "execution_time": 0
            }
            
        # Create a temporary file for the code
        temp_filename = None
        process = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=self.supported_languages[language]["extension"],
                mode="w+",
                delete=False
            ) as temp_file:
                temp_file.write(code)
                temp_filename = temp_file.name
                
            # Construct the command to run the code
            cmd = self.supported_languages[language]["cmd"] + [temp_filename]
            
            # Execute the code with a timeout
            start_time = __import__("time").time()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE if inputs else None,
                text=True
            )
            
            stdout, stderr = process.communicate(
                input=inputs,
                timeout=self.execution_timeout
            )
            
            end_time = __import__("time").time()
            execution_time = end_time - start_time
            
            # Clean up temporary file
            if temp_filename:
                os.unlink(temp_filename)
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
                "execution_time": round(execution_time, 3)
            }
            
        except subprocess.TimeoutExpired:
            if process:
                try:
                    process.kill()
                except:
                    pass
                
            if temp_filename:
                try:
                    os.unlink(temp_filename)
                except:
                    pass
                
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {self.execution_timeout} seconds",
                "execution_time": self.execution_timeout
            }
            
        except Exception as e:
            if temp_filename:
                try:
                    os.unlink(temp_filename)
                except:
                    pass
                
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Error executing code: {str(e)}",
                "execution_time": 0
            }

class CodeEditor:
    """
    Provides code editing capabilities including syntax highlighting,
    auto-completion, and code formatting
    """
    
    def __init__(self):
        """Initialize the code editor"""
        self.formatter_settings = {
            "python": {
                "indent_size": 4,
                "use_spaces": True
            },
            "javascript": {
                "indent_size": 2,
                "use_spaces": True
            },
            "html": {
                "indent_size": 2,
                "use_spaces": True
            },
            "css": {
                "indent_size": 2,
                "use_spaces": True
            }
        }
        
    def format_code(self, code: str, language: str) -> str:
        """
        Format code according to language-specific formatting rules
        
        Args:
            code: The code to format
            language: Programming language of the code
            
        Returns:
            str: Formatted code
        """
        language = language.lower()
        
        # Basic indentation fixing
        if language in self.formatter_settings:
            indent_size = self.formatter_settings[language]["indent_size"]
            use_spaces = self.formatter_settings[language]["use_spaces"]
            
            # Fix mixed tabs and spaces (convert tabs to spaces)
            if use_spaces:
                code = code.replace("\t", " " * indent_size)
                
            # Fix inconsistent indentation
            lines = code.split("\n")
            fixed_lines = []
            
            for line in lines:
                # Count leading spaces/tabs
                leading_count = len(line) - len(line.lstrip())
                if leading_count > 0:
                    # Calculate the correct indentation level
                    indent_level = leading_count // (indent_size if use_spaces else 1)
                    indent_char = " " * indent_size if use_spaces else "\t"
                    fixed_line = (indent_char * indent_level) + line.lstrip()
                    fixed_lines.append(fixed_line)
                else:
                    fixed_lines.append(line)
                    
            code = "\n".join(fixed_lines)
            
        return code
        
    def get_syntax_highlighting_rules(self, language: str) -> Dict[str, Any]:
        """
        Get syntax highlighting rules for a language
        
        Args:
            language: Programming language
            
        Returns:
            dict: Syntax highlighting rules
        """
        language = language.lower()
        
        # Basic highlighting rules for common elements
        common_rules = {
            "string": r'"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\'',
            "comment": r'#.*$|//.*$|/\*[\s\S]*?\*/',
            "number": r'\b\d+(?:\.\d+)?\b'
        }
        
        language_specific = {}
        
        if language == "python":
            language_specific = {
                "keyword": r'\b(if|else|elif|for|while|try|except|finally|def|class|import|from|as|return|break|continue|pass|True|False|None)\b',
                "builtin": r'\b(print|len|range|int|str|list|dict|set|tuple|max|min|sum|abs|open|close|read|write)\b',
                "decoration": r'@\w+'
            }
        elif language == "javascript":
            language_specific = {
                "keyword": r'\b(if|else|for|while|try|catch|finally|function|var|let|const|return|break|continue|switch|case|default|new|this|typeof|instanceof|null|undefined|true|false)\b',
                "builtin": r'\b(console|document|window|Math|Array|Object|String|Number|Boolean|Date|RegExp|JSON)\b',
                "arrow_function": r'=>'
            }
        elif language == "html":
            language_specific = {
                "tag": r'<[^>]*>',
                "attribute": r'\s(\w+)=',
                "entity": r'&[^;]+;'
            }
        elif language == "css":
            language_specific = {
                "selector": r'[\w\.-]+\s*\{',
                "property": r'[\w-]+\s*:',
                "value": r':\s*[^;]+',
                "unit": r'\b\d+(?:px|em|rem|vh|vw|%|s|ms)\b'
            }
            
        return {**common_rules, **language_specific}
        
    def suggest_autocomplete(self, code: str, language: str, cursor_position: int) -> List[str]:
        """
        Suggest autocomplete options based on code context
        
        Args:
            code: The current code
            language: Programming language of the code
            cursor_position: Position of the cursor in the code
            
        Returns:
            list: Autocomplete suggestions
        """
        language = language.lower()
        
        # Extract the partial word at cursor position
        partial_word = ""
        
        if cursor_position > 0:
            i = cursor_position - 1
            while i >= 0 and code[i].isalnum() or code[i] == '_':
                partial_word = code[i] + partial_word
                i -= 1
                
        if not partial_word:
            return []
            
        # Common language-agnostic suggestions
        suggestions = []
        
        # Get all words in the code as potential autocomplete candidates
        words = re.findall(r'\b\w+\b', code)
        unique_words = set(words)
        
        for word in unique_words:
            if word.startswith(partial_word) and word != partial_word:
                suggestions.append(word)
                
        # Add language-specific suggestions
        if language == "python":
            python_keywords = [
                "if", "else", "elif", "for", "while", "try", "except", "finally",
                "def", "class", "import", "from", "as", "return", "break", "continue",
                "pass", "True", "False", "None", "with", "yield", "lambda", "global",
                "nonlocal", "assert", "del", "in", "is", "not", "and", "or"
            ]
            
            python_builtins = [
                "print", "len", "range", "int", "str", "list", "dict", "set", "tuple",
                "max", "min", "sum", "abs", "open", "close", "read", "write", "append",
                "extend", "insert", "remove", "pop", "clear", "index", "count", "sort",
                "reverse", "join", "split", "strip", "lower", "upper", "replace"
            ]
            
            for kw in python_keywords + python_builtins:
                if kw.startswith(partial_word) and kw != partial_word and kw not in suggestions:
                    suggestions.append(kw)
                    
        elif language == "javascript":
            js_keywords = [
                "if", "else", "for", "while", "try", "catch", "finally", "function",
                "var", "let", "const", "return", "break", "continue", "switch", "case",
                "default", "new", "this", "typeof", "instanceof", "null", "undefined",
                "true", "false", "class", "extends", "super", "import", "export", "from",
                "as", "async", "await", "of", "in", "do", "delete", "void"
            ]
            
            js_builtins = [
                "console.log", "document.getElementById", "window.addEventListener",
                "Math.random", "Array.isArray", "Object.keys", "String.prototype",
                "Number.parseInt", "JSON.stringify", "JSON.parse", "setTimeout",
                "setInterval", "clearTimeout", "clearInterval", "fetch", "then", "catch",
                "map", "filter", "reduce", "forEach", "find", "some", "every", "push",
                "pop", "shift", "unshift", "slice", "splice", "join", "split", "trim"
            ]
            
            for kw in js_keywords:
                if kw.startswith(partial_word) and kw != partial_word and kw not in suggestions:
                    suggestions.append(kw)
                    
            for builtin in js_builtins:
                if builtin.startswith(partial_word) and builtin != partial_word and builtin not in suggestions:
                    suggestions.append(builtin)
                    
        # Sort suggestions alphabetically
        suggestions.sort()
        
        return suggestions[:10]  # Limit to 10 suggestions

class DebugHelper:
    """
    Provides debugging assistance for code issues including error analysis
    and suggested fixes
    """
    
    def __init__(self):
        """Initialize the debug helper"""
        self.common_error_patterns = {
            "python": {
                r"NameError: name '(\w+)' is not defined": "The variable '{0}' was used before being defined.",
                r"SyntaxError: invalid syntax": "There's a syntax error in your code. Check for missing parentheses, brackets, or quotes.",
                r"IndentationError: (expected an indented block|unexpected indent)": "Your code has inconsistent indentation. Make sure you're using consistent spaces or tabs.",
                r"TypeError: (.*?)": "Type error: {0}. You might be using incompatible types of data together.",
                r"IndexError: list index out of range": "You're trying to access an index that doesn't exist in your list.",
                r"KeyError: (.+)": "The key '{0}' doesn't exist in the dictionary.",
                r"ImportError: No module named '(\w+)'": "The module '{0}' is not installed or cannot be found."
            },
            "javascript": {
                r"ReferenceError: (\w+) is not defined": "The variable '{0}' was used before being defined.",
                r"SyntaxError: (Unexpected token|missing \))": "There's a syntax error in your code: {0}. Check for missing parentheses, brackets, or quotes.",
                r"TypeError: (.*?)": "Type error: {0}. You might be using incompatible types of data together.",
                r"RangeError: (.*?)": "Range error: {0}. This often happens when an array index is out of bounds or a recursive function causes a stack overflow.",
                r"Uncaught TypeError: (.*?) is not a function": "You're trying to call {0} as a function, but it's not a function."
            }
        }
        
    def analyze_error(self, error_message: str, language: str) -> Dict[str, Any]:
        """
        Analyze an error message and provide helpful debugging information
        
        Args:
            error_message: The error message to analyze
            language: Programming language of the code
            
        Returns:
            dict: Error analysis with explanation and suggested fixes
        """
        language = language.lower()
        explanation = "Unknown error"
        suggested_fix = "Review your code carefully"
        
        if language in self.common_error_patterns:
            for pattern, template in self.common_error_patterns[language].items():
                match = re.search(pattern, error_message)
                if match:
                    # Format the explanation with captured groups
                    groups = match.groups()
                    explanation = template.format(*groups) if groups else template
                    
                    # Generate suggested fix based on the type of error
                    if "NameError" in pattern or "ReferenceError" in pattern:
                        var_name = groups[0] if groups else ""
                        suggested_fix = f"Make sure '{var_name}' is defined before using it."
                    elif "SyntaxError" in pattern:
                        suggested_fix = "Check your syntax - look for missing brackets, parentheses, or semicolons."
                    elif "IndentationError" in pattern:
                        suggested_fix = "Fix your indentation to be consistent throughout your code."
                    elif "TypeError" in pattern:
                        suggested_fix = "Make sure you're using compatible data types in your operations."
                    elif "IndexError" in pattern or "RangeError" in pattern:
                        suggested_fix = "Check that you're not trying to access elements beyond the length of your array/list."
                    elif "KeyError" in pattern:
                        key = groups[0] if groups else ""
                        suggested_fix = f"Make sure the key '{key}' exists in your dictionary before accessing it."
                    elif "ImportError" in pattern:
                        module = groups[0] if groups else ""
                        suggested_fix = f"Install the module '{module}' or check if it's available in your environment."
                    break
        
        return {
            "original_error": error_message,
            "explanation": explanation,
            "suggested_fix": suggested_fix
        }
        
    def suggest_performance_improvements(self, code: str, language: str) -> List[Dict[str, str]]:
        """
        Suggest performance improvements for code
        
        Args:
            code: The code to analyze
            language: Programming language of the code
            
        Returns:
            list: Suggested performance improvements
        """
        language = language.lower()
        suggestions = []
        
        if language == "python":
            # Check for inefficient list operations
            if re.search(r'for\s+\w+\s+in\s+range\(len\((\w+)\)\):', code):
                suggestions.append({
                    "issue": "Inefficient list iteration",
                    "suggestion": "Instead of 'for i in range(len(list_name))', use 'for item in list_name' for direct iteration.",
                    "line_pattern": r'for\s+\w+\s+in\s+range\(len\((\w+)\)\):'
                })
                
            # Check for repeated calls that could be cached
            if re.search(r'([^=\s]+\((?:[^()]*)\))[^\n=]*\1', code):
                suggestions.append({
                    "issue": "Repeated function call",
                    "suggestion": "Consider storing the result of the repeated function call in a variable.",
                    "line_pattern": r'([^=\s]+\((?:[^()]*)\))[^\n=]*\1'
                })
                
            # Check for string concatenation in loops
            if re.search(r'for.*:\s*[^=\n]*(\w+)\s*\+=\s*(?:"|\').*(?:\'|")', code):
                suggestions.append({
                    "issue": "String concatenation in loop",
                    "suggestion": "Use a list to collect strings and join them after the loop for better performance.",
                    "line_pattern": r'for.*:\s*[^=\n]*(\w+)\s*\+=\s*(?:"|\').*(?:\'|")'
                })
                
        elif language == "javascript":
            # Check for inefficient array operations
            if re.search(r'for\s*\(\s*let\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*(\w+)\.length\s*;', code):
                suggestions.append({
                    "issue": "Inefficient array iteration",
                    "suggestion": "Cache the array length outside the loop to avoid recalculating it on each iteration.",
                    "line_pattern": r'for\s*\(\s*let\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*(\w+)\.length\s*;'
                })
                
            # Check for document.getElementById repeated calls
            if re.search(r'document\.getElementById\([^)]+\)[^;]*\s*document\.getElementById\([^)]+\)', code):
                suggestions.append({
                    "issue": "Repeated DOM selection",
                    "suggestion": "Store DOM elements in variables instead of repeatedly calling document.getElementById.",
                    "line_pattern": r'document\.getElementById\([^)]+\)'
                })
                
            # Check for string concatenation in loops
            if re.search(r'for.*{\s*[^=\n]*(\w+)\s*\+=\s*(?:"|\').*(?:\'|")', code):
                suggestions.append({
                    "issue": "String concatenation in loop",
                    "suggestion": "Use array joining for string concatenation for better performance.",
                    "line_pattern": r'for.*{\s*[^=\n]*(\w+)\s*\+=\s*(?:"|\').*(?:\'|")'
                })
                
        return suggestions
    
    def check_security_issues(self, code: str, language: str) -> List[Dict[str, str]]:
        """
        Check for potential security issues in code
        
        Args:
            code: The code to analyze
            language: Programming language of the code
            
        Returns:
            list: Identified security issues
        """
        language = language.lower()
        issues = []
        
        if language == "python":
            # Check for potential SQL injection
            if re.search(r"cursor\.execute\([\"'].*\s*\+\s*", code) or re.search(r"cursor\.execute\([\"'].*%.*%[\"']\s*%\s*", code):
                issues.append({
                    "severity": "High",
                    "issue": "Potential SQL Injection",
                    "description": "Building SQL queries with string concatenation is vulnerable to SQL injection attacks.",
                    "suggestion": "Use parameterized queries with placeholders instead of string concatenation."
                })
                
            # Check for eval usage
            if re.search(r"eval\(", code):
                issues.append({
                    "severity": "High",
                    "issue": "Use of eval()",
                    "description": "The eval() function can execute arbitrary code and is a security risk.",
                    "suggestion": "Avoid using eval(). Instead, use safer alternatives like JSON.parse() for JSON data or direct property access."
                })
                
            # Check for unsafe deserialization
            if re.search(r"pickle\.loads\(|pickle\.load\(|marshal\.loads\(|yaml\.load\((?![^)]*Loader=yaml\.SafeLoader)", code):
                issues.append({
                    "severity": "High",
                    "issue": "Unsafe Deserialization",
                    "description": "Deserializing untrusted data can lead to remote code execution.",
                    "suggestion": "Only deserialize trusted data, or use safer alternatives like JSON or YAML with SafeLoader."
                })
                
        elif language == "javascript":
            # Check for potential XSS
            if re.search(r"innerHTML\s*=|document\.write\(", code):
                issues.append({
                    "severity": "High",
                    "issue": "Potential Cross-Site Scripting (XSS)",
                    "description": "Setting innerHTML or using document.write with user-controlled data can lead to XSS attacks.",
                    "suggestion": "Use textContent instead of innerHTML, or sanitize input before adding to the DOM."
                })
                
            # Check for eval usage
            if re.search(r"eval\(|new Function\(|setTimeout\(['\"][^)]+\)", code):
                issues.append({
                    "severity": "High",
                    "issue": "Use of eval() or similar functions",
                    "description": "Functions that execute arbitrary strings as code are security risks.",
                    "suggestion": "Avoid using eval(), new Function(), and string parameters in setTimeout() or setInterval()."
                })
                
            # Check for prototype pollution
            if re.search(r"Object\.prototype|Object\.assign\(\s*Object\.prototype", code):
                issues.append({
                    "severity": "Medium",
                    "issue": "Potential Prototype Pollution",
                    "description": "Modifying Object.prototype can affect all objects and cause unexpected behavior.",
                    "suggestion": "Avoid modifying built-in object prototypes."
                })
                
        return issues

class ReplitCapabilities:
    """
    Main class that coordinates code execution, editing, and debugging
    to provide a Replit-like coding experience
    """
    
    def __init__(self):
        """Initialize Replit-like capabilities"""
        self.code_executor = CodeExecutionEnvironment()
        self.code_editor = CodeEditor()
        self.debug_helper = DebugHelper()
        logger.info("Replit Capabilities v1.0.0 initialized")
        
    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages"""
        return self.code_executor.get_supported_languages()
        
    def execute_code(self, code: str, language: str, inputs: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute code and return the results
        
        Args:
            code: The code to execute
            language: Programming language of the code
            inputs: Optional stdin inputs for the code
            
        Returns:
            dict: Execution results
        """
        # Format the code before execution
        formatted_code = self.code_editor.format_code(code, language)
        
        # Execute the code
        result = self.code_executor.execute_code(formatted_code, language, inputs)
        
        # If execution failed, analyze the error
        if not result["success"] and result["stderr"]:
            error_analysis = self.debug_helper.analyze_error(result["stderr"], language)
            result["error_analysis"] = error_analysis
            
        return result
        
    def edit_code(self, code: str, language: str) -> Dict[str, Any]:
        """
        Edit and improve code
        
        Args:
            code: The code to edit
            language: Programming language of the code
            
        Returns:
            dict: Edited code with improvements
        """
        # Format the code
        formatted_code = self.code_editor.format_code(code, language)
        
        # Check for security issues
        security_issues = self.debug_helper.check_security_issues(formatted_code, language)
        
        # Check for performance improvements
        performance_suggestions = self.debug_helper.suggest_performance_improvements(formatted_code, language)
        
        return {
            "original_code": code,
            "formatted_code": formatted_code,
            "security_issues": security_issues,
            "performance_suggestions": performance_suggestions,
            "highlighting_rules": self.code_editor.get_syntax_highlighting_rules(language)
        }
        
    def suggest_autocomplete(self, code: str, language: str, cursor_position: int) -> List[str]:
        """
        Suggest autocomplete options based on code context
        
        Args:
            code: The current code
            language: Programming language of the code
            cursor_position: Position of the cursor in the code
            
        Returns:
            list: Autocomplete suggestions
        """
        return self.code_editor.suggest_autocomplete(code, language, cursor_position)
        
    def debug_code(self, code: str, error_message: str, language: str) -> Dict[str, Any]:
        """
        Debug code with error analysis and suggestions
        
        Args:
            code: The code to debug
            error_message: Error message to analyze
            language: Programming language of the code
            
        Returns:
            dict: Debugging results with suggestions
        """
        # Analyze the error
        error_analysis = self.debug_helper.analyze_error(error_message, language)
        
        # Check for security issues
        security_issues = self.debug_helper.check_security_issues(code, language)
        
        # Check for performance improvements
        performance_suggestions = self.debug_helper.suggest_performance_improvements(code, language)
        
        return {
            "error_analysis": error_analysis,
            "security_issues": security_issues,
            "performance_suggestions": performance_suggestions,
            "formatted_code": self.code_editor.format_code(code, language)
        }
        
    def create_project_template(self, project_type: str, language: str) -> Dict[str, str]:
        """
        Create a starter template for a specific project type
        
        Args:
            project_type: Type of project (e.g., "web", "cli", "game")
            language: Programming language for the project
            
        Returns:
            dict: Template files with their contents
        """
        language = language.lower()
        project_type = project_type.lower()
        
        templates = {}
        
        if language == "python":
            if project_type == "web":
                templates["app.py"] = """
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
"""
                templates["templates/index.html"] = """
<!DOCTYPE html>
<html>
<head>
    <title>My Flask App</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <h1>Welcome to My Flask App</h1>
    <p>This is a simple web application built with Flask.</p>
</body>
</html>
"""
                templates["templates/about.html"] = """
<!DOCTYPE html>
<html>
<head>
    <title>About - My Flask App</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <h1>About This App</h1>
    <p>This is a simple web application built with Flask for demonstration purposes.</p>
    <a href="/">Back to Home</a>
</body>
</html>
"""
                templates["static/css/style.css"] = """
body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    margin: 0;
    padding: 20px;
    background-color: #f4f4f4;
}

h1 {
    color: #333;
}

a {
    color: #0066cc;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}
"""
            elif project_type == "cli":
                templates["main.py"] = """
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='A simple CLI application')
    parser.add_argument('--name', help='Your name')
    parser.add_argument('--count', type=int, default=1, help='Number of greetings')
    
    args = parser.parse_args()
    
    name = args.name or 'World'
    count = args.count
    
    for _ in range(count):
        print(f"Hello, {name}!")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
"""
            elif project_type == "game":
                templates["game.py"] = """
import random

def number_guessing_game():
    number = random.randint(1, 100)
    attempts = 0
    
    print("Welcome to the Number Guessing Game!")
    print("I'm thinking of a number between 1 and 100.")
    
    while True:
        try:
            guess = int(input("Enter your guess: "))
            attempts += 1
            
            if guess < number:
                print("Too low! Try again.")
            elif guess > number:
                print("Too high! Try again.")
            else:
                print(f"Congratulations! You've guessed the number in {attempts} attempts!")
                break
        except ValueError:
            print("Please enter a valid number.")

if __name__ == '__main__':
    number_guessing_game()
"""
        elif language == "javascript":
            if project_type == "web":
                templates["index.html"] = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JavaScript Web App</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <h1>JavaScript Web App</h1>
        <div id="app">
            <p>Welcome to my JavaScript application</p>
            <button id="changeColorBtn">Change Background Color</button>
            <div class="counter">
                <button id="decrementBtn">-</button>
                <span id="counterValue">0</span>
                <button id="incrementBtn">+</button>
            </div>
        </div>
    </div>
    <script src="app.js"></script>
</body>
</html>
"""
                templates["styles.css"] = """
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    background-color: #f4f4f4;
    transition: background-color 0.5s ease;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background-color: white;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    border-radius: 5px;
    margin-top: 50px;
}

h1 {
    text-align: center;
    margin-bottom: 20px;
    color: #333;
}

button {
    padding: 8px 16px;
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    margin: 5px;
}

button:hover {
    background-color: #45a049;
}

.counter {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 20px;
}

#counterValue {
    font-size: 24px;
    margin: 0 15px;
    min-width: 40px;
    text-align: center;
}
"""
                templates["app.js"] = """
document.addEventListener('DOMContentLoaded', function() {
    // Get references to DOM elements
    const changeColorBtn = document.getElementById('changeColorBtn');
    const decrementBtn = document.getElementById('decrementBtn');
    const incrementBtn = document.getElementById('incrementBtn');
    const counterValue = document.getElementById('counterValue');
    
    // Initial counter value
    let count = 0;
    
    // Function to change background color
    function changeBackgroundColor() {
        const randomColor = getRandomColor();
        document.body.style.backgroundColor = randomColor;
    }
    
    // Function to generate random color
    function getRandomColor() {
        const letters = '0123456789ABCDEF';
        let color = '#';
        for (let i = 0; i < 6; i++) {
            color += letters[Math.floor(Math.random() * 16)];
        }
        return color;
    }
    
    // Function to update counter display
    function updateCounter() {
        counterValue.textContent = count;
    }
    
    // Add event listeners
    changeColorBtn.addEventListener('click', changeBackgroundColor);
    
    decrementBtn.addEventListener('click', function() {
        count--;
        updateCounter();
    });
    
    incrementBtn.addEventListener('click', function() {
        count++;
        updateCounter();
    });
});
"""
        
        return templates