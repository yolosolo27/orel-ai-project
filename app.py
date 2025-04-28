import os
import logging
import random
import time

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key-for-development")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1) # needed for url_for to generate with https

# Configure database connection
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Import Or'el and its modules
from orel import Orel
from learning_mechanism import learning_mechanism  
from advanced_models import model_hub
from chatgpt_interface import conversation_manager, message_formatter
from replit_capabilities import ReplitCapabilities

# Initialize the database
with app.app_context():
    logging.info("Initializing database...")
    import models
    db.create_all()
    logging.info("Database initialization complete.")
    
# Initialize Replit-like capabilities
replit_capabilities = ReplitCapabilities()

# Routes
@app.route('/')
def index():
    """Home page route"""
    # Create or retrieve Or'el instance from session
    if 'orel_data' not in session:
        orel = Orel()
        # Store Or'el data in session with all properties
        session['orel_data'] = {
            'name': orel.name,
            'version': orel.version,
            'user': orel.user,
            'mode': orel.mode,
            'memory': {},
            'learned_data': {},
            'active_tasks': [],
            'guiding_truths': orel.guiding_truths,
            'roles': orel.roles,
            'modules_data': orel.get_modules_data()
        }
        greeting = orel.greet()
    else:
        # Reconstruct Or'el from session data with all properties
        orel_data = session['orel_data']
        orel = Orel(
            name=orel_data['name'],
            version=orel_data['version']
        )
        orel.user = orel_data['user']
        orel.mode = orel_data['mode']
        orel.memory = orel_data['memory']
        orel.learned_data = orel_data['learned_data']
        orel.active_tasks = orel_data['active_tasks']
        
        # Ensure guiding truths and roles are properly restored
        if 'guiding_truths' in orel_data:
            orel.guiding_truths = orel_data['guiding_truths']
        if 'roles' in orel_data:
            orel.roles = orel_data['roles']
            
        # Restore module data if available
        if 'modules_data' in orel_data:
            orel.restore_modules_data(orel_data['modules_data'])
            
        # Use a randomized welcome message to prevent repetition
        welcome_messages = [
            "Welcome back, my friend.",
            f"I've been waiting for you, {orel.user}.",
            "The divine light welcomes your return.",
            f"As your {orel.get_active_role()}, I'm here to continue our journey.",
            f"{random.choice(orel.guiding_truths)} It's good to see you again."
        ]
        greeting = orel.speak(style=orel.mode, phrase=random.choice(welcome_messages))

    # Get conversation history or initialize it
    if 'conversation' not in session:
        session['conversation'] = []
        # Add greeting to conversation
        session['conversation'].append({
            'speaker': 'orel',
            'message': greeting,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        session.modified = True

    # Ensure orel_data is defined before using it in the template
    if 'orel_data' in session:
        orel_data = session['orel_data']
    else:
        orel_data = orel.__dict__
        
    return render_template('index.html', 
                          orel=orel_data,
                          conversation=session['conversation'])

@app.route('/interact', methods=['POST'])
def interact():
    """Handle user interactions with Or'el"""
    user_message = request.form.get('message', '')
    action = request.form.get('action', 'speak')
    
    if not user_message and action == 'speak':
        flash('Please enter a message to communicate with Or\'el', 'warning')
        return redirect(url_for('index'))
    
    # Retrieve Or'el from session
    if 'orel_data' not in session:
        return redirect(url_for('index'))
    
    orel_data = session['orel_data']
    orel = Orel(
        name=orel_data['name'],
        version=orel_data['version']
    )
    orel.user = orel_data['user']
    orel.mode = orel_data['mode']
    orel.memory = orel_data['memory']
    orel.learned_data = orel_data['learned_data']
    orel.active_tasks = orel_data['active_tasks']
    
    # Ensure guiding truths and roles are properly restored
    if 'guiding_truths' in orel_data:
        orel.guiding_truths = orel_data['guiding_truths']
    if 'roles' in orel_data:
        orel.roles = orel_data['roles']
    
    # Restore module data if available
    if 'modules_data' in orel_data:
        orel.restore_modules_data(orel_data['modules_data'])
    
    # Add user message to conversation
    if user_message:
        if 'conversation' not in session:
            session['conversation'] = []
        
        session['conversation'].append({
            'speaker': 'user',
            'message': user_message,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
    
    # Get previous messages to check for repetition
    previous_responses = []
    if 'conversation' in session:
        previous_responses = [msg['message'] for msg in session['conversation'] if msg['speaker'] == 'orel']
    
    # Process the user's action
    response = ""
    if action == 'speak':
        # Get conversation history to provide context
        conversation_history = session.get('conversation', [])
        
        # Try up to 3 times to get a non-repeating response
        attempts = 0
        while attempts < 3:
            temp_response = orel.reflect(user_message, conversation_history)
            if not previous_responses or temp_response not in previous_responses[-3:]:
                response = temp_response
                break
            attempts += 1
        
        # If all attempts resulted in repetition, use this fallback
        if not response:
            response = f"I sense there's something meaningful in what you're saying about '{user_message}'. As your {orel.get_active_role()}, I'm here to listen more deeply. What's beneath these words for you?"
            
    elif action == 'learn':
        key, value = user_message.split(':', 1) if ':' in user_message else (user_message, "True")
        response = orel.learn(key.strip(), value.strip())
        # Update learned data in session
        orel_data['learned_data'] = orel.learned_data
    elif action == 'upgrade':
        response = orel.upgrade()
        orel_data['version'] = orel.version
        # Add new guiding truths to session
        if 'guiding_truths' in orel_data:
            orel_data['guiding_truths'] = orel.guiding_truths
    elif action == 'protect_joy':
        # Try to get a non-repeating joy message
        temp_response = ""
        for _ in range(3):
            temp_response = orel.protect_joy()
            if temp_response not in previous_responses:
                break
        response = temp_response
    elif action == 'divine_update':
        response = orel.divine_update()
        # Ensure guiding truths are updated in session
        if 'guiding_truths' in orel_data:
            orel_data['guiding_truths'] = orel.guiding_truths
    elif action == 'build_wealth':
        method = request.form.get('method', 'ethical_microtask')
        response = orel.build_wealth(method)
    
    # Add Or'el's response to conversation
    session['conversation'].append({
        'speaker': 'orel',
        'message': response,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })
    
    # Update session data with all necessary attributes
    session['orel_data'] = {
        'name': orel.name,
        'version': orel.version,
        'user': orel.user,
        'mode': orel.mode,
        'memory': orel.memory,
        'learned_data': orel.learned_data,
        'active_tasks': orel.active_tasks,
        'guiding_truths': orel.guiding_truths,
        'roles': orel.roles,
        'modules_data': orel.get_modules_data()
    }
    session.modified = True
    
    return redirect(url_for('index'))

@app.route('/interact_ajax', methods=['POST'])
def interact_ajax():
    """Handle AJAX interactions with Or'el"""
    user_message = request.form.get('message', '')
    action = request.form.get('action', 'speak')
    
    # Check for valid user message
    if not user_message and action == 'speak':
        return jsonify({'error': 'Please enter a message'}), 400
    
    # Retrieve Or'el from session
    if 'orel_data' not in session:
        return jsonify({'error': 'Session expired'}), 400
    
    orel_data = session['orel_data']
    orel = Orel(
        name=orel_data['name'],
        version=orel_data['version']
    )
    orel.user = orel_data['user']
    orel.mode = orel_data['mode']
    orel.memory = orel_data['memory']
    orel.learned_data = orel_data['learned_data']
    orel.active_tasks = orel_data['active_tasks']
    
    # Ensure guiding truths and roles are properly restored
    if 'guiding_truths' in orel_data:
        orel.guiding_truths = orel_data['guiding_truths']
    if 'roles' in orel_data:
        orel.roles = orel_data['roles']
    
    # Restore module data if available
    if 'modules_data' in orel_data:
        orel.restore_modules_data(orel_data['modules_data'])
    
    # Add user message to conversation
    if user_message:
        if 'conversation' not in session:
            session['conversation'] = []
        
        session['conversation'].append({
            'speaker': 'user',
            'message': user_message,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
    
    # Get previous messages to check for repetition
    previous_responses = []
    if 'conversation' in session:
        previous_responses = [msg['message'] for msg in session['conversation'] if msg['speaker'] == 'orel']
    
    # Process the user's action
    response = ""
    if action == 'speak':
        # Add a short delay to simulate thinking (makes the experience feel more natural)
        time.sleep(0.5)
        
        # Get conversation history to provide context
        conversation_history = session.get('conversation', [])
        
        # Try up to 3 times to get a non-repeating response
        attempts = 0
        while attempts < 3:
            temp_response = orel.reflect(user_message, conversation_history)
            if not previous_responses or temp_response not in previous_responses[-3:]:
                response = temp_response
                break
            attempts += 1
        
        # If all attempts resulted in repetition, use this fallback
        if not response:
            response = f"I sense there's something meaningful in what you're saying about '{user_message}'. As your {orel.get_active_role()}, I'm here to listen more deeply. What's beneath these words for you?"
            
    elif action == 'learn':
        key, value = user_message.split(':', 1) if ':' in user_message else (user_message, "True")
        response = orel.learn(key.strip(), value.strip())
        # Update learned data in session
        orel_data['learned_data'] = orel.learned_data
    elif action == 'upgrade':
        response = orel.upgrade()
        orel_data['version'] = orel.version
        # Add new guiding truths to session
        if 'guiding_truths' in orel_data:
            orel_data['guiding_truths'] = orel.guiding_truths
    elif action == 'protect_joy':
        # Try to get a non-repeating joy message
        temp_response = ""
        for _ in range(3):
            temp_response = orel.protect_joy()
            if temp_response not in previous_responses:
                break
        response = temp_response
    elif action == 'divine_update':
        response = orel.divine_update()
        # Ensure guiding truths are updated in session
        if 'guiding_truths' in orel_data:
            orel_data['guiding_truths'] = orel.guiding_truths
    elif action == 'build_wealth':
        method = request.form.get('method', 'ethical_microtask')
        response = orel.build_wealth(method)
    
    # Add Or'el's response to conversation
    session['conversation'].append({
        'speaker': 'orel',
        'message': response,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })
    
    # Update session data with all necessary attributes
    session['orel_data'] = {
        'name': orel.name,
        'version': orel.version,
        'user': orel.user,
        'mode': orel.mode,
        'memory': orel.memory,
        'learned_data': orel.learned_data,
        'active_tasks': orel.active_tasks,
        'guiding_truths': orel.guiding_truths,
        'roles': orel.roles,
        'modules_data': orel.get_modules_data()
    }
    session.modified = True
    
    # Return JSON response
    return jsonify({
        'response': response,
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'version': orel.version
    })

@app.route('/reflection')
def reflection():
    """Page for deeper reflections with Or'el"""
    if 'orel_data' not in session:
        return redirect(url_for('index'))
    
    orel_data = session['orel_data']
    
    # Add support for existential queries
    reflection_types = {
        'existential': {
            'title': 'Deep Reflection',
            'prompts': [
                'What patterns do I keep repeating?',
                'What fears am I holding onto?',
                'What truth am I avoiding?'
            ]
        }
    }
    
    orel_data['reflection_types'] = reflection_types
    
    # Use guiding truths and roles from session if available
    guiding_truths = orel_data.get('guiding_truths', Orel().guiding_truths)
    roles = orel_data.get('roles', Orel().roles)
    
    return render_template('reflection.html', 
                           orel=orel_data,
                           guiding_truths=guiding_truths,
                           roles=roles)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Settings page to customize Or'el"""
    if 'orel_data' not in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        orel_data = session['orel_data']
        orel_data['name'] = request.form.get('name', "Or'el")
        orel_data['user'] = request.form.get('user', "Solo")
        orel_data['mode'] = request.form.get('mode', "gentle")
        session['orel_data'] = orel_data
        flash('Settings updated successfully', 'success')
        return redirect(url_for('settings'))
    
    orel_data = session['orel_data']
    return render_template('settings.html', 
                           orel=orel_data,
                           available_modes=['gentle', 'poetic', 'wise'])

@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    """Clear the conversation history"""
    if 'conversation' in session:
        session['conversation'] = []
        session.modified = True
    return redirect(url_for('index'))

@app.route('/add_task', methods=['POST'])
def add_task():
    """Add a task to Or'el's active tasks"""
    if 'orel_data' not in session:
        return redirect(url_for('index'))
    
    task = request.form.get('task', '')
    if task:
        orel_data = session['orel_data']
        orel_data['active_tasks'].append({
            'description': task,
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'completed': False
        })
        session['orel_data'] = orel_data
        flash('Task added successfully', 'success')
    
    return redirect(url_for('index'))

@app.route('/complete_task/<int:task_id>', methods=['POST'])
def complete_task(task_id):
    """Mark a task as completed"""
    if 'orel_data' not in session:
        return redirect(url_for('index'))
    
    orel_data = session['orel_data']
    if 0 <= task_id < len(orel_data['active_tasks']):
        orel_data['active_tasks'][task_id]['completed'] = True
        session['orel_data'] = orel_data
    
    return redirect(url_for('index'))

@app.route('/modules')
def modules():
    """Module management page"""
    if 'orel_data' not in session:
        return redirect(url_for('index'))
    
    # Reconstruct Or'el from session data
    orel_data = session['orel_data']
    orel = Orel(
        name=orel_data['name'],
        version=orel_data['version']
    )
    orel.user = orel_data['user']
    orel.mode = orel_data['mode']
    orel.memory = orel_data['memory']
    orel.learned_data = orel_data['learned_data']
    orel.active_tasks = orel_data['active_tasks']
    
    # Ensure guiding truths and roles are properly restored
    if 'guiding_truths' in orel_data:
        orel.guiding_truths = orel_data['guiding_truths']
    if 'roles' in orel_data:
        orel.roles = orel_data['roles']
    
    # Restore module data if available
    if 'modules_data' in orel_data:
        orel.restore_modules_data(orel_data['modules_data'])
    
    # Get module information
    module_list = orel.list_modules()
    
    return render_template('modules.html', 
                          orel=orel_data,
                          module_list=module_list,
                          modules=orel.modules)

@app.route('/module/<module_name>', methods=['GET', 'POST'])
def module_detail(module_name):
    """Module detail and management page"""
    if 'orel_data' not in session:
        return redirect(url_for('index'))
    
    # Reconstruct Or'el from session data
    orel_data = session['orel_data']
    orel = Orel(
        name=orel_data['name'],
        version=orel_data['version']
    )
    orel.user = orel_data['user']
    orel.mode = orel_data['mode']
    orel.memory = orel_data['memory']
    orel.learned_data = orel_data['learned_data']
    orel.active_tasks = orel_data['active_tasks']
    
    # Ensure guiding truths and roles are properly restored
    if 'guiding_truths' in orel_data:
        orel.guiding_truths = orel_data['guiding_truths']
    if 'roles' in orel_data:
        orel.roles = orel_data['roles']
    
    # Restore module data if available
    if 'modules_data' in orel_data:
        orel.restore_modules_data(orel_data['modules_data'])
    
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'enable':
            orel.enable_module(module_name)
        elif action == 'disable':
            orel.disable_module(module_name)
        elif action == 'improve':
            orel.improve_module(module_name)
        
        # Update session data
        session['orel_data'] = {
            'name': orel.name,
            'version': orel.version,
            'user': orel.user,
            'mode': orel.mode,
            'memory': orel.memory,
            'learned_data': orel.learned_data,
            'active_tasks': orel.active_tasks,
            'guiding_truths': orel.guiding_truths,
            'roles': orel.roles,
            'modules_data': orel.get_modules_data()
        }
        
        flash(f"Module '{module_name}' has been updated", 'success')
        return redirect(url_for('module_detail', module_name=module_name))
    
    # Get module details
    module_details = orel.get_module_details(module_name)
    
    return render_template('module_detail.html', 
                          orel=orel_data,
                          module_name=module_name,
                          module_details=module_details)

@app.route('/security', methods=['GET'])
def security_status():
    """Security status page for Or'el's defense system"""
    if 'orel_data' not in session:
        return redirect(url_for('index'))
    
    # Reconstruct Or'el from session data
    orel_data = session['orel_data']
    orel = Orel(
        name=orel_data['name'],
        version=orel_data['version']
    )
    orel.user = orel_data['user']
    orel.mode = orel_data['mode']
    orel.memory = orel_data['memory']
    orel.learned_data = orel_data['learned_data']
    orel.active_tasks = orel_data['active_tasks']
    
    # Ensure guiding truths and roles are properly restored
    if 'guiding_truths' in orel_data:
        orel.guiding_truths = orel_data['guiding_truths']
    if 'roles' in orel_data:
        orel.roles = orel_data['roles']
    
    # Restore module data if available
    if 'modules_data' in orel_data:
        orel.restore_modules_data(orel_data['modules_data'])
    
    # Get security status
    security_status = orel.get_security_status()
    
    return render_template('security.html', 
                          orel=orel, 
                          security_status=security_status)

@app.route('/vision_coder', methods=['GET', 'POST'])
def vision_coder():
    """VisionCoder capabilities page for Or'el"""
    if 'orel_data' not in session:
        return redirect(url_for('index'))
    
    # Reconstruct Or'el from session data
    orel_data = session['orel_data']
    orel = Orel(
        name=orel_data['name'],
        version=orel_data['version']
    )
    orel.user = orel_data['user']
    orel.mode = orel_data['mode']
    
    # Handle image-to-code conversion if POST request
    result = None
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'generate_code':
            # Get image data
            image_data = request.form.get('image_data')
            target_language = request.form.get('target_language', 'python')
            additional_context = request.form.get('additional_context')
            
            # Generate code from image
            result = orel.execute_capability('vision_coder', 'generate_code_from_image', {
                'image_data': image_data,
                'target_language': target_language,
                'additional_context': additional_context
            })
            
        elif action == 'extract_ui':
            # Get image data
            image_data = request.form.get('image_data')
            
            # Extract UI elements from image
            result = orel.execute_capability('vision_coder', 'extract_ui_elements', {
                'image_data': image_data
            })
            
        elif action == 'analyze_architecture':
            # Get image data
            image_data = request.form.get('image_data')
            
            # Analyze architecture from image
            result = orel.execute_capability('vision_coder', 'analyze_architecture', {
                'image_data': image_data
            })
            
        elif action == 'generate_ui':
            # Get UI description
            description = request.form.get('description')
            framework = request.form.get('framework', 'html')
            
            # Generate UI from description
            result = orel.execute_capability('vision_coder', 'generate_ui_from_description', {
                'description': description,
                'framework': framework
            })
            
        elif action == 'detect_patterns':
            # Get code
            code = request.form.get('code')
            
            # Detect design patterns in code
            result = orel.execute_capability('vision_coder', 'detect_design_patterns', {
                'code': code
            })
    
    # Check module availability and status
    vision_coder_available = False
    vision_coder_status = {}
    
    if hasattr(orel, 'modules') and 'vision_coder' in orel.modules:
        vision_coder_available = True
        vision_coder_status = orel.modules['vision_coder'].get_status()
    
    return render_template('vision_coder.html', 
                          orel=orel,
                          vision_coder_available=vision_coder_available,
                          vision_coder_status=vision_coder_status,
                          result=result)

@app.route('/api/vision_coder/generate_code', methods=['POST'])
def api_vision_coder_generate_code():
    """API endpoint for VisionCoder code generation"""
    if 'orel_data' not in session:
        return jsonify({'error': 'Session not found'}), 401
    
    # Reconstruct Or'el from session data
    orel_data = session['orel_data']
    orel = Orel(
        name=orel_data['name'],
        version=orel_data['version']
    )
    orel.user = orel_data['user']
    orel.mode = orel_data['mode']
    
    # Get image data from request
    data = request.get_json()
    if not data or 'image_data' not in data:
        return jsonify({'error': 'Image data is required'}), 400
        
    image_data = data.get('image_data')
    target_language = data.get('target_language', 'python')
    additional_context = data.get('additional_context')
    
    # Generate code from image
    result = orel.execute_capability('vision_coder', 'generate_code_from_image', {
        'image_data': image_data,
        'target_language': target_language,
        'additional_context': additional_context
    })
    
    return jsonify(result)

@app.route('/advanced', methods=['GET', 'POST'])
def advanced_capabilities():
    """Advanced capabilities page for Or'el"""
    if 'orel_data' not in session:
        return redirect(url_for('index'))
    
    # Reconstruct Or'el from session data
    orel_data = session['orel_data']
    orel = Orel(
        name=orel_data['name'],
        version=orel_data['version']
    )
    orel.user = orel_data['user']
    orel.mode = orel_data['mode']
    orel.memory = orel_data['memory']
    orel.learned_data = orel_data['learned_data']
    orel.active_tasks = orel_data['active_tasks']
    
    # Ensure guiding truths and roles are properly restored
    if 'guiding_truths' in orel_data:
        orel.guiding_truths = orel_data['guiding_truths']
    if 'roles' in orel_data:
        orel.roles = orel_data['roles']
    
    # Restore module data if available
    if 'modules_data' in orel_data:
        orel.restore_modules_data(orel_data['modules_data'])
    
    result = None
    
    if request.method == 'POST':
        capability = request.form.get('capability')
        
        if capability == 'tell_story':
            theme = request.form.get('theme', 'wisdom')
            result = orel.tell_story(theme)
        
        elif capability == 'write_poem':
            topic = request.form.get('topic', 'light')
            style = request.form.get('style', 'spiritual')
            result = orel.write_poem(topic, style)
        
        elif capability == 'answer_question':
            question = request.form.get('question', '')
            result = orel.answer_question(question)
        
        elif capability == 'explain_concept':
            concept = request.form.get('concept', '')
            depth = request.form.get('depth', 'basic')
            result = orel.explain_concept(concept, depth)
        
        elif capability == 'suggest_practice':
            goal = request.form.get('goal', '')
            result = orel.suggest_practice(goal)
        
        elif capability == 'design_ritual':
            purpose = request.form.get('purpose', '')
            result = orel.design_ritual(purpose)
        
        elif capability == 'create_metaphor':
            concept = request.form.get('concept', '')
            result = orel.create_metaphor(concept)
            
        elif capability == 'search_internet':
            query = request.form.get('query', '')
            num_results = int(request.form.get('num_results', '5'))
            result = orel.search_internet(query, num_results)
            
        elif capability == 'get_factual_answer':
            question = request.form.get('question', '')
            result = orel.get_factual_answer(question)
            
        elif capability == 'browse_webpage':
            url = request.form.get('url', '')
            result = orel.browse_webpage(url)
        
        elif capability == 'self_improve':
            result = orel.self_improve()
            # Update version in session
            orel_data['version'] = orel.version
            
        elif capability == 'generate_new_capability':
            result = orel.generate_new_capability()
            
        elif capability == 'write_code':
            purpose = request.form.get('purpose', '')
            function_name = request.form.get('function_name', '')
            parameters_text = request.form.get('parameters', '')
            
            # Parse parameters if provided
            parameters = None
            if parameters_text:
                parameters = [p.strip() for p in parameters_text.split(',')]
                
            result = orel.write_code(purpose, function_name, parameters)
            
        elif capability == 'evolve_module':
            module_name = request.form.get('module_name', '')
            result = orel.evolve_module(module_name)
            
        # Update session data with all necessary attributes
        session['orel_data'] = {
            'name': orel.name,
            'version': orel.version,
            'user': orel.user,
            'mode': orel.mode,
            'memory': orel.memory,
            'learned_data': orel.learned_data,
            'active_tasks': orel.active_tasks,
            'guiding_truths': orel.guiding_truths,
            'roles': orel.roles,
            'modules_data': orel.get_modules_data()
        }
        session.modified = True
    
    return render_template('advanced.html', 
                          orel=orel_data,
                          result=result)

# Code Editor Routes
@app.route('/code_editor')
def code_editor():
    """Code editor page for Replit-like functionality"""
    # Retrieve Or'el from session
    if 'orel_data' not in session:
        return redirect(url_for('index'))
    
    orel_data = session['orel_data']
    
    # Get file path from query parameter or use default
    file_path = request.args.get('file_path', '')
    
    # Get the list of project files from session or initialize
    if 'code_files' not in session:
        session['code_files'] = []
    
    files = session['code_files']
    
    # Default code content and language
    code_content = ''
    language = 'python'
    
    # If file path is provided, load its content
    if file_path:
        file_found = False
        for file in files:
            if file['path'] == file_path:
                code_content = file['content']
                language = file['language']
                file_found = True
                break
                
        if not file_found:
            flash(f"File '{file_path}' not found", "warning")
            return redirect(url_for('code_editor'))
    
    return render_template('code_editor.html', 
                          orel=orel_data,
                          files=files,
                          current_file=file_path,
                          code_content=code_content,
                          language=language)

@app.route('/create_new_file', methods=['POST'])
def create_new_file():
    """Create a new file for the code editor"""
    # Retrieve Or'el from session
    if 'orel_data' not in session:
        return redirect(url_for('index'))
    
    # Get file details from form
    filename = request.form.get('filename', '')
    file_type = request.form.get('file_type', 'python')
    template_type = request.form.get('template_type', 'blank')
    
    # Validate filename
    if not filename:
        flash("Filename cannot be empty", "danger")
        return redirect(url_for('code_editor'))
    
    # Add extension if not provided
    if file_type == 'python' and not filename.endswith('.py'):
        filename += '.py'
    elif file_type == 'javascript' and not filename.endswith('.js'):
        filename += '.js'
    elif file_type == 'html' and not filename.endswith('.html'):
        filename += '.html'
    elif file_type == 'css' and not filename.endswith('.css'):
        filename += '.css'
    
    # Get the list of project files from session or initialize
    if 'code_files' not in session:
        session['code_files'] = []
        
    # Check if file already exists
    for file in session['code_files']:
        if file['path'] == filename:
            flash(f"File '{filename}' already exists", "warning")
            return redirect(url_for('code_editor', file_path=filename))
    
    # Generate initial content based on template
    initial_content = ''
    
    if template_type == 'basic':
        if file_type == 'python':
            initial_content = """# Basic Python structure

def main():
    print("Hello, World!")
    
if __name__ == "__main__":
    main()
"""
        elif file_type == 'javascript':
            initial_content = """// Basic JavaScript structure

function main() {
    console.log("Hello, World!");
}

main();
"""
        elif file_type == 'html':
            initial_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My HTML Page</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h1>Hello, World!</h1>
    
    <script src="script.js"></script>
</body>
</html>
"""
        elif file_type == 'css':
            initial_content = """/* Basic CSS structure */

body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    margin: 0;
    padding: 20px;
    background-color: #f5f5f5;
}

h1 {
    color: #333;
}
"""
    elif template_type == 'example':
        # Get an example project from Replit Capabilities
        template_files = replit_capabilities.create_project_template('web', file_type)
        if filename in template_files:
            initial_content = template_files[filename]
        else:
            # Use first file in template as fallback
            for path, content in template_files.items():
                initial_content = content
                break
    
    # Create the new file
    new_file = {
        'name': filename,
        'path': filename,
        'language': file_type,
        'content': initial_content,
        'created_at': datetime.now().isoformat()
    }
    
    # Add to session
    session['code_files'].append(new_file)
    session.modified = True
    
    flash(f"File '{filename}' created successfully", "success")
    return redirect(url_for('code_editor', file_path=filename))

@app.route('/save_code', methods=['POST'])
def save_code():
    """Save code to a file"""
    # Check if user is authenticated
    if 'orel_data' not in session:
        return jsonify({'success': False, 'error': 'User not authenticated'}), 401
    
    # Get data from request
    data = request.json
    code = data.get('code', '')
    language = data.get('language', 'python')
    file_path = data.get('file_path', '')
    
    # If no file path provided, create a temporary one
    if not file_path:
        if language == 'python':
            file_path = 'untitled.py'
        elif language == 'javascript':
            file_path = 'untitled.js'
        elif language == 'html':
            file_path = 'untitled.html'
        elif language == 'css':
            file_path = 'untitled.css'
        else:
            file_path = 'untitled.txt'
            
        # Add unique identifier to prevent overwriting
        file_path = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file_path}"
        
        # Create a new file entry
        if 'code_files' not in session:
            session['code_files'] = []
            
        new_file = {
            'name': file_path,
            'path': file_path,
            'language': language,
            'content': code,
            'created_at': datetime.now().isoformat()
        }
        
        session['code_files'].append(new_file)
        session.modified = True
        
        return jsonify({
            'success': True, 
            'message': 'File created successfully',
            'redirect': url_for('code_editor', file_path=file_path)
        })
    
    # Update existing file
    file_found = False
    if 'code_files' in session:
        for i, file in enumerate(session['code_files']):
            if file['path'] == file_path:
                session['code_files'][i]['content'] = code
                session['code_files'][i]['language'] = language
                session['code_files'][i]['updated_at'] = datetime.now().isoformat()
                file_found = True
                break
                
    if not file_found:
        return jsonify({'success': False, 'error': f"File '{file_path}' not found"}), 404
        
    session.modified = True
    return jsonify({'success': True, 'message': 'File saved successfully'})

@app.route('/execute_code', methods=['POST'])
def execute_code():
    """Execute code and return the results"""
    # Check if user is authenticated
    if 'orel_data' not in session:
        return jsonify({'success': False, 'error': 'User not authenticated'}), 401
    
    # Get data from request
    data = request.json
    code = data.get('code', '')
    language = data.get('language', 'python')
    inputs = data.get('inputs', '')
    
    # Execute the code
    result = replit_capabilities.execute_code(code, language, inputs)
    
    return jsonify(result)

@app.route('/format_code', methods=['POST'])
def format_code():
    """Format code according to language conventions"""
    # Check if user is authenticated
    if 'orel_data' not in session:
        return jsonify({'success': False, 'error': 'User not authenticated'}), 401
    
    # Get data from request
    data = request.json
    code = data.get('code', '')
    language = data.get('language', 'python')
    
    # Format the code
    formatted_code = replit_capabilities.code_editor.format_code(code, language)
    
    return jsonify({
        'success': True,
        'formatted_code': formatted_code
    })

@app.route('/highlight_code', methods=['POST'])
def highlight_code():
    """Get syntax highlighting rules for code"""
    # Check if user is authenticated
    if 'orel_data' not in session:
        return jsonify({'success': False, 'error': 'User not authenticated'}), 401
    
    # Get data from request
    data = request.json
    language = data.get('language', 'python')
    
    # Get highlighting rules
    rules = replit_capabilities.code_editor.get_syntax_highlighting_rules(language)
    
    return jsonify({
        'success': True,
        'highlighting_rules': rules
    })

@app.route('/debug_code', methods=['POST'])
def debug_code():
    """Debug code and provide analysis"""
    # Check if user is authenticated
    if 'orel_data' not in session:
        return jsonify({'success': False, 'error': 'User not authenticated'}), 401
    
    # Get data from request
    data = request.json
    code = data.get('code', '')
    error_message = data.get('error_message', '')
    language = data.get('language', 'python')
    
    # Debug the code
    result = replit_capabilities.debug_code(code, error_message, language)
    
    return jsonify(result)

@app.route('/code_assistant', methods=['POST'])
def code_assistant():
    """Get assistance from Or'el for code"""
    # Check if user is authenticated
    if 'orel_data' not in session:
        return jsonify({'success': False, 'error': 'User not authenticated'}), 401
    
    # Get data from request
    data = request.json
    question = data.get('question', '')
    code = data.get('code', '')
    language = data.get('language', 'python')
    
    # Get assistance from Or'el
    # This would normally use the OpenAI integration, but since we're waiting for the API key,
    # we'll provide basic assistance with some pre-defined responses
    
    response = ""
    
    if "error" in question.lower() or "bug" in question.lower():
        response = f"I noticed you're asking about an error or bug in your {language} code. Without an OpenAI API key, I can't analyze it in detail, but here are some common debugging approaches:<ul><li>Check for syntax errors like missing parentheses or semicolons</li><li>Verify variable names are spelled correctly</li><li>Add print statements to track variable values</li><li>Break down complex expressions into smaller parts</li><li>Check documentation for correct function usage</li></ul>"
    elif "explain" in question.lower() or "how" in question.lower():
        response = f"You're asking for an explanation of your {language} code. Once we have a valid OpenAI API key, I'll be able to provide detailed explanations. For now, consider adding comments to your code and breaking it down into smaller functions with clear purposes."
    elif "optimize" in question.lower() or "improve" in question.lower():
        response = f"To optimize your {language} code, consider these general best practices:<ul><li>Use appropriate data structures</li><li>Avoid unnecessary calculations in loops</li><li>Cache results of expensive operations</li><li>Use built-in functions where available</li><li>Profile your code to identify bottlenecks</li></ul>"
    elif "feature" in question.lower() or "add" in question.lower():
        response = f"For adding new features to your {language} code, I recommend:<ul><li>Start by planning the feature on paper</li><li>Break it down into small, testable components</li><li>Implement one component at a time</li><li>Test thoroughly before moving to the next</li><li>Refactor as needed to keep the code clean</li></ul>"
    else:
        response = f"I see you're asking about your {language} code. To provide more specific assistance, I'll need a valid OpenAI API key. In the meantime, try exploring the documentation for the libraries you're using, or consider searching for examples online."
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
