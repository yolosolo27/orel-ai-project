"""
Database models for Or'el

This module defines the database models for storing Or'el's data including:
- User profiles
- Conversation history
- Calendar events and reminders
- Learning data and system settings
- Advanced learning mechanisms
- Multi-model AI capabilities
- Training and feedback systems
"""

from datetime import datetime
from app import db
from sqlalchemy.dialects.postgresql import JSON
from flask_login import UserMixin

class User(UserMixin, db.Model):
    """User model for authentication and personalization"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    # User preferences
    preferred_role = db.Column(db.String(50), default="Guide")
    preferred_mode = db.Column(db.String(50), default="gentle")
    timezone = db.Column(db.String(50), default="UTC")
    
    # Relationships
    conversations = db.relationship('Conversation', backref='user', lazy=True)
    calendar_events = db.relationship('CalendarEvent', backref='user', lazy=True)
    reminders = db.relationship('Reminder', backref='user', lazy=True)
    learned_data = db.relationship('LearnedData', backref='user', lazy=True)
    
    def __repr__(self):
        return f'<User {self.username}>'


class Conversation(db.Model):
    """Store conversation sessions with Or'el"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    started_at = db.Column(db.DateTime, default=datetime.now)
    title = db.Column(db.String(255))
    
    # Conversation metadata
    orel_role = db.Column(db.String(50))
    orel_mode = db.Column(db.String(50))
    sentiment_score = db.Column(db.Float)
    
    # Relationships
    messages = db.relationship('Message', backref='conversation', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Conversation {self.id} with User {self.user_id}>'


class Message(db.Model):
    """Individual messages in a conversation"""
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    sender = db.Column(db.String(20), nullable=False)  # 'user' or 'orel'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    
    # Message metadata
    emotion = db.Column(db.String(50))
    intent = db.Column(db.String(50))
    processed_by = db.Column(db.String(50))  # 'openai', 'module', 'fallback', etc.
    
    def __repr__(self):
        return f'<Message {self.id} from {self.sender}>'


class LearnedData(db.Model):
    """Data learned by Or'el from user interactions"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    key = db.Column(db.String(255), nullable=False)
    value = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Category for organization
    category = db.Column(db.String(50), default='general')
    
    # Unique constraint per user and key
    __table_args__ = (db.UniqueConstraint('user_id', 'key', name='unique_user_key'),)
    
    def __repr__(self):
        return f'<LearnedData {self.key} for User {self.user_id}>'


class Task(db.Model):
    """Tasks and intentions tracked by Or'el"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    description = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    completed = db.Column(db.Boolean, default=False)
    completed_at = db.Column(db.DateTime)
    
    # Task metadata
    priority = db.Column(db.Integer, default=1)  # 1-5 scale
    category = db.Column(db.String(50), default='general')
    due_date = db.Column(db.DateTime)
    
    def __repr__(self):
        return f'<Task {self.id}: {self.description[:20]}{"..." if len(self.description) > 20 else ""}>'


class CalendarEvent(db.Model):
    """Calendar events managed by Or'el"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime)
    description = db.Column(db.Text)
    location = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    # Event metadata
    category = db.Column(db.String(50), default='general')
    is_recurring = db.Column(db.Boolean, default=False)
    recurrence_rule = db.Column(db.String(255))  # iCal format
    
    def __repr__(self):
        return f'<CalendarEvent {self.id}: {self.title}>'


class Reminder(db.Model):
    """Reminders set by the user through Or'el"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    reminder_time = db.Column(db.DateTime, nullable=False)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.now)
    completed = db.Column(db.Boolean, default=False)
    
    def __repr__(self):
        return f'<Reminder {self.id}: {self.title}>'


class SocialMediaAccount(db.Model):
    """Social media accounts connected to Or'el"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    platform = db.Column(db.String(50), nullable=False)  # 'twitter', 'linkedin', etc.
    username = db.Column(db.String(100))
    access_token = db.Column(db.String(255))
    token_secret = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.now)
    last_used = db.Column(db.DateTime)
    
    # Unique constraint per user and platform
    __table_args__ = (db.UniqueConstraint('user_id', 'platform', name='unique_user_platform'),)
    
    def __repr__(self):
        return f'<SocialMediaAccount {self.platform} for User {self.user_id}>'


class SystemModule(db.Model):
    """Or'el system modules and their status"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    description = db.Column(db.Text)
    version = db.Column(db.String(20), default='1.0.0')
    enabled = db.Column(db.Boolean, default=True)
    capabilities = db.Column(JSON)  # Store module capabilities as JSON
    last_updated = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f'<SystemModule {self.name} v{self.version}>'


class WebSource(db.Model):
    """Trusted web sources for Or'el information retrieval"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    url = db.Column(db.String(255), nullable=False, unique=True)
    category = db.Column(db.String(50))
    trust_level = db.Column(db.Integer, default=3)  # 1-5 scale
    last_accessed = db.Column(db.DateTime)
    
    def __repr__(self):
        return f'<WebSource {self.name}>'


class SystemLog(db.Model):
    """System logs for tracking Or'el's operations"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    level = db.Column(db.String(20), default='INFO')  # 'INFO', 'WARNING', 'ERROR', etc.
    module = db.Column(db.String(50))
    message = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    
    def __repr__(self):
        return f'<SystemLog {self.level}: {self.message[:30]}{"..." if len(self.message) > 30 else ""}>'


class APIKey(db.Model):
    """API keys for external services used by Or'el"""
    id = db.Column(db.Integer, primary_key=True)
    service = db.Column(db.String(50), nullable=False, unique=True)
    key_name = db.Column(db.String(50), nullable=False)
    key_value = db.Column(db.String(255))  # Consider encryption in production
    is_active = db.Column(db.Boolean, default=True)
    last_used = db.Column(db.DateTime)
    
    def __repr__(self):
        return f'<APIKey for {self.service}>'


class UserInteraction(db.Model):
    """Detailed tracking of user interactions for learning purposes"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_id = db.Column(db.String(100))  # Link to a learning session
    input_text = db.Column(db.Text)
    response_text = db.Column(db.Text)
    interaction_type = db.Column(db.String(50), default='conversation')  # conversation, code, image, etc.
    timestamp = db.Column(db.DateTime, default=datetime.now)
    embedding = db.Column(db.Text)  # JSON string of vector embedding
    interaction_data = db.Column(db.Text)  # JSON string with additional interaction data
    
    # Feedback data
    feedback_score = db.Column(db.Float)
    feedback_text = db.Column(db.Text)
    feedback_timestamp = db.Column(db.DateTime)
    
    # Relationships
    user = db.relationship('User', backref='interactions')
    
    def __repr__(self):
        return f'<UserInteraction {self.id} by User {self.user_id}>'


class LearningSession(db.Model):
    """Sessions for tracking and improving Or'el's learning"""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.now, nullable=False)
    end_time = db.Column(db.DateTime)
    context = db.Column(db.Text)  # JSON string with session context
    active = db.Column(db.Boolean, default=True)
    performance_data = db.Column(db.Text)  # JSON string with performance metrics
    
    # Relationships
    user = db.relationship('User', backref='learning_sessions')
    
    def __repr__(self):
        return f'<LearningSession {self.session_id}>'


class ModelPerformance(db.Model):
    """Performance tracking for AI models used by Or'el"""
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    interaction_id = db.Column(db.Integer, db.ForeignKey('user_interaction.id'))
    score = db.Column(db.Float)  # Performance score (0-1)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    performance_data = db.Column(db.Text)  # JSON string with additional performance data
    
    # Relationships
    interaction = db.relationship('UserInteraction', backref='model_performances')
    
    def __repr__(self):
        return f'<ModelPerformance {self.id} for {self.model_name}>'


class AIModel(db.Model):
    """Information about AI models available to Or'el"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    provider = db.Column(db.String(50), nullable=False)  # openai, anthropic, google, etc.
    model_type = db.Column(db.String(50))  # text, vision, multimodal, etc.
    version = db.Column(db.String(50))
    capabilities = db.Column(JSON)  # Array of capabilities
    parameters = db.Column(JSON)  # Default parameters
    performance_rating = db.Column(db.Float, default=0.0)  # 0-1 scale
    active = db.Column(db.Boolean, default=True)
    
    def __repr__(self):
        return f'<AIModel {self.name} by {self.provider}>'


class UserFeedback(db.Model):
    """Explicit feedback from users about Or'el's responses"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    interaction_id = db.Column(db.Integer, db.ForeignKey('user_interaction.id'))
    rating = db.Column(db.Integer)  # 1-5 star rating
    comment = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    
    # Relationships
    user = db.relationship('User', backref='feedback')
    interaction = db.relationship('UserInteraction', backref='feedback')
    
    def __repr__(self):
        return f'<UserFeedback {self.id} by User {self.user_id}>'


class KnowledgeDomain(db.Model):
    """Knowledge domains for specialized Or'el capabilities"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    description = db.Column(db.Text)
    parent_domain_id = db.Column(db.Integer, db.ForeignKey('knowledge_domain.id'))
    knowledge_base = db.Column(JSON)  # Core knowledge references
    learning_rate = db.Column(db.Float, default=0.01)  # How quickly this domain learns
    
    # Relationships
    parent = db.relationship('KnowledgeDomain', remote_side=[id], backref='subdomains')
    
    def __repr__(self):
        return f'<KnowledgeDomain {self.name}>'


class ExperimentalModelTemp(db.Model):
    """Temporary table for migration"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    
class ExperimentalModel(db.Model):
    """Experimental AI models being developed by Or'el"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    description = db.Column(db.Text)
    base_model = db.Column(db.String(100))
    active = db.Column(db.Boolean, default=True, nullable=False)
    architecture = db.Column(db.String(100))  # Transformer, MLP, CNN, etc.
    training_progress = db.Column(db.Float, default=0.0)  # 0-1 scale
    parameters = db.Column(JSON)  # Model hyperparameters
    evaluation_metrics = db.Column(JSON)  # Performance metrics
    created_at = db.Column(db.DateTime, default=datetime.now)
    last_trained = db.Column(db.DateTime)
    
    def __repr__(self):
        return f'<ExperimentalModel {self.name}>'


class TrainingDataset(db.Model):
    """Datasets used for training Or'el's experimental models"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    description = db.Column(db.Text)
    domain = db.Column(db.String(100))
    source = db.Column(db.String(255))
    record_count = db.Column(db.Integer)
    format = db.Column(db.String(50))  # json, csv, text, images, etc.
    dataset_info = db.Column(JSON)  # Dataset characteristics
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f'<TrainingDataset {self.name}>'