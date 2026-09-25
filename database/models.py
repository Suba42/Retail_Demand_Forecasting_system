from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin


# =========================================================
# DATABASE
# =========================================================

db = SQLAlchemy()


# =========================================================
# USER MODEL
# =========================================================

class User(UserMixin, db.Model):

    __tablename__ = "users"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(150),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(255),
        nullable=False
    )

    role = db.Column(
        db.String(50),
        nullable=False,
        default="Business Analyst"
    )

    created_at = db.Column(
        db.DateTime,
        server_default=db.func.current_timestamp()
    )


# =========================================================
# EMAIL NOTIFICATION SUBSCRIPTION
# =========================================================

class NotificationSubscription(db.Model):

    __tablename__ = "notification_subscriptions"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False,
        unique=True
    )

    # =====================================================
    # EMAIL
    # =====================================================

    email = db.Column(
        db.String(255),
        nullable=True
    )

    email_enabled = db.Column(
        db.Boolean,
        default=True,
        nullable=False
    )

    # =====================================================
    # ALERT PRIORITIES
    # =====================================================

    critical_enabled = db.Column(
        db.Boolean,
        default=True,
        nullable=False
    )

    high_enabled = db.Column(
        db.Boolean,
        default=True,
        nullable=False
    )

    medium_enabled = db.Column(
        db.Boolean,
        default=False,
        nullable=False
    )

    low_enabled = db.Column(
        db.Boolean,
        default=False,
        nullable=False
    )

    # =====================================================
    # TIMESTAMPS
    # =====================================================

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    # =====================================================
    # RELATIONSHIP
    # =====================================================

    user = db.relationship(
        "User",
        backref=db.backref(
            "notification_subscription",
            uselist=False
        )
    )
# =========================================================
# NOTIFICATION LOG
# =========================================================

class NotificationLog(db.Model):

    __tablename__ = "notification_logs"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False
    )

    alert_hash = db.Column(
        db.String(64),
        nullable=False,
        index=True
    )

    severity = db.Column(
        db.String(20),
        nullable=False
    )

    alert_title = db.Column(
        db.String(255),
        nullable=True
    )

    channel = db.Column(
        db.String(20),
        nullable=False,
        default="email"
    )

    status = db.Column(
        db.String(20),
        nullable=False
    )

    error_message = db.Column(
        db.Text,
        nullable=True
    )

    sent_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    user = db.relationship(
        "User",
        backref=db.backref(
            "notification_logs",
            lazy=True
        )
    )