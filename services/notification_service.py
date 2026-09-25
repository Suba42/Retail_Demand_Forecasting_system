# ============================================================
# NOTIFICATION SERVICE
# Retail Demand Forecasting System
#
# EMAIL  -> Gmail SMTP
# SMS    -> FREE DEMO MODE
#
# SMS Demo Mode:
# - Does NOT use Twilio
# - Does NOT require payment
# - Records SMS as DEMO SENT
# - Allows the UI to demonstrate SMS notifications
# - Can later be replaced by a real SMS provider
# ============================================================

import os
import hashlib
import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv
from flask_login import current_user

from database.models import (
    db,
    NotificationSubscription,
    NotificationLog
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# EMAIL CONFIGURATION
# ============================================================

SMTP_HOST = os.getenv(
    "SMTP_HOST",
    "smtp.gmail.com"
)

SMTP_PORT = int(
    os.getenv(
        "SMTP_PORT",
        "587"
    )
)

SMTP_USERNAME = os.getenv(
    "SMTP_USERNAME",
    ""
)

SMTP_PASSWORD = os.getenv(
    "SMTP_PASSWORD",
    ""
)

SMTP_FROM_EMAIL = os.getenv(
    "SMTP_FROM_EMAIL",
    SMTP_USERNAME
)


# ============================================================
# SMS MODE
# ============================================================
#
# demo = FREE
#
# No Twilio credentials are required.
#
# Later, if you want real SMS, this can be changed to:
#
# SMS_MODE=twilio
#
# and a Twilio implementation can be added separately.
# ============================================================

SMS_MODE = os.getenv(
    "SMS_MODE",
    "demo"
).lower().strip()


# ============================================================
# HELPER: NORMALIZE SEVERITY
# ============================================================

def normalize_severity(severity):
    """
    Convert severity to a standard format.
    """

    if not severity:
        return "medium"

    return str(
        severity
    ).strip().lower()


# ============================================================
# HELPER: CREATE ALERT HASH
# ============================================================

def create_alert_hash(alert):
    """
    Create a unique hash for an alert.

    This is used to prevent the same alert from being
    repeatedly sent when the Smart Alerts page is refreshed.
    """

    alert_title = str(
        alert.get(
            "title",
            alert.get(
                "alert_title",
                "Smart Alert"
            )
        )
    )

    severity = normalize_severity(
        alert.get(
            "severity",
            "medium"
        )
    )

    message = str(
        alert.get(
            "message",
            alert.get(
                "description",
                ""
            )
        )
    )

    product = str(
        alert.get(
            "product",
            alert.get(
                "product_name",
                ""
            )
        )
    )

    raw_value = (
        f"{alert_title}|"
        f"{severity}|"
        f"{message}|"
        f"{product}"
    )

    return hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()


# ============================================================
# HELPER: CHECK SEVERITY PERMISSION
# ============================================================

def severity_enabled(
    subscription,
    severity
):
    """
    Check whether the user enabled notifications
    for the alert severity.
    """

    severity = normalize_severity(
        severity
    )

    if severity == "critical":
        return subscription.critical_enabled

    if severity == "high":
        return subscription.high_enabled

    if severity == "medium":
        return subscription.medium_enabled

    if severity == "low":
        return subscription.low_enabled

    return False


# ============================================================
# HELPER: CHECK DUPLICATE NOTIFICATION
# ============================================================

def notification_already_sent(
    user_id,
    alert_hash,
    channel
):
    """
    Check whether this exact alert has already been
    processed for the specified channel.
    """

    existing = (
        NotificationLog.query
        .filter_by(
            user_id=user_id,
            alert_hash=alert_hash,
            channel=channel
        )
        .first()
    )

    return existing is not None


# ============================================================
# EMAIL: SEND EMAIL
# ============================================================

def send_email_notification(
    recipient_email,
    subject,
    message
):
    """
    Send a real email using Gmail SMTP.
    """

    if not recipient_email:
        return {
            "success": False,
            "status": "failed",
            "error": "No email address provided."
        }

    if not SMTP_USERNAME:
        return {
            "success": False,
            "status": "failed",
            "error": "SMTP_USERNAME is not configured."
        }

    if not SMTP_PASSWORD:
        return {
            "success": False,
            "status": "failed",
            "error": "SMTP_PASSWORD is not configured."
        }

    try:

        email_message = MIMEMultipart()

        email_message[
            "From"
        ] = SMTP_FROM_EMAIL

        email_message[
            "To"
        ] = recipient_email

        email_message[
            "Subject"
        ] = subject

        email_message.attach(
            MIMEText(
                message,
                "plain"
            )
        )

        with smtplib.SMTP(
            SMTP_HOST,
            SMTP_PORT,
            timeout=30
        ) as server:

            server.starttls()

            server.login(
                SMTP_USERNAME,
                SMTP_PASSWORD
            )

            server.sendmail(
                SMTP_FROM_EMAIL,
                recipient_email,
                email_message.as_string()
            )

        return {
            "success": True,
            "status": "sent",
            "error": None
        }

    except Exception as e:

        print(
            f"[EMAIL ERROR] {str(e)}"
        )

        return {
            "success": False,
            "status": "failed",
            "error": str(e)
        }


# ============================================================
# SMS: FREE DEMO MODE
# ============================================================

def send_sms_notification(
    recipient_phone,
    message
):
    """
    FREE SMS DEMO MODE.

    This does NOT send a real SMS through a telecom carrier.

    Instead, it records the notification as a successful
    demonstration notification.

    This is useful for the final-year project demo because
    the complete Smart Alerts -> SMS workflow can be shown
    without depending on a paid SMS provider.
    """

    if not recipient_phone:

        return {
            "success": False,
            "status": "failed",
            "error": "No phone number provided."
        }

    # --------------------------------------------------------
    # FREE DEMO MODE
    # --------------------------------------------------------

    if SMS_MODE == "demo":

        print(
            "\n"
            "====================================================\n"
            "             FREE SMS DEMO MODE\n"
            "====================================================\n"
            f"To      : {recipient_phone}\n"
            f"Message : {message}\n"
            "Status  : DEMO SENT SUCCESSFULLY\n"
            "====================================================\n"
        )

        return {
            "success": True,
            "status": "demo_sent",
            "error": None
        }

    # --------------------------------------------------------
    # UNKNOWN MODE
    # --------------------------------------------------------

    return {
        "success": False,
        "status": "failed",
        "error": (
            f"Unsupported SMS_MODE: {SMS_MODE}"
        )
    }


# ============================================================
# HELPER: SAVE NOTIFICATION LOG
# ============================================================

def save_notification_log(
    user_id,
    alert_hash,
    severity,
    alert_title,
    channel,
    status,
    error_message=None
):
    """
    Save notification delivery information
    into NotificationLog.
    """

    try:

        log = NotificationLog(
            user_id=user_id,
            alert_hash=alert_hash,
            severity=severity,
            alert_title=alert_title,
            channel=channel,
            status=status,
            error_message=error_message
        )

        db.session.add(
            log
        )

        db.session.commit()

        return True

    except Exception as e:

        db.session.rollback()

        print(
            f"[NOTIFICATION LOG ERROR] {str(e)}"
        )

        return False


# ============================================================
# MAIN NOTIFICATION DISPATCHER
# ============================================================

def dispatch_alert_notifications(
    alerts,
    user_id
):
    """
    Process all Smart Alerts for the logged-in user.

    Email:
        Real Gmail SMTP notification.

    SMS:
        Free demo notification.

    Duplicate alerts are skipped using alert_hash.
    """

    result = {
        "enabled": False,
        "alerts_processed": 0,
        "notifications_sent": 0,
        "notifications_skipped": 0,
        "notifications_failed": 0,
        "details": []
    }

    # ========================================================
    # VALIDATE ALERT LIST
    # ========================================================

    if not alerts:

        return result

    # ========================================================
    # GET USER SUBSCRIPTION
    # ========================================================

    subscription = (
        NotificationSubscription.query
        .filter_by(
            user_id=user_id
        )
        .first()
    )

    if subscription is None:

        print(
            "[NOTIFICATION] "
            "No notification subscription found."
        )

        return result

    result["enabled"] = True

    # ========================================================
    # PROCESS EACH ALERT
    # ========================================================

    for alert in alerts:

        result[
            "alerts_processed"
        ] += 1

        # ----------------------------------------------------
        # ALERT INFORMATION
        # ----------------------------------------------------

        severity = normalize_severity(
            alert.get(
                "severity",
                "medium"
            )
        )

        alert_title = str(
            alert.get(
                "title",
                alert.get(
                    "alert_title",
                    "Smart Alert"
                )
            )
        )

        alert_message = str(
            alert.get(
                "message",
                alert.get(
                    "description",
                    "A Smart Alert has been generated."
                )
            )
        )

        alert_hash = create_alert_hash(
            alert
        )

        # ----------------------------------------------------
        # CHECK SEVERITY
        # ----------------------------------------------------

        if not severity_enabled(
            subscription,
            severity
        ):

            result[
                "notifications_skipped"
            ] += 1

            result[
                "details"
            ].append(
                {
                    "alert": alert_title,
                    "severity": severity,
                    "channel": "all",
                    "status": "skipped",
                    "reason": (
                        "Severity notification "
                        "is disabled."
                    )
                }
            )

            continue

        # ====================================================
        # EMAIL NOTIFICATION
        # ====================================================

        if (
            subscription.email_enabled
            and subscription.email
        ):

            if notification_already_sent(
                user_id,
                alert_hash,
                "email"
            ):

                result[
                    "notifications_skipped"
                ] += 1

                result[
                    "details"
                ].append(
                    {
                        "alert": alert_title,
                        "severity": severity,
                        "channel": "email",
                        "status": "skipped",
                        "reason": (
                            "Already sent."
                        )
                    }
                )

            else:

                email_subject = (
                    f"[{severity.upper()}] "
                    f"Retail Demand Alert"
                )

                email_body = (
                    f"Retail Demand Forecasting System\n\n"
                    f"Alert: {alert_title}\n"
                    f"Severity: {severity.upper()}\n\n"
                    f"{alert_message}\n\n"
                    f"This notification was generated "
                    f"by the Smart Alerts module."
                )

                email_result = (
                    send_email_notification(
                        recipient_email=subscription.email,
                        subject=email_subject,
                        message=email_body
                    )
                )

                if email_result[
                    "success"
                ]:

                    save_notification_log(
                        user_id=user_id,
                        alert_hash=alert_hash,
                        severity=severity,
                        alert_title=alert_title,
                        channel="email",
                        status="sent"
                    )

                    result[
                        "notifications_sent"
                    ] += 1

                    result[
                        "details"
                    ].append(
                        {
                            "alert": alert_title,
                            "severity": severity,
                            "channel": "email",
                            "status": "sent"
                        }
                    )

                else:

                    save_notification_log(
                        user_id=user_id,
                        alert_hash=alert_hash,
                        severity=severity,
                        alert_title=alert_title,
                        channel="email",
                        status="failed",
                        error_message=email_result[
                            "error"
                        ]
                    )

                    result[
                        "notifications_failed"
                    ] += 1

                    result[
                        "details"
                    ].append(
                        {
                            "alert": alert_title,
                            "severity": severity,
                            "channel": "email",
                            "status": "failed",
                            "error": email_result[
                                "error"
                            ]
                        }
                    )

        # ====================================================
        # SMS NOTIFICATION
        # ====================================================

        if (
            subscription.sms_enabled
            and subscription.phone
        ):

            if notification_already_sent(
                user_id,
                alert_hash,
                "sms"
            ):

                result[
                    "notifications_skipped"
                ] += 1

                result[
                    "details"
                ].append(
                    {
                        "alert": alert_title,
                        "severity": severity,
                        "channel": "sms",
                        "status": "skipped",
                        "reason": (
                            "Already sent."
                        )
                    }
                )

            else:

                sms_message = (
                    f"RETAIL ALERT - "
                    f"{severity.upper()}\n"
                    f"{alert_title}\n"
                    f"{alert_message}"
                )

                sms_result = (
                    send_sms_notification(
                        recipient_phone=subscription.phone,
                        message=sms_message
                    )
                )

                if sms_result[
                    "success"
                ]:

                    save_notification_log(
                        user_id=user_id,
                        alert_hash=alert_hash,
                        severity=severity,
                        alert_title=alert_title,
                        channel="sms",
                        status="demo_sent"
                    )

                    result[
                        "notifications_sent"
                    ] += 1

                    result[
                        "details"
                    ].append(
                        {
                            "alert": alert_title,
                            "severity": severity,
                            "channel": "sms",
                            "status": "demo_sent",
                            "phone": subscription.phone
                        }
                    )

                else:

                    save_notification_log(
                        user_id=user_id,
                        alert_hash=alert_hash,
                        severity=severity,
                        alert_title=alert_title,
                        channel="sms",
                        status="failed",
                        error_message=sms_result[
                            "error"
                        ]
                    )

                    result[
                        "notifications_failed"
                    ] += 1

                    result[
                        "details"
                    ].append(
                        {
                            "alert": alert_title,
                            "severity": severity,
                            "channel": "sms",
                            "status": "failed",
                            "error": sms_result[
                                "error"
                            ]
                        }
                    )

    # ========================================================
    # RETURN FINAL RESULT
    # ========================================================

    return result