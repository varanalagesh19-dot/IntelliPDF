import uuid
import pytest
from backend.services import auth
from backend.utils import db as db_module


def test_local_registration_and_login():
    """Test user registration and subsequent login with password hash."""
    db_module.init_db()
    test_email = f"tester_{uuid.uuid4().hex[:8]}@intellipdf.test"
    test_pwd = "password123"

    user = auth.register_local_user(test_email, test_pwd, "Test Student")
    assert user["email"] == test_email
    assert user["display_name"] == "Test Student"

    auth_user = auth.authenticate_local_user(test_email, test_pwd)
    assert auth_user["uid"] == user["uid"]

    # Wrong password should fail
    with pytest.raises(ValueError, match="Incorrect password"):
        auth.authenticate_local_user(test_email, "wrongpassword")


def test_phone_authentication():
    """Test phone verification with testing OTP."""
    db_module.init_db()
    user = auth.authenticate_phone_user("+919999999999", "123456")
    assert user["phone"] == "+919999999999"
    assert user["provider"] == "phone"

    with pytest.raises(ValueError, match="Invalid verification code"):
        auth.authenticate_phone_user("+919999999999", "999999")


def test_social_authentication():
    """Test 1-click social sign in."""
    db_module.init_db()
    user = auth.authenticate_social_mock("google", "google.user@example.com", "Google User")
    assert user["email"] == "google.user@example.com"
    assert user["provider"] == "google"
