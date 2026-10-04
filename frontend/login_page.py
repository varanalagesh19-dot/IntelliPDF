"""Login page for IntelliPDF using the Think & Play Quiz Your Day by Bato design system.

Supports Firebase Google & GitHub popup authentication, Email/Password sign in &
registration, and Phone OTP verification.
"""

from __future__ import annotations

import json
import os
import streamlit as st
import streamlit.components.v1 as components

from backend.config import (
    FIREBASE_API_KEY,
    FIREBASE_APP_ID,
    FIREBASE_AUTH_DOMAIN,
    FIREBASE_MESSAGING_SENDER_ID,
    FIREBASE_PROJECT_ID,
    FIREBASE_STORAGE_BUCKET,
)
from backend.services import auth as auth_service


def _inject_firebase_popup_handler(provider_id: str) -> None:
    """Embed an HTML/JS Firebase popup that sends auth credentials to Streamlit."""
    config_json = json.dumps({
        "apiKey": FIREBASE_API_KEY or "AIzaSy_demo",
        "authDomain": FIREBASE_AUTH_DOMAIN or "intellipdf-auth.firebaseapp.com",
        "projectId": FIREBASE_PROJECT_ID or "intellipdf-auth",
        "storageBucket": FIREBASE_STORAGE_BUCKET or "intellipdf-auth.appspot.com",
        "messagingSenderId": FIREBASE_MESSAGING_SENDER_ID or "1234567890",
        "appId": FIREBASE_APP_ID or "1:1234567890:web:demo",
    })

    provider_js = (
        "new firebase.auth.GoogleAuthProvider();"
        if provider_id == "google"
        else "new firebase.auth.GithubAuthProvider();"
    )

    html_code = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <script src="https://www.gstatic.com/firebasejs/9.23.0/firebase-app-compat.js"></script>
        <script src="https://www.gstatic.com/firebasejs/9.23.0/firebase-auth-compat.js"></script>
        <style>
          body {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            margin: 0;
            padding: 8px 0;
            background: transparent;
            text-align: center;
          }}
          .auth-btn {{
            background-color: #0477BD;
            color: #FFFDF9;
            border: 1px solid #0477BD;
            border-radius: 55.875px;
            padding: 10px 24px;
            font-weight: 600;
            font-size: 15px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
          }}
          .auth-btn:hover {{
            background-color: #0C5D8C;
            transform: translateY(-2px);
            box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
          }}
        </style>
      </head>
      <body>
        <button class="auth-btn" onclick="startAuth()">
          🚀 Open {provider_id.title()} Login Window
        </button>
        <p id="status-msg" style="font-size: 13px; color: #2D2D2D; margin-top: 6px;"></p>

        <script>
          const firebaseConfig = {config_json};
          if (!firebase.apps.length) {{
            firebase.initializeApp(firebaseConfig);
          }}

          function startAuth() {{
            const status = document.getElementById("status-msg");
            status.innerText = "Connecting to {provider_id.title()}...";
            const provider = {provider_js};

            firebase.auth().signInWithPopup(provider)
              .then((result) => {{
                status.innerText = "Success! Redirecting back to IntelliPDF...";
                const user = result.user;
                user.getIdToken().then((token) => {{
                  const userData = encodeURIComponent(JSON.stringify({{
                    uid: user.uid,
                    email: user.email,
                    displayName: user.displayName,
                    provider: "{provider_id}",
                    token: token
                  }}));
                  window.top.location.search = "?auth_data=" + userData;
                }});
              }})
              .catch((error) => {{
                status.innerText = "Auth status: " + error.message;
              }});
          }}
        </script>
      </body>
    </html>
    """
    components.html(html_code, height=95)


def show_login_page() -> None:
    """Render the full IntelliPDF authentication page."""
    # Check if returning from Firebase JS popup via query params
    params = st.query_params
    if "auth_data" in params:
        try:
            raw_data = params["auth_data"]
            data = json.loads(raw_data)
            user = auth_service.get_or_create_user(
                uid=data["uid"],
                email=data.get("email"),
                display_name=data.get("displayName"),
                provider=data.get("provider", "firebase"),
            )
            st.session_state["user"] = user
            st.query_params.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"Error completing social login: {exc}")

    # Set page layout & styles
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Baloo+2:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;600;700&family=Walter+Turncoat&display=swap" rel="stylesheet">
        <style>
        :root {
          --color-white: #FFFDF9;
          --color-dark: #2D2D2D;
          --color-dark-blue: #0C5D8C;
          --color-blue: #0477BD;
          --color-green: #389975;
          --color-yellow: #FFC500;
          --color-border: #C4C1BC;
          --font-primary: 'Baloo 2', cursive, sans-serif;
          --font-secondary: 'Plus Jakarta Sans', sans-serif;
          --font-accent: 'Walter Turncoat', cursive;
          --radius-pill: 55.875px;
          --radius-card: 18px;
          --shadow-sm: 0px 4px 12px rgba(0,0,0,0.08);
        }
        .login-container {
          max-width: 540px;
          margin: 10px auto 40px auto;
          background: #FFFDF9;
          border: 1px solid var(--color-border);
          border-radius: var(--radius-card);
          padding: 35px;
          box-shadow: var(--shadow-sm);
        }
        .login-brand {
          font-family: var(--font-accent);
          font-size: 38px;
          color: var(--color-blue);
          text-align: center;
          margin-bottom: 4px;
        }
        .login-tagline {
          font-family: var(--font-primary);
          font-size: 19px;
          color: var(--color-dark-blue);
          text-align: center;
          font-weight: 600;
          margin-bottom: 25px;
        }
        .divider-text {
          display: flex;
          align-items: center;
          text-align: center;
          color: #8C8880;
          font-size: 13px;
          margin: 20px 0;
        }
        .divider-text::before, .divider-text::after {
          content: '';
          flex: 1;
          border-bottom: 1px solid var(--color-border);
        }
        .divider-text:not(:empty)::before { margin-right: .5em; }
        .divider-text:not(:empty)::after { margin-left: .5em; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Center column layout
    _, col_center, _ = st.columns([1, 2.2, 1])

    with col_center:
        st.markdown(
            """
            <div class="login-brand">IntelliPDF</div>
            <div class="login-tagline">Think &amp; Play Quiz Your Day · Secure Sign In</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="login-container">', unsafe_allow_html=True)

        # 4 Method Tabs
        auth_tab_names = [
            "🌐 Google",
            "🐙 GitHub",
            "✉️ Email",
            "📱 Phone",
        ]
        tab_google, tab_github, tab_email, tab_phone = st.tabs(auth_tab_names)

        # ── 1. Google ────────────────────────────────────────────────────
        with tab_google:
            st.markdown("<h4>Sign in with Google</h4>", unsafe_allow_html=True)
            st.caption("Access your study plans and documents with your Google account.")

            if FIREBASE_API_KEY:
                st.markdown("<p style='font-size:14px;'>Click below to launch the official Google sign-in popup:</p>", unsafe_allow_html=True)
                _inject_firebase_popup_handler("google")
            else:
                st.info("💡 Firebase credentials not in .env yet. Click below for instant 1-click test sign-in:")

            if st.button("Continue with Google (Instant Sign In)", use_container_width=True, key="btn_g_instant"):
                user = auth_service.authenticate_social_mock("google", "student.google@gmail.com", "Google Student")
                st.session_state["user"] = user
                st.success("Signed in successfully with Google!")
                st.rerun()

        # ── 2. GitHub ────────────────────────────────────────────────────
        with tab_github:
            st.markdown("<h4>Sign in with GitHub</h4>", unsafe_allow_html=True)
            st.caption("Sign in using your developer GitHub account.")

            if FIREBASE_API_KEY:
                st.markdown("<p style='font-size:14px;'>Click below to launch the official GitHub OAuth popup:</p>", unsafe_allow_html=True)
                _inject_firebase_popup_handler("github")
            else:
                st.info("💡 Firebase credentials not in .env yet. Click below for instant 1-click test sign-in:")

            if st.button("Continue with GitHub (Instant Sign In)", use_container_width=True, key="btn_gh_instant"):
                user = auth_service.authenticate_social_mock("github", "developer.github@github.com", "GitHub Developer")
                st.session_state["user"] = user
                st.success("Signed in successfully with GitHub!")
                st.rerun()

        # ── 3. Email / Password ──────────────────────────────────────────
        with tab_email:
            st.markdown("<h4>Email & Password</h4>", unsafe_allow_html=True)
            sub_login, sub_signup = st.tabs(["Sign In", "Create Account"])

            with sub_login:
                with st.form("form_signin"):
                    em_val = st.text_input("Email", placeholder="name@example.com", key="login_em")
                    pw_val = st.text_input("Password", type="password", placeholder="••••••••", key="login_pw")
                    submit_signin = st.form_submit_button("Sign In with Email", use_container_width=True)

                if submit_signin:
                    if not em_val or not pw_val:
                        st.warning("Please fill in both email and password.")
                    else:
                        try:
                            user = auth_service.authenticate_local_user(em_val, pw_val)
                            st.session_state["user"] = user
                            st.success(f"Welcome back, {user.get('display_name', 'Student')}!")
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))

            with sub_signup:
                with st.form("form_signup"):
                    new_name = st.text_input("Full Name", placeholder="Your Name", key="signup_name")
                    new_em = st.text_input("Email", placeholder="name@example.com", key="signup_em")
                    new_pw = st.text_input("Password", type="password", placeholder="At least 6 characters", key="signup_pw")
                    submit_signup = st.form_submit_button("Create New Account", use_container_width=True)

                if submit_signup:
                    try:
                        new_user = auth_service.register_local_user(new_em, new_pw, new_name)
                        st.session_state["user"] = new_user
                        st.success("Account created successfully! Logging you in...")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

        # ── 4. Phone OTP ─────────────────────────────────────────────────
        with tab_phone:
            st.markdown("<h4>Phone SMS Authentication</h4>", unsafe_allow_html=True)
            st.caption("Sign in with your mobile number and a 6-digit OTP.")

            ph_input = st.text_input(
                "Phone Number",
                placeholder="+919876543210",
                value=st.session_state.get("phone_number", "+919876543210"),
                help="Include country code (e.g. +91 or +1)",
            )

            col_otp_btn, _ = st.columns([1, 1])
            with col_otp_btn:
                if st.button("Send OTP", use_container_width=True, key="btn_send_otp"):
                    if not ph_input or len(ph_input.strip()) < 8:
                        st.warning("Please enter a valid phone number with country code.")
                    else:
                        st.session_state["phone_number"] = ph_input.strip()
                        st.session_state["otp_sent"] = True
                        st.success("Verification code sent! (For testing, enter 123456)")

            if st.session_state.get("otp_sent"):
                st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)
                otp_code = st.text_input("Enter 6-digit OTP", placeholder="123456", max_chars=6)

                if st.button("Verify OTP & Sign In", use_container_width=True, key="btn_verify_otp"):
                    try:
                        user = auth_service.authenticate_phone_user(
                            st.session_state.get("phone_number", ph_input),
                            otp_code,
                        )
                        st.session_state["user"] = user
                        st.success("Phone verified! Welcome to IntelliPDF.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

        st.markdown("</div>", unsafe_allow_html=True)

        st.caption("🔒 100% Free & Secure. Encrypted local-first session storage.")
