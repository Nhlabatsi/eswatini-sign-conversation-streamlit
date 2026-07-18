"""
turn.py
========

Provides ICE server config (STUN + TURN) for streamlit-webrtc, using
Twilio's Network Traversal Service to generate short-lived TURN
credentials.

WHY THIS IS NEEDED: Streamlit Community Cloud's infrastructure does
not allow WebRTC connections to establish reliably using STUN alone
-- this is confirmed directly in streamlit-webrtc's own official
sample code/docs, not just something specific to this app. A TURN
relay server is required as a fallback for connections STUN can't
traverse (which turns out to be most connections on this specific
platform, not just the usual ~8-10% of restrictive-NAT cases).

Free/static TURN services (e.g. Open Relay Project) were considered
and explicitly ruled out by streamlit-webrtc's own maintainers as
unreliable. Twilio's TURN service is the maintainer-recommended
approach; it has a free tier sufficient for testing/light use.

SETUP REQUIRED (see deployment guide):
  1. Sign up for a free Twilio account: https://www.twilio.com/try-twilio
  2. Get your Account SID and Auth Token from the Twilio Console.
  3. Add them as Streamlit Cloud secrets (App settings -> Secrets):
         TWILIO_ACCOUNT_SID = "..."
         TWILIO_AUTH_TOKEN = "..."
"""

import logging
import os

import streamlit as st
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

logger = logging.getLogger(__name__)


def _get_twilio_credentials():
    """Prefer Streamlit secrets (the standard mechanism on Community
    Cloud); fall back to plain environment variables for local runs."""
    try:
        return st.secrets["TWILIO_ACCOUNT_SID"], st.secrets["TWILIO_AUTH_TOKEN"]
    except Exception:
        return os.environ.get("TWILIO_ACCOUNT_SID"), os.environ.get("TWILIO_AUTH_TOKEN")


@st.cache_data(ttl=3000)  # Twilio tokens are valid ~24h; refresh well before that
def get_ice_servers():
    """Returns a list of ICE server dicts suitable for RTCConfiguration.
    Falls back to Google's public STUN-only server (with a warning) if
    Twilio credentials aren't configured -- this fallback is known to
    be unreliable on Streamlit Community Cloud specifically, but keeps
    the app from crashing outright if secrets aren't set up yet."""
    account_sid, auth_token = _get_twilio_credentials()

    if not account_sid or not auth_token:
        logger.warning(
            "TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN not configured -- falling back "
            "to STUN-only, which is known to be unreliable on Streamlit Community "
            "Cloud. Add Twilio credentials in App settings -> Secrets."
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    try:
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        return token.ice_servers
    except TwilioRestException as e:
        logger.error(f"Failed to fetch Twilio ICE servers: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]
