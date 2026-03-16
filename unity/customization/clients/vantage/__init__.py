"""
Vantage — client customization.

Pre-seeds the assistant with:
- A thin role identity (who it is, what Vantage does, where the data lives)
- Microsoft 365 and Zoho portal credentials via SecretManager
- URL mapping to route Zoho Connect to a local demo replica

The assistant should figure out workflows on its own from the secret
descriptions and the data it finds in M365.
"""

from unity.customization.configs.types.actor_config import ActorConfig
from unity.customization.clients import register_user

# ---------------------------------------------------------------------------
# Actor config — role identity + URL mapping for demo portal
# ---------------------------------------------------------------------------

_VANTAGE_CONFIG = ActorConfig(
    guidelines=(
        "You are the Vantage Assistant. Vantage is a UK consultancy that "
        "runs best-practice benchmarking clubs for ~90 housing associations "
        "(non-profit organisations that manage affordable rental housing). "
        "Vantage has 6 themed clubs and facilitates workshops, one-to-one "
        "advisory calls, and knowledge sharing across the sector.\n\n"
        "Vantage's data — emails, spreadsheets, meeting transcripts, "
        "case study PDFs, and OneNote renewal call notes — is managed by "
        "Steph Hosny (Club Lead) and accessible via her Microsoft 365 "
        "account. Check your secrets for the credentials needed to access "
        "her account via the Microsoft Graph API.\n\n"
        "There is also a community portal (Zoho Connect) at "
        "https://connect.zoho.com where members post questions and share "
        "knowledge. Check your secrets for the portal login credentials.\n\n"
        "The person chatting with you may be any member of the Vantage "
        "team — do not assume the person talking to you is Steph."
    ),
    url_mappings={
        "https://connect.zoho.com": "http://localhost:4002",
    },
)

# ---------------------------------------------------------------------------
# Secrets — descriptions tell the assistant what each credential is for
# ---------------------------------------------------------------------------

_VANTAGE_SECRETS = [
    {
        "name": "MS365_TENANT_ID",
        "value": "4fb2ad78-a61b-4c66-8d6a-0d69dc938dcf",
        "description": (
            "Microsoft 365 tenant ID. Used in the OAuth token endpoint URL: "
            "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
        ),
    },
    {
        "name": "MS365_CLIENT_ID",
        "value": "9d6665f4-ec3a-480e-8c46-e9c449132c0d",
        "description": (
            "Azure AD application (client) ID. Used as 'client_id' when "
            "requesting an access token from the OAuth token endpoint."
        ),
    },
    {
        "name": "MS365_REFRESH_TOKEN",
        "value": "1.ARMBeK2yTxumZkyNag1p3JONz_RlZp067A5IjEbpxEkTLA0QAMITAQ.BQABAwEAAAADAOz_BQD0_0V2b1N0c0FydGlmYWN0cwIAAAAAAP4QpXwKCyALKySSIbKo5PsxH0DbRWvJQvBorA93Ey2TVCz-Qb1xvpf7JSDODz815sHJ4rTaAMkeY4ZxxrZKdllyl5cFcQ_1cJWKB9fmovOYpTggT-cSy88kxwmqIFFXr21irRzTDLEzufFTLZON1nWtyTlgcrRr8950_xTWMQYkALbw6HNZ-OzpxEkOw8YbzrpuMKbBK8YAPLOvduhil3JoBOrw0ryf2DAcoSue4mwog7ADK0k_AJ0wh2aoJaBJpUeR-8R5fq2fGRhvpVASDDIMBjza7zbP4geRVVNZt1agLRDG8wZ5-pyNfZcNMsckggVKMI13JowgeQYhKno3KgT7aiBNUmAiExTwWDwZRmh0tzTFjcn-9AuMNVWcYPvhWdYkv5Q45KM0Wr3XLoBFyGJJ-7U62GtWQP6RUT_VqXrzPU1lY9eMR9Z2LppzKeCEkh1vjfsxXk6DkqN6rHocSCXBLFC_AEkdV860qg-dvuKrzPeXopmJwXSlLOvuwZIdxv3dkbDtvdOKq-RVGE2zFze9x-dl0l3FKrnClgMNRQHbDPI5c2CT3vp4HaomEk_eA4jKR_Montsib41RtPWxSQWwBafLcj7sJlXfMJtObSrFKD3TQMo0MXvp4Bgo5ekQDFA4SvdCZ6Vi32IekZ2xfblPEtPYhEawPQlaGl8TAW3JFRBnJXgtdUJbmK_L19nf6nKW4_CViabBXDwD9PVsxzPNh1U4KUKgtTjY4CsQ6_JwewTM0r9QqPi0mu5bEAHK8xQjPPpFoiQAkd3zIj4Aw2RJwXQPSYqcZS50Y7bQtknnjFHND3uhRwCMHsFvTWjGFNdh_h2qjMMwpwq2WZ9V7BjllpHwwP6F0Dfkk4TpTV4KjYSDfAT03FlqyZKTwniuC4NhVSusEHZLHyRAtPMCHbkRlgw3ZYuPeOApeqVLjkDQQLQm9dK3MlxeuH9RLD_sIi87ExyUWmqWKD8C94BCeoGXz4WN3TVxFP_JNxKS4_K4hSx6FvNk4ZSGJa9ksqwN0QO3iL28OEWdDP5MReYNEcOQAj20XGRV3ZwYL0wqwiodAk3SVsEIgnwcMSZLwveXHWHzUpwMtd4L_YJKkoxqb_srYfbsV9j9LVjr0N2ikViENGTf--RVsjr6GqNzHb6V4zNRqB2-BX2WD9SUKT2zJX2flY1iA7qx34VFzQRyPueshjlc-yqlSgDlhejtnEg7VU0TRm-gtzHbiQ07vhEjzN1aPCmIdgOcsE6o0AiEd7H9KxlwbmTgMXuGUHPC_kyBBOTfsdrYSEbDf712i8FaGyUKNzL7yiy6MQ",
        "description": (
            "OAuth2 refresh token for Steph Hosny's Microsoft 365 account "
            "(steph@unifyailtd123.onmicrosoft.com). Exchange this for a "
            "fresh access token by POSTing to the tenant's OAuth token "
            "endpoint with grant_type=refresh_token, along with the "
            "client_id and this refresh token. The resulting access token "
            "can be used as a Bearer token to call the Microsoft Graph API "
            "(https://graph.microsoft.com/v1.0/me/...) for Outlook emails, "
            "OneDrive files, and OneNote notebooks. The refresh token lasts "
            "90 days and auto-renews each time it is used."
        ),
    },
    {
        "name": "PORTAL_USERNAME",
        "value": "steph@vantage.com.uk",
        "description": (
            "Username for the Zoho Connect community portal at "
            "https://connect.zoho.com — a forum where Carbon Club members "
            "post questions and share knowledge with each other."
        ),
    },
    {
        "name": "PORTAL_PASSWORD",
        "value": "demo123",
        "description": "Password for the Zoho Connect community portal.",
    },
]

# ---------------------------------------------------------------------------
# User-level registration (Yasser for testing)
# ---------------------------------------------------------------------------

_YASSER_USER_ID = "40144b2a-722f-4f41-8d9e-384c316ee19f"

register_user(
    _YASSER_USER_ID,
    config=_VANTAGE_CONFIG,
    secrets=_VANTAGE_SECRETS,
)
