"""
One-time Device Code flow to get a refresh token for Steph's M365 account.

Run this once, log in when prompted, and save the refresh token.
The refresh token lasts 90 days and auto-renews on use.

Usage:
    uv run get_refresh_token.py
"""

import requests
import time
from pathlib import Path

try:
    from .m365_auth import DEFAULT_SCOPES, load_tenant_and_client
except ImportError:
    from m365_auth import DEFAULT_SCOPES, load_tenant_and_client

DIR = Path(__file__).parent
TENANT_ID, CLIENT_ID = load_tenant_and_client()
SCOPES = DEFAULT_SCOPES

print("=" * 60)
print("Microsoft 365 — Device Code Authentication")
print("=" * 60)

r = requests.post(
    f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/devicecode",
    data={"client_id": CLIENT_ID, "scope": SCOPES},
)

if r.status_code != 200:
    print(f"Error: {r.status_code} {r.text[:300]}")
    exit(1)

data = r.json()
print(f"\n{data['message']}\n")
print("Waiting for you to complete the login...")

interval = data.get("interval", 5)
while True:
    time.sleep(interval)
    r = requests.post(
        f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token",
        data={
            "client_id": CLIENT_ID,
            "device_code": data["device_code"],
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        },
    )
    if r.status_code == 200:
        tokens = r.json()
        refresh_token = tokens["refresh_token"]
        access_token = tokens["access_token"]

        token_path = DIR / "refresh_token.txt"
        token_path.write_text(refresh_token)

        print("\nAuthentication successful!")
        print(f"Refresh token saved to: {token_path}")
        print(f"Refresh token length: {len(refresh_token)} chars")

        print("\nVerifying access...")
        headers = {"Authorization": f"Bearer {access_token}"}
        r2 = requests.get("https://graph.microsoft.com/v1.0/me", headers=headers)
        if r2.status_code == 200:
            user = r2.json()
            print(
                f"  Logged in as: {user['displayName']} ({user['userPrincipalName']})",
            )

        r3 = requests.get(
            "https://graph.microsoft.com/v1.0/me/onenote/notebooks",
            headers=headers,
        )
        if r3.status_code == 200:
            notebooks = r3.json().get("value", [])
            print(f"  OneNote access: OK ({len(notebooks)} notebooks)")
            for nb in notebooks:
                print(f"    - {nb['displayName']}")
        else:
            print(f"  OneNote access: FAILED ({r3.status_code})")

        print("\nDone! You can now use this refresh token for all M365 access.")
        break

    error = r.json().get("error", "")
    if error == "authorization_pending":
        print(".", end="", flush=True)
    elif error == "expired_token":
        print("\nThe code expired. Please run the script again.")
        exit(1)
    else:
        print(f"\nUnexpected error: {error} — {r.json().get('error_description', '')}")
        exit(1)
