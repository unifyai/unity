#!/usr/bin/env python3
"""
Script to add credits to the current user account.

This script:
1. Fetches the user_id using the UNIFY_KEY from .env
2. Adds credits to that user_id using the ORCHESTRA_ADMIN_KEY from .env
"""

import argparse
import os
import sys
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global configuration fetched from .env
ORCHESTRA_URL = os.getenv("ORCHESTRA_URL")
API_KEY = os.getenv("UNIFY_KEY")
ADMIN_KEY = os.getenv("ORCHESTRA_ADMIN_KEY")

def get_user_id():
    """
    Fetches the user_id using the global API_KEY.
    """
    if not ORCHESTRA_URL or not API_KEY:
        print("❌ Error: ORCHESTRA_URL or UNIFY_KEY not set in .env.")
        return None

    url = f"{ORCHESTRA_URL.rstrip('/')}/user/basic-info"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get("user_id")
        else:
            print(f"❌ Failed to fetch user_id: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error fetching user_id: {e}")
        return None

def add_credits(user_id, quantity):
    """
    Calls the Orchestra admin API to add credits using global ADMIN_KEY.
    """
    if not ORCHESTRA_URL or not ADMIN_KEY:
        print("❌ Error: UNIFY_BASE_URL or ORCHESTRA_ADMIN_KEY not set in .env.")
        return False

    # Construct the admin endpoint URL
    url = f"{ORCHESTRA_URL.rstrip('/')}/admin/create_recharge"
    
    payload = {
        "user_id": user_id,
        "quantity": quantity,
        "type": "promo"
    }
    
    headers = {
        "Authorization": f"Bearer {ADMIN_KEY}"
    }

    print(f"🚀 Sending request to {url}...")

    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            print(f"✅ Successfully added {quantity} credits to User {user_id}.")
            return True
        else:
            print(f"❌ Failed to add credits: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during request: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Add credits to the user account",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--quantity",
        type=float,
        required=True,
        help="Number of credits to add",
    )

    args = parser.parse_args()

    # 1. Get User ID
    print("🔍 Fetching User ID...")
    user_id = get_user_id()
    
    if not user_id:
        print("❌ Could not resolve User ID from UNIFY_KEY.")
        sys.exit(1)
    
    print(f"👤 Resolved User ID: {user_id}")

    # 2. Add Credits
    success = add_credits(
        user_id=user_id,
        quantity=args.quantity
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
