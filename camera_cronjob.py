#!/usr/bin/env python3
"""
Camera Snapshot Cronjob - Pure Python Background Service
No HTTP server, just runs scheduled tasks like original Node.js cron jobs
"""

import asyncio
import json
import os
import time
import hashlib
import random
import string
import signal
import sys
import threading
from datetime import datetime
from typing import Dict, Any

import httpx
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Parse camera map from environment
try:
    CAMERA_MAP = json.loads(os.getenv('CAMERA_MAP', '{}'))
    if not CAMERA_MAP:
        print("❌ CAMERA_MAP is empty or not found in environment variables")
        sys.exit(1)
except json.JSONDecodeError:
    print("❌ Error parsing CAMERA_MAP from environment variables")
    sys.exit(1)

# Supabase setup
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')

if not supabase_url or not supabase_key:
    print("❌ SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
    sys.exit(1)

print(supabase_url, supabase_key)
supabase: Client = create_client(supabase_url, supabase_key)

# Configuration object
config = {
    'url': os.getenv('IMOU_API_URL'),
    'app_id': os.getenv('IMOU_APP_ID'),
    'app_secret': os.getenv('IMOU_APP_SECRET'),
    'table_name': os.getenv('POSTGRES_TABLE_NAME', 'images')
}

# Validate required configuration
required_config = ['url', 'app_id', 'app_secret']
missing_config = [key for key in required_config if not config[key]]
if missing_config:
    print(f"❌ Missing required environment variables: {', '.join(missing_config)}")
    sys.exit(1)

# Global control variable
running = True

# Helper functions
def get_time() -> Dict[str, Any]:
    """Get current timestamp in both Unix and ISO format"""
    now = datetime.now()
    return {
        'time_stamp': int(now.timestamp()),
        'iso_time_stamp': now.isoformat()
    }

def generate_nonce() -> str:
    """Generate a random nonce string"""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))

def calc_sign(timestamp: int, nonce: str, app_secret: str) -> str:
    """Calculate MD5 signature for API authentication"""
    sign_string = f"time:{timestamp},nonce:{nonce},appSecret:{app_secret}"
    return hashlib.md5(sign_string.encode()).hexdigest()

def generate_id() -> int:
    """Generate a random ID between 1 and 50"""
    return random.randint(1, 50)

def prepare_system_params() -> Dict[str, Any]:
    """Prepare common system parameters for API requests"""
    time_data = get_time()
    nonce = generate_nonce()
    sign = calc_sign(time_data['time_stamp'], nonce, config['app_secret'])
    request_id = generate_id()
    
    return {
        'time': time_data['time_stamp'],
        'nonce': nonce,
        'sign': sign,
        'id': request_id
    }

# API Functions
async def get_access_token() -> str:
    """Get access token from IMOU API"""
    params = prepare_system_params()
    
    request_body = {
        'system': {
            'ver': '1.0',
            'sign': params['sign'],
            'appId': config['app_id'],
            'time': params['time'],
            'nonce': params['nonce']
        },
        'params': {},
        'id': params['id']
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{config['url']}/accessToken", json=request_body)
            response.raise_for_status()
            
            data = response.json()
            if data and data.get('result', {}).get('code') == '0':
                return data['result']['data']['accessToken']
            else:
                raise Exception(f"Failed to get access token: {json.dumps(data)}")
                
    except Exception as e:
        print(f"❌ Error getting access token: {e}")
        raise e

async def get_snapshot(access_token: str, device_id: str) -> Dict[str, Any]:
    """Get device snapshot from IMOU API"""
    params = prepare_system_params()
    
    request_body = {
        'system': {
            'ver': '1.0',
            'sign': params['sign'],
            'appId': config['app_id'],
            'time': params['time'],
            'nonce': params['nonce']
        },
        'params': {
            'token': access_token,
            'deviceId': device_id,
            'channelId': '0'
        },
        'id': params['id']
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{config['url']}/setDeviceSnapEnhanced", json=request_body)
            response.raise_for_status()
            
            data = response.json()
            if data and data.get('result', {}).get('code') == '0':
                return data['result']['data']
            else:
                raise Exception(f"Failed to get device snapshot: {json.dumps(data)}")
                
    except Exception as e:
        print(f"❌ Error getting device snapshot: {e}")
        raise e

async def initialize_database() -> None:
    """Initialize and verify database tables"""
    try:
        print("🔍 Verifying database tables...")
        
        # Check images table
        try:
            response = supabase.table(config['table_name']).select('id').limit(1).execute()
            print(f"✅ TABLE {config['table_name']} verified successfully")
        except Exception as e:
            if 'PGRST116' in str(e):
                print(f"❌ Table {config['table_name']} does not exist. Please create it in Supabase dashboard.")
            else:
                raise e
        
        # Check patients table
        try:
            response = supabase.table('patients').select('patient_id').limit(1).execute()
            print("✅ TABLE patients verified successfully")
        except Exception as e:
            if 'PGRST116' in str(e):
                print("❌ Table patients does not exist. Please create it in Supabase dashboard.")
            else:
                raise e
        
        # Check patient_records table
        try:
            response = supabase.table('patient_records').select('id').limit(1).execute()
            print("✅ TABLE patient_records verified successfully")
        except Exception as e:
            if 'PGRST116' in str(e):
                print("❌ Table patient_records does not exist. Please create it in Supabase dashboard.")
            else:
                raise e
        
        print("✅ Database verification completed")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        raise e

async def process_snapshot(device_id: str, room_name: str) -> None:
    """Main function to process camera snapshot"""
    try:
        print(f"📸 Processing snapshot for {room_name} (camera: {device_id})")
        
        # Step 1: Get access token
        access_token = await get_access_token()
        
        # Step 2: Get device snapshot
        snapshot = await get_snapshot(access_token, device_id)
        
        # Step 3: Save to Supabase
        now = get_time()
        data = {
            'id': now['time_stamp'],
            'url': snapshot.get('url'),
            'created_at': now['iso_time_stamp'],
            'camera_imou_id': device_id
        }
        
        response = supabase.table(config['table_name']).insert(data).execute()
        
        if hasattr(response, 'error') and response.error:
            raise Exception(f"Supabase error: {response.error}")
        
        print(f"✅ Snapshot saved for {room_name}: {snapshot.get('url')}")
        
    except Exception as e:
        print(f"❌ Error processing snapshot for {room_name}: {e}")

# Async wrapper for sync context
def run_async_task(coro):
    """Helper to run async tasks in sync context"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# Scheduled job functions
def room2_snapshot_job():
    """Scheduled job for room2 camera"""
    if "room2" in CAMERA_MAP and "imou" in CAMERA_MAP["room2"]:
        device_id = CAMERA_MAP["room2"]["imou"]
        run_async_task(process_snapshot(device_id, "room2"))
    else:
        print("❌ Room2 camera configuration not found in CAMERA_MAP")

def room3_snapshot_job():
    """Scheduled job for room3 camera"""
    if "room3" in CAMERA_MAP and "imou" in CAMERA_MAP["room3"]:
        device_id = CAMERA_MAP["room3"]["imou"]
        run_async_task(process_snapshot(device_id, "room3"))
    else:
        print("❌ Room3 camera configuration not found in CAMERA_MAP")

# Simple cron-like scheduler
def cron_scheduler():
    """Simple cron scheduler - runs jobs at specified intervals"""
    global running
    
    print("⏰ Starting cron scheduler...")
    last_room2_time = 0
    last_room3_time = 0
    
    while running:
        current_time = time.time()
        
        try:
            # Room2 every 5 seconds
            if current_time - last_room2_time >= 5:
                print(f"🔄 [{datetime.now().strftime('%H:%M:%S')}] Running room2 snapshot job...")
                room2_snapshot_job()
                last_room2_time = current_time
            
            # Room3 every 5 seconds (offset by 2.5 seconds to avoid conflicts)
            if current_time - last_room3_time >= 5 and (current_time % 10) >= 2.5:
                print(f"🔄 [{datetime.now().strftime('%H:%M:%S')}] Running room3 snapshot job...")
                room3_snapshot_job()
                last_room3_time = current_time
            
        except Exception as e:
            print(f"❌ Error in scheduler: {e}")
        
        # Sleep for 500ms to avoid high CPU usage
        time.sleep(0.5)

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global running
    print(f"\n🛑 Received signal {signum}. Shutting down gracefully...")
    running = False
    print("✅ Shutdown complete")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Main function
async def main():
    """Main application function"""
    print("🚀 Camera Snapshot Cronjob Service")
    print("=" * 50)
    
    try:
        # Test database connection
        print("🔍 Testing database connection...")
        response = supabase.table(config['table_name']).select('id').limit(1).execute()
        
        if hasattr(response, 'error') and response.error:
            raise Exception(f"Database connection failed: {response.error}")
        
        print("✅ Successfully connected to Supabase")
        
        # Initialize database
        await initialize_database()
        
        print(f"\n📷 Camera configuration: {len(CAMERA_MAP)} cameras found")
        for room, config_data in CAMERA_MAP.items():
            print(f"   📹 {room}: {config_data.get('imou', 'No IMOU ID')}")
        
        print(f"\n⏰ Schedule configuration:")
        print(f"   📸 Room2: Every 5 seconds")
        print(f"   📸 Room3: Every 5 seconds (offset)")
        print(f"\n🛑 Press Ctrl+C to stop")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return False

if __name__ == "__main__":
    try:
        # Initialize application
        success = run_async_task(main())
        
        if success:
            # Start the cron scheduler
            cron_scheduler()
        else:
            print("❌ Failed to start application")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Service stopped by user")
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)
