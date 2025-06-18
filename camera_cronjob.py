#!/usr/bin/env python3
"""
Master script to run all services simultaneously:
1. camera_cronjob.py - Captures snapshots from cameras
2. analyze_image.py - AI analysis of captured images
3. video_streaming.py - Available for motion detection when triggered

This script manages all processes and handles graceful shutdown.
"""

import os
import sys
import json
import time
import signal
import subprocess
import threading
from datetime import datetime
from pathlib import Path
import psutil

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment")

# Global variables for process management
processes = {}
running = True
shutdown_requested = False

class ServiceManager:
    """Manages multiple Python services"""
    
    def __init__(self):
        self.services = {}
        self.monitoring_thread = None
        
    def add_service(self, name, script_path, description, args=None):
        """Add a service to be managed"""
        self.services[name] = {
            'script_path': script_path,
            'description': description,
            'args': args or [],
            'process': None,
            'start_time': None,
            'restart_count': 0,
            'status': 'stopped'
        }
    
    def start_service(self, name):
        """Start a specific service"""
        if name not in self.services:
            print(f"❌ Service '{name}' not found")
            return False
            
        service = self.services[name]
        
        if service['process'] and service['process'].poll() is None:
            print(f"⚠️ Service '{name}' is already running")
            return True
            
        try:
            # Prepare command
            command = [sys.executable, service['script_path']] + service['args']
            
            print(f"🚀 Starting {name}: {service['description']}")
            print(f"   Command: {' '.join(command)}")
            
            # Start process
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            service['process'] = process
            service['start_time'] = datetime.now()
            service['status'] = 'running'
            
            print(f"✅ {name} started with PID: {process.pid}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start {name}: {e}")
            service['status'] = 'failed'
            return False
    
    def stop_service(self, name):
        """Stop a specific service"""
        if name not in self.services:
            return False
            
        service = self.services[name]
        process = service['process']
        
        if not process or process.poll() is not None:
            service['status'] = 'stopped'
            return True
            
        try:
            print(f"🛑 Stopping {name}...")
            
            # Try graceful shutdown first
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=5)
                print(f"✅ {name} stopped gracefully")
            except subprocess.TimeoutExpired:
                print(f"⚠️ {name} didn't stop gracefully, forcing...")
                process.kill()
                process.wait()
                print(f"✅ {name} forced to stop")
                
            service['status'] = 'stopped'
            service['process'] = None
            return True
            
        except Exception as e:
            print(f"❌ Error stopping {name}: {e}")
            return False
    
    def start_all_services(self):
        """Start all registered services"""
        print("🚀 Starting all services...")
        success_count = 0
        
        for name in self.services:
            if self.start_service(name):
                success_count += 1
                time.sleep(2)  # Stagger startup
        
        print(f"✅ Started {success_count}/{len(self.services)} services")
        return success_count == len(self.services)
    
    def stop_all_services(self):
        """Stop all running services"""
        print("🛑 Stopping all services...")
        
        for name in self.services:
            self.stop_service(name)
    
    def get_service_status(self, name):
        """Get status of a specific service"""
        if name not in self.services:
            return None
            
        service = self.services[name]
        process = service['process']
        
        if not process:
            return 'stopped'
        
        if process.poll() is None:
            return 'running'
        else:
            service['status'] = 'crashed'
            return 'crashed'
    
    def monitor_services(self):
        """Monitor services and restart if needed"""
        global running
        
        print("👁️ Starting service monitoring...")
        
        while running and not shutdown_requested:
            try:
                for name, service in self.services.items():
                    status = self.get_service_status(name)
                    
                    if status == 'crashed' and service['restart_count'] < 3:
                        print(f"💥 Service '{name}' crashed, restarting...")
                        service['restart_count'] += 1
                        self.start_service(name)
                        time.sleep(5)
                
                # Print status every 60 seconds
                time.sleep(60)
                self.print_status()
                
            except Exception as e:
                print(f"❌ Error in service monitoring: {e}")
                time.sleep(10)
        
        print("👁️ Service monitoring stopped")
    
    def print_status(self):
        """Print current status of all services"""
        print("\n" + "="*60)
        print(f"📊 Service Status - {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        
        for name, service in self.services.items():
            status = self.get_service_status(name)
            process = service['process']
            
            if status == 'running' and process:
                # Get memory usage
                try:
                    proc = psutil.Process(process.pid)
                    memory_mb = proc.memory_info().rss / 1024 / 1024
                    cpu_percent = proc.cpu_percent()
                    uptime = datetime.now() - service['start_time']
                    
                    print(f"✅ {name:20} | PID: {process.pid:6} | Memory: {memory_mb:6.1f}MB | CPU: {cpu_percent:5.1f}% | Uptime: {str(uptime).split('.')[0]}")
                except:
                    print(f"✅ {name:20} | PID: {process.pid:6} | Status: Running")
            else:
                print(f"❌ {name:20} | Status: {status}")
        
        print("="*60 + "\n")

def validate_environment():
    """Validate required environment variables and files"""
    print("🔍 Validating environment...")
    
    # Check required files
    required_files = [
        'camera_cronjob.py',
        'analyze_image.py', 
        'video_streaming.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    # Check required environment variables
    required_vars = [
        'SUPABASE_URL',
        'SUPABASE_KEY', 
        'IMOU_API_URL',
        'IMOU_APP_ID',
        'IMOU_APP_SECRET',
        'CAMERA_MAP',
        'OPENAI_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        return False
    
    # Validate CAMERA_MAP
    try:
        camera_map = json.loads(os.getenv('CAMERA_MAP', '{}'))
        if not camera_map:
            print("⚠️ WARNING: CAMERA_MAP is empty")
        else:
            print(f"✅ Found {len(camera_map)} cameras in CAMERA_MAP")
    except json.JSONDecodeError:
        print("❌ CAMERA_MAP is not valid JSON")
        return False
    
    print("✅ Environment validation passed")
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'httpx', 'supabase', 'dotenv', 'openai',
        'requests', 'cv2', 'mediapipe', 'numpy', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements_full.txt")
        return False
    
    print("✅ All dependencies are available")
    return True

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global running, shutdown_requested
    print(f"\n🛑 Received signal {signum}. Initiating graceful shutdown...")
    running = False
    shutdown_requested = True

def main():
    """Main function"""
    global running
    
    print("🚀 Multi-Service Camera System Manager")
    print("="*70)
    
    # Validate environment and dependencies
    if not validate_environment():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create service manager
    manager = ServiceManager()
    
    # Register services
    manager.add_service(
        'camera_cronjob',
        'camera_cronjob.py',
        'Camera Snapshot Cronjob - Captures images from cameras'
    )
    
    manager.add_service(
        'analyze_image',
        'analyze_image.py', 
        'AI Image Analysis - Processes captured images with OpenAI'
    )
    
    # Note: video_streaming.py is managed by analyze_image.py, not started directly
    print("📝 Services configured:")
    print("   🔄 camera_cronjob.py - Captures snapshots every 5 seconds")
    print("   🤖 analyze_image.py - AI analysis of captured images")
    print("   🎥 video_streaming.py - Triggered automatically by AI analysis")
    print()
    
    try:
        # Start all services
        if not manager.start_all_services():
            print("❌ Failed to start some services")
            sys.exit(1)
        
        print("✅ All services started successfully!")
        print("\n📊 System Overview:")
        print("   • Camera snapshots will be captured every 5 seconds")
        print("   • AI will analyze new images every 5 seconds") 
        print("   • Video streaming will auto-trigger on 'lying' behavior detection")
        print("   • Press Ctrl+C to stop all services")
        print("\n🔄 Starting monitoring loop...")
        print("="*70)
        
        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(
            target=manager.monitor_services,
            daemon=True
        )
        monitor_thread.start()
        
        # Print initial status
        time.sleep(3)
        manager.print_status()
        
        # Keep main thread alive
        while running and not shutdown_requested:
            time.sleep(1)
        
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        print("\n🔄 Shutting down all services...")
        manager.stop_all_services()
        print("✅ All services stopped")
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
