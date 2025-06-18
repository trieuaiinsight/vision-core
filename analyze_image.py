import openai
import os
import time
import json
import logging
import requests
import re
import sys
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from io import BytesIO
from collections import defaultdict
import subprocess
import threading
import keyboard

# Add this for timing measurements
import time
timing_stats = {}

# Global variables for managing video streaming sessions
active_video_streams = {}  # Track active video streams by camera_imou_id
paused_cameras = set()     # Track which cameras have paused AI analysis

# Helper function to measure execution time
def measure_time(step_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            print(f"⏱️ Starting {step_name}...")
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed = end_time - start_time
            timing_stats[step_name] = elapsed
            print(f"⏱️ {step_name} completed in {elapsed:.2f} seconds")
            return result
        return wrapper
    return decorator

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(
    filename='upload_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Replace with your Assistant ID
assistant_id = "asst_dC4rlfFAMJsA5c2ForclSxwi"

# Supabase configuration
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
supabase = create_client(supabase_url, supabase_key)

images_table = os.getenv('POSTGRES_TABLE_NAME', 'images')

# Load camera map from environment
camera_map = json.loads(os.getenv("CAMERA_MAP", "{}"))

def get_rtsp_url_by_camera_imou_id(camera_imou_id):
    """Get RTSP URL from camera_map using camera_imou_id"""
    try:
        if not camera_map:
            print("❌ CAMERA_MAP not found or empty in environment")
            return None
            
        # Find the room that matches the camera_imou_id
        for room_name, room_data in camera_map.items():
            if room_data.get('imou') == camera_imou_id:
                rtsp_url = room_data.get('rtsp')
                if rtsp_url:
                    print(f"✅ Found RTSP URL for camera {camera_imou_id} in room {room_name}: {rtsp_url}")
                    logging.info(f"Found RTSP URL for camera {camera_imou_id} in room {room_name}")
                    return rtsp_url
                else:
                    print(f"❌ No RTSP URL found for camera {camera_imou_id} in room {room_name}")
                    logging.error(f"No RTSP URL found for camera {camera_imou_id} in room {room_name}")
                    return None
        
        print(f"❌ Camera {camera_imou_id} not found in CAMERA_MAP")
        logging.error(f"Camera {camera_imou_id} not found in CAMERA_MAP")
        return None
        
    except Exception as e:
        print(f"❌ Error getting RTSP URL for camera {camera_imou_id}: {e}")
        logging.error(f"Error getting RTSP URL for camera {camera_imou_id}: {e}")
        return None

# Get patient info from database
def get_patient_info(patient_code):
    try:
        response = supabase.table('patients').select('patient_id', 'pathology').eq('code', patient_code).execute()
        if response.data and len(response.data) > 0:
            result = response.data[0]
            return {"patientId": result["patient_id"], "pathology": result["pathology"]}
        return None
    except Exception as e:
        logging.error(f"Error retrieving patient info: {e}")
        return None

# Save patient info to database
def save_patient_info(patient_id, pathology):
    try:
        response = supabase.table('patients').upsert({
            'patient_id': patient_id,
            'pathology': pathology
        }).execute()
        
        logging.info(f"Patient {patient_id} info saved to database")
    except Exception as e:
        logging.error(f"Error saving patient info: {e}")

def get_latest_image_for_camera(camera_imou_id, last_processed_id=None):
    """Get the most recent unprocessed image for a specific camera"""
    try:
        # Build query for specific camera
        query = supabase.table(images_table).select('id', 'url', 'aireaded', 'created_at', 'camera_imou_id').eq('camera_imou_id', camera_imou_id).order('created_at', desc=True).limit(1)
        
        response = query.execute()
        
        if response.data and len(response.data) > 0:
            result = response.data[0]
            if result.get("aireaded", True):
                return None
            # Skip if it's the same as the last processed ID
            if last_processed_id is not None and result["id"] == last_processed_id:
                return None
                
            print(f"📷 Camera {camera_imou_id} - Found image:")
            print(f"  id: {result['id']}")
            print(f"  url: {result['url']}")
            print(f"  created_at: {result['created_at']}")
            
            return {"id": result["id"], "url": result["url"], "createdAt": result["created_at"], "cameraImouId": result["camera_imou_id"]}
        return None
    except Exception as e:
        logging.error(f"Error retrieving latest image for camera {camera_imou_id}: {e}")
        return None

@measure_time("Get latest image")
# Get the most recent image from the database
def get_latest_image():
    try:
        # Get the most recent image regardless of aireaded status
        response = supabase.table(images_table).select('id', 'url', 'aireaded', 'created_at', 'camera_imou_id').order('created_at', desc=True).limit(1).execute()
        
        if response.data and len(response.data) > 0:
            result = response.data[0]
            print("id:", result["id"])
            print("url:", result["url"])
            print("aireaded:", result["aireaded"])
            print("created_at:", result["created_at"])
            print("camera_imou_id:", result["camera_imou_id"])
            
            # If the image has already been processed, return None
            if result.get("aireaded", True):
                print("Most recent image has already been processed, skipping.")
                return None
                
            # Otherwise return the image info
            return {"id": result["id"], "url": result["url"], "createdAt": result["created_at"], "cameraImouId": result["camera_imou_id"]}
        return None
    except Exception as e:
        logging.error(f"Error retrieving latest image: {e}")
        return None

@measure_time("Download image")
def download_image(url):
    try:
        response = requests.get(url)
        if response.status_code == 200 and len(response.content) > 0:
            byte_io = BytesIO(response.content)
            byte_io.name = "image.jpg"  # required for OpenAI file upload
            return byte_io
        else:
            logging.error(f"Failed to download image: HTTP {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Error downloading image: {e}")
        return None

def mark_image_as_processed(image_id):
    """Mark an image as processed in the database"""
    try:
        response = supabase.table(images_table).update({'aireaded': True}).eq('id', image_id).execute()
        logging.info(f"Image {image_id} marked as processed")
    except Exception as e:
        logging.error(f"Error marking image as processed: {e}")

def get_patient_id_by_code(patient_code):
    """Get patient ID from the database by patient code"""
    try:
        response = supabase.table('patients').select('patient_id').eq('code', patient_code).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]["patient_id"]
        return None
    except Exception as e:
        logging.error(f"Error retrieving patient ID by code: {e}")
        return None

def save_patient_record(patientCode, record_data, image_id, camera_imou_id=None):
    """Save patient record to database"""
    print(f"Saving patient record: {record_data}")
    try:
        # Generate a timestamp-based ID for the record
        record_id = int(time.time() * 1000)  # milliseconds since epoch
        
        # Get the patient_id from patient_code if needed
        patient_id = get_patient_id_by_code(patientCode)
        print(f"Patient ID from code {patientCode}: {patient_id}")
        if not patient_id:
            # If patient not found, create a new patient record with default pathology
            try:
                patient_id = int(time.time())  # Use timestamp as a numeric ID
                supabase.table('patients').insert({
                    'patient_id': patient_id,
                    'code': patientCode,
                    'pathology': "Unknown pathology"
                }).execute()
                
                print(f"Created new patient with ID: {patient_id}")
            except Exception as patient_error:
                logging.error(f"Error creating patient: {patient_error}")
                print(f"❌ Error creating patient: {patient_error}")
                # If we can't create a patient, use a default numeric ID
                patient_id = 999999
        
        # Ensure we have proper data types for database
        # Convert confidence to float and ensure it's in valid range (0-100)
        try:
            confidence = float(record_data.get('confidence', 0))
            if confidence < 0 or confidence > 100:
                confidence = 0
        except (ValueError, TypeError):
            confidence = 0
            
        # Ensure timestamp is a proper ISO format string
        if not isinstance(record_data.get('timestamp'), str):
            timestamp = datetime.now().isoformat()
        else:
            timestamp = record_data['timestamp']
            
        # Ensure status is one of the allowed values
        status = record_data.get('status', 'normal')
        if status not in ('normal', 'warning', 'critical', 'not in the room'):
            status = 'normal'
            
        # Truncate behavior and note fields if too long
        behavior = (record_data.get('behavior', '') or '')[:500]  # Limit to 500 chars
        behavior = re.sub(r'[^a-zA-Z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ\s]', '', behavior).strip()[:500]
        note = (record_data.get('note', '') or '')
        
        response = supabase.table('patient_records').insert({
            'id': record_id,
            'image_id': image_id,
            'patient_id': patient_id,
            'patient_code': patientCode,
            'timestamp': timestamp,
            'behavior': behavior,
            'confidence': confidence,
            'status': status,
            'note': note,
            'camera_imou_id': camera_imou_id
        }).execute()
        print(f"Saved patient record with ID: {record_id}")
        
        return {"id": record_id, "behavior": behavior, "status": status}
    except Exception as e:
        logging.error(f"Error saving patient record: {e}")
        print(f"❌ Error saving patient record: {e}")
        # Additional detailed error logging
        import traceback
        logging.error(f"Detailed error: {traceback.format_exc()}")
        return None

def parse_ai_response(text, image_id):
    """Parse the AI response into a structured object"""
    try:
        # Try to parse as JSON if it's already in that format
        try:
            data = json.loads(text)
            # Check if it has the required fields
            required_fields = ['timestamp', 'patient_id', 'behavior', 'confidence', 'status', 'note']
            if all(field in data for field in required_fields):
                return data
        except json.JSONDecodeError:
            # Not valid JSON, continue with text parsing
            pass
        
        # Fallback to text parsing logic - this is a simple implementation
        # You might need more sophisticated parsing based on your AI's actual output format
        lines = text.strip().split('\n')
        data = {
            'timestamp': datetime.now().isoformat(),
            'patient_code': 'P026',  # Default placeholder
            'behavior': '',
            'confidence': 0,
            'status': 'normal',
            'note': ''
        }
        
        for line in lines:
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                
                if 'timestamp' in key:
                    try:
                        # Try to parse the timestamp or use current time
                        data['timestamp'] = value or datetime.now().isoformat()
                    except Exception:
                        data['timestamp'] = datetime.now().isoformat()
                elif 'patient' in key:
                    data['patient_id'] = value or 'P026'
                elif 'behavior' in key:
                    # Clean the behavior value by removing quotes and extra whitespace
                    cleaned_value = re.sub(r'["\',]', '', value.strip()).strip()
                    data['behavior'] = cleaned_value
                elif 'confidence' in key:
                    try:
                        # Extract numeric value from confidence
                        confidence_str = ''.join(filter(lambda x: x.isdigit() or x == '.', value))
                        data['confidence'] = float(confidence_str) if confidence_str else 0
                    except Exception:
                        data['confidence'] = 0
                elif 'status' in key:
                    cleaned_value = re.sub(r'["\',]', '', value.strip()).strip()
                    data['status'] = cleaned_value
                elif 'note' in key:
                    data['note'] = value
        
        # If there's no structured data found, use the entire text as the note
        if not any([data['behavior'], data['note']]):
            data['note'] = text[:1000]  # Limit note length
            
        # Include the image ID
        data['image_id'] = image_id
        data['patient_code'] = data['patient_code']  # Use patient_id as code
            
        return data
        
    except Exception as e:
        logging.error(f"Error parsing AI response: {e}")
        # Return a minimal valid object
        return {
            'timestamp': datetime.now().isoformat(),
            'patient_id': 'P026',
            'patient_code': 'P026',
            'behavior': 'Unknown',
            'confidence': 0,
            'status': 'normal',
            'note': f"Error parsing response: {str(e)}",
            'image_id': image_id
        }

def run_vision_analysis(image_data, image_id, camera_imou_id=None):
    # Use a placeholder patient ID based on the image ID
    patient_code = "P026"
    
    try:
        # Check if image_data is valid
        if not image_data or image_data.getbuffer().nbytes == 0:
            logging.error(f"Invalid image data for ID {image_id}")
            print(f"❌ Invalid image data for ID {image_id}")
            return
        
        # Get patient info
        patient_info = get_patient_info(patient_code)
        
        # Reset file position to beginning
        image_data.seek(0)
        
        # Debug the content
        image_sample = image_data.read(20)
        print(f"📷 Camera {camera_imou_id} - Image data starts with: {image_sample.hex()}")
        image_data.seek(0)  # Reset position again
        
        # Always create a new thread for faster processing (no thread reuse)
        thread = openai.beta.threads.create()
        thread_id = thread.id
        logging.info(f"Created new thread with ID: {thread_id} for patient {patient_code} (Camera: {camera_imou_id})")
        
        # Upload image data to OpenAI with explicit filename
        try:
            # Ensure we're at the beginning of the file
            image_data.seek(0)
            
            # Check file size again
            file_size = image_data.getbuffer().nbytes
            print(f"📷 Camera {camera_imou_id} - Uploading file of size: {file_size} bytes")
            
            # Make sure the file has a proper name with extension
            if not hasattr(image_data, 'name') or not image_data.name:
                image_data.name = f"image_{image_id}.jpg"
            
            uploaded_file = openai.files.create(
                file=image_data,
                purpose="vision"
            )
            logging.info(f"Uploaded file with ID: {uploaded_file.id} (Camera: {camera_imou_id})")
        except Exception as upload_error:
            logging.error(f"Failed to upload image to OpenAI: {upload_error}")
            print(f"❌ Camera {camera_imou_id} - Failed to upload image to OpenAI: {upload_error}")
            
            # Try to save the image locally for debugging
            try:
                with open(f"debug_image_{image_id}.jpg", "wb") as debug_file:
                    image_data.seek(0)
                    debug_file.write(image_data.read())
                print(f"Saved problematic image to debug_image_{image_id}.jpg for inspection")
            except Exception as save_error:
                print(f"Could not save debug image: {save_error}")
                
            return

        # Prepare content for the message
        content = []
        
        # Always include text prompt for new threads
        if patient_info is None:
            prompt = f"Patient {patient_code} has pathology: Unknown pathology. Please analyze the patient's behavior based on the image below. Respond in JSON format: timestamp, patient_id, behavior: none|standing|lying|sitting|kneeling, status (normal/warning/critical/not in the room), note."
        else:
            prompt = f"Patient {patient_code} has pathology: {patient_info['pathology']}. Please analyze the patient's behavior based on the image below. Respond in JSON format: timestamp, patient_id, behavior: none|standing|lying|sitting|kneeling, status (normal/warning/critical/not in the room), note."

        
        content.append({"type": "text", "text": prompt})
        content.append({"type": "image_file", "image_file": {"file_id": uploaded_file.id}})

        # Create message
        openai.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content
        )
        logging.info(f"Created message in thread {thread_id} with image file ID: {uploaded_file.id} (Camera: {camera_imou_id})")

        # Run the assistant
        run = openai.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )
        logging.info(f"Started run with ID: {run.id} for thread {thread_id} (Camera: {camera_imou_id})")

        print(f"⏳ Camera {camera_imou_id} - Processing AI for image ID: {image_id}...")

        # Poll for completion
        poll_count = 0
        while True:
            poll_count += 1
            run_status = openai.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            if run_status.status == "completed":
                logging.info(f"Run {run.id} completed successfully (Camera: {camera_imou_id})")
                break
            elif run_status.status == "failed":
                logging.error(f"Run {run.id} failed (Camera: {camera_imou_id})")
                print(f"❌ Camera {camera_imou_id} - AI failed for image ID: {image_id}")
                return
            time.sleep(1)
        
        print(f"✅ Camera {camera_imou_id} - AI processing completed (polls: {poll_count})")

        # Retrieve response
        ai_response_text = None
        messages = openai.beta.threads.messages.list(thread_id=thread_id, limit=1)
        for message in messages.data:
            for content in message.content:
                if content.type == "text":
                    ai_response_text = content.text.value
                    print(f"💬 Camera {camera_imou_id} - AI response for image ID {image_id}:\n{ai_response_text}")
                    logging.info(f"Assistant response for image ID {image_id} (Camera: {camera_imou_id}): {ai_response_text}")
                    break
            if ai_response_text:
                break
                
        if ai_response_text:
            # Parse response
            structured_data = parse_ai_response(ai_response_text, image_id)
            print(f"🔄 Camera {camera_imou_id} - Parsed data: {json.dumps(structured_data, indent=2, ensure_ascii=False)}")
            
            # Save to database
            response = save_patient_record(patient_code, structured_data, image_id, camera_imou_id)
            
            if response:
                print(f"✅ Camera {camera_imou_id} - Saved record with ID: {response.get('id', '')}")
                if response.get('behavior', '').lower() == "lying" and response.get('status', '').lower() == "critical":
                    print(f"⚠️ Camera {camera_imou_id} - Patient detected as lying down. Starting video streaming...")
                    # Start video streaming using the refactored function - pass both image_id and camera_imou_id
                    run_video_streaming(image_id, camera_imou_id)
            else:
                print(f"❌ Camera {camera_imou_id} - Could not save record to database")
            
            # Mark as processed
            mark_image_as_processed(image_id)
            print(f"✅ Camera {camera_imou_id} - Marked image {image_id} as processed")
            
        else:
            print(f"❌ Camera {camera_imou_id} - No AI response received")

    except Exception as e:
        logging.error(f"An error occurred for image ID {image_id} (Camera: {camera_imou_id}): {e}")
        print(f"❌ Camera {camera_imou_id} - Error with image ID {image_id}: {e}")
        import traceback
        print(traceback.format_exc())

def get_rtsp_url_for_camera(camera_imou_id):
    """Get RTSP URL for a specific camera from CAMERA_MAP"""
    try:
        if not camera_map:
            print("❌ CAMERA_MAP not found or empty in environment")
            return None
            
        # Find the room that matches the camera_imou_id
        for room_name, room_data in camera_map.items():
            if room_data.get('imou') == camera_imou_id:
                rtsp_url = room_data.get('rtsp')
                if rtsp_url:
                    print(f"✅ Found RTSP URL for camera {camera_imou_id} in room {room_name}: {rtsp_url}")
                    return rtsp_url
                else:
                    print(f"❌ No RTSP URL found for camera {camera_imou_id} in room {room_name}")
                    return None
        
        print(f"❌ Camera {camera_imou_id} not found in CAMERA_MAP")
        return None
        
    except Exception as e:
        print(f"❌ Error getting RTSP URL for camera {camera_imou_id}: {e}")
        return None

def process_latest_image_for_camera(camera_imou_id, last_processed_id):
    """
    Process the latest unprocessed image for a specific camera.
    
    Args:
        camera_imou_id: The camera ID to process images for
        last_processed_id: ID of the last processed image to avoid reprocessing
        
    Returns:
        The ID of the processed image if successful, None otherwise
    """
    global paused_cameras
    
    # Check if AI analysis is paused for this camera (due to active video streaming)
    if camera_imou_id in paused_cameras:
        print(f"⏸️ Camera {camera_imou_id} - AI analysis paused (video streaming active)")
        return None
    
    image_info = get_latest_image_for_camera(camera_imou_id, last_processed_id)
    temp_file_path = None
    
    # If no image found or it's the same as the last processed one, return None
    if not image_info:
        return None
    
    print(f"📷 Camera {camera_imou_id} - Processing image ID: {image_info['id']}")
    image_data = download_image(image_info['url'])
    
    if image_data:
        try:
            # Store the path to clean up later
            if hasattr(image_data, 'name') and os.path.exists(image_data.name):
                temp_file_path = image_data.name
                
            run_vision_analysis(image_data, image_info['id'], camera_imou_id)
            return image_info['id']  # Return the processed image ID
        finally:
            # Clean up
            if image_data and hasattr(image_data, 'close'):
                image_data.close()
                
            # Delete temporary file if it exists
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    print(f"📷 Camera {camera_imou_id} - Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    print(f"📷 Camera {camera_imou_id} - Error cleaning up temporary file: {e}")
    else:
        print(f"❌ Camera {camera_imou_id} - Failed to download image data")
        # Mark the image as processed even though we couldn't download it
        mark_image_as_processed(image_info['id'])
        print(f"✅ Camera {camera_imou_id} - Marked failed image {image_info['id']} as processed to prevent retry")
        return image_info['id']  # Return the ID so we don't retry it

def camera_processing_thread(camera_imou_id, room_name):
    """
    Background thread function to continuously process images for a specific camera
    """
    print(f"🎬 Started processing thread for camera {camera_imou_id} in room '{room_name}'")
    last_processed_id = None
    
    while True:
        try:
            # Process the latest image for this camera
            processed_id = process_latest_image_for_camera(camera_imou_id, last_processed_id)
            if processed_id:
                last_processed_id = processed_id
                print(f"📷 Camera {camera_imou_id} - Processed image {processed_id}")
            
            # Wait before checking for new images
            time.sleep(5)  # Check every 5 seconds
            
        except Exception as e:
            print(f"❌ Camera {camera_imou_id} - Error in processing thread: {e}")
            logging.error(f"Error in camera processing thread for {camera_imou_id}: {e}")
            time.sleep(10)  # Wait longer on error before retrying

def run_video_streaming(image_id, camera_imou_id):
    """
    Refactored video streaming function that:
    1. Uses existing camera_imou_id (no redundant Supabase fetch)
    2. Gets RTSP URL from camera_map
    3. Prevents multiple sessions for same camera
    4. Launches streaming in non-blocking thread
    5. Pauses AI analysis for that camera while streaming
    """
    global active_video_streams, paused_cameras
    
    try:
        print(f"🎬 BEFORE starting video streaming worker - image_id: {image_id}, camera_imou_id: {camera_imou_id}")
        logging.info(f"BEFORE starting video streaming worker - image_id: {image_id}, camera_imou_id: {camera_imou_id}")
        
        # Validate inputs
        if not camera_imou_id:
            print(f"❌ No camera_imou_id provided for video streaming")
            logging.error(f"No camera_imou_id provided for video streaming")
            return
        
        # Get RTSP URL from camera_map
        rtsp_url = get_rtsp_url_by_camera_imou_id(camera_imou_id)
        if not rtsp_url:
            print(f"❌ Skipping video streaming - no RTSP URL found for camera: {camera_imou_id}")
            logging.error(f"Skipping video streaming - no RTSP URL found for camera: {camera_imou_id}")
            return
        
        # Check if video streaming is already running for this camera
        if camera_imou_id in active_video_streams:
            print(f"⚠️ Video streaming already active for camera {camera_imou_id}, skipping new session")
            logging.warning(f"Video streaming already active for camera {camera_imou_id}, skipping new session")
            return
        
        print(f"🎥 Starting video streaming for camera {camera_imou_id} with image_id {image_id}")
        logging.info(f"Starting video streaming for camera {camera_imou_id} with image_id {image_id}")
        
        # Pause AI analysis for this camera
        paused_cameras.add(camera_imou_id)
        print(f"⏸️ Paused AI analysis for camera {camera_imou_id}")
        logging.info(f"Paused AI analysis for camera {camera_imou_id}")
        
        # Create and start the video streaming thread
        streaming_thread = threading.Thread(
            target=video_streaming_worker,
            args=(camera_imou_id, rtsp_url, image_id),
            daemon=True,
            name=f"VideoStream-{camera_imou_id}"
        )
        
        # Mark this camera as having an active stream
        active_video_streams[camera_imou_id] = {
            'thread': streaming_thread,
            'image_id': image_id,
            'rtsp_url': rtsp_url,
            'start_time': datetime.now()
        }
        
        print(f"🚀 ABOUT TO START video streaming thread for camera {camera_imou_id}")
        logging.info(f"ABOUT TO START video streaming thread for camera {camera_imou_id}")
        
        streaming_thread.start()
        
        print(f"✅ Video streaming thread STARTED for camera {camera_imou_id}")
        logging.info(f"Video streaming thread STARTED for camera {camera_imou_id}")
        
    except Exception as e:
        print(f"❌ Error in run_video_streaming for image_id {image_id}: {e}")
        logging.error(f"Error in run_video_streaming for image_id {image_id}: {e}")
        # Clean up on error
        if camera_imou_id:
            paused_cameras.discard(camera_imou_id)
            active_video_streams.pop(camera_imou_id, None)

def video_streaming_worker(camera_imou_id, rtsp_url, image_id):
    """
    Worker function that runs the actual video streaming process
    """
    global active_video_streams, paused_cameras
    
    try:
        print(f"🎬 Video streaming worker STARTED for camera {camera_imou_id}")
        logging.info(f"Video streaming worker STARTED for camera {camera_imou_id}")
        
        # Check for required modules
        try:
            import importlib
            importlib.import_module('mediapipe')
            importlib.import_module('cv2')
            print(f"✅ Required modules (mediapipe, cv2) are available")
        except ImportError as e:
            print(f"\n❌ Missing required module: {e}")
            print("Please install the required dependencies:")
            print("pip install mediapipe opencv-python")
            return
        
        # Get current directory and video_streaming.py path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        video_streaming_path = os.path.join(current_dir, 'video_streaming.py')
        
        print(f"📁 Current directory: {current_dir}")
        print(f"📄 Video streaming path: {video_streaming_path}")
        print(f"📋 File exists: {os.path.exists(video_streaming_path)}")
        print(f"🐍 Python executable: {sys.executable}")
        
        if not os.path.exists(video_streaming_path):
            print(f"❌ video_streaming.py not found at: {video_streaming_path}")
            logging.error(f"video_streaming.py not found at: {video_streaming_path}")
            return
        
        # Prepare command with proper arguments using sys.executable
        command = [
            sys.executable,  # Use same Python interpreter as current script
            video_streaming_path,
            rtsp_url,
            '--image_id', str(image_id)
        ]
        
        print(f"🚀 EXECUTING command: {' '.join(command)}")
        logging.info(f"EXECUTING command: {' '.join(command)}")
        
        # Start the video streaming process with proper encoding handling
        print(f"📋 About to call subprocess.Popen...")
        
        # Set environment variables to handle Unicode properly
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        process = subprocess.Popen(
            command,
            cwd=current_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',  # Explicitly set UTF-8 encoding
            errors='replace',  # Replace invalid characters instead of failing
            bufsize=1,
            universal_newlines=True,
            env=env  # Pass environment with UTF-8 encoding
        )
        
        print(f"✅ subprocess.Popen called successfully, PID: {process.pid}")
        logging.info(f"subprocess.Popen called successfully, PID: {process.pid}")
        
        # Monitor the process and wait for it to complete
        try:
            print(f"📹 Monitoring video streaming process for camera {camera_imou_id}...")
            # Read output in real-time with safe Unicode handling
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Strip Unicode characters that might cause encoding issues
                    safe_output = output.strip()
                    # Remove or replace problematic Unicode characters
                    safe_output = safe_output.encode('ascii', 'ignore').decode('ascii')
                    if safe_output:  # Only print if there's content after cleaning
                        print(f"📹 Camera {camera_imou_id}: {safe_output}")
            
            # Wait for process to complete
            return_code = process.wait()
            print(f"🎬 Video streaming ended for camera {camera_imou_id} with return code: {return_code}")
            logging.info(f"Video streaming ended for camera {camera_imou_id} with return code: {return_code}")
            
        except Exception as e:
            print(f"❌ Error monitoring video streaming process for camera {camera_imou_id}: {e}")
            logging.error(f"Error monitoring video streaming process for camera {camera_imou_id}: {e}")
            process.terminate()
            
    except Exception as e:
        print(f"❌ Error in video streaming worker for camera {camera_imou_id}: {e}")
        logging.error(f"Error in video streaming worker for camera {camera_imou_id}: {e}")
        import traceback
        print(f"📋 Full traceback: {traceback.format_exc()}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        
    finally:
        # Clean up: Resume AI analysis and remove from active streams
        paused_cameras.discard(camera_imou_id)
        active_video_streams.pop(camera_imou_id, None)
        print(f"▶️ Resumed AI analysis for camera {camera_imou_id}")
        logging.info(f"Video streaming ended and AI analysis resumed for camera {camera_imou_id}")


def process_latest_image(last_processed_id=None):
    """
    Process the latest unprocessed image from the database.
    This function is kept for backward compatibility but won't be used in parallel mode.
    
    Args:
        last_processed_id: Optional ID of the last processed image to skip processing the same image again
        
    Returns:
        The ID of the processed image if successful, None otherwise
    """
    image_info = get_latest_image()
    temp_file_path = None
    print("Processing latest image...", last_processed_id)
    # If no image found or it's the same as the last processed one, return None
    if not image_info or (last_processed_id is not None and image_info['id'] == last_processed_id):
        if not image_info:
            print("No unprocessed images found in the database")
        return None
    
    print(f"Processing image ID: {image_info['id']}")
    image_data = download_image(image_info['url'])
    
    if image_data:
        try:
            # Store the path to clean up later
            if hasattr(image_data, 'name') and os.path.exists(image_data.name):
                temp_file_path = image_data.name
                
            run_vision_analysis(image_data, image_info['id'], image_info.get('cameraImouId'))
            return image_info['id']  # Return the processed image ID
        finally:
            # Clean up
            if image_data and hasattr(image_data, 'close'):
                image_data.close()
                
            # Delete temporary file if it exists
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    print(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    print(f"Error cleaning up temporary file: {e}")
    else:
        print("❌ Failed to download image data")
        # Mark the image as processed even though we couldn't download it
        mark_image_as_processed(image_info['id'])
        print(f"✅ Marked failed image {image_info['id']} as processed to prevent retry")
        return image_info['id']  # Return the ID so we don't retry it

if __name__ == "__main__":
    print("🚀 Starting Parallel AI Analysis System...")
    print("="*60)
    
    # Check if camera_map is available and has entries
    if camera_map:
        print(f"📷 Found {len(camera_map)} cameras in CAMERA_MAP:")
        
        # Display camera configuration
        for room_name, room_data in camera_map.items():
            camera_imou_id = room_data.get('imou')
            rtsp_url = room_data.get('rtsp', 'No RTSP URL')
            print(f"  🏠 Room: {room_name}")
            print(f"     📹 Camera ID: {camera_imou_id}")
            print(f"     🔗 RTSP: {rtsp_url}")
            print()
        
        # Start a processing thread for each camera
        threads = []
        
        for room_name, room_data in camera_map.items():
            camera_imou_id = room_data.get('imou')
            
            if camera_imou_id:
                # Create and start thread for this camera
                thread = threading.Thread(
                    target=camera_processing_thread,
                    args=(camera_imou_id, room_name),
                    daemon=True,
                    name=f"Camera-{camera_imou_id}"
                )
                thread.start()
                threads.append(thread)
                print(f"✅ Started processing thread for camera {camera_imou_id} in room '{room_name}'")
            else:
                print(f"⚠️ No camera ID found for room '{room_name}', skipping...")
        
        print(f"\n🎬 {len(threads)} camera processing threads started!")
        print("📢 Press Ctrl+C to stop all processing...")
        print("="*60)
        
        try:
            # Keep the main process alive so threads continue running
            while True:
                # Check if all threads are still alive
                alive_threads = [t for t in threads if t.is_alive()]
                if len(alive_threads) != len(threads):
                    print(f"⚠️ Warning: Only {len(alive_threads)}/{len(threads)} camera threads are running")
                
                time.sleep(30)  # Check thread status every 30 seconds
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down parallel AI analysis system...")
            print("Waiting for threads to complete current operations...")
            
            # Wait for threads to finish (they are daemon threads, so they'll stop when main process exits)
            time.sleep(2)
            print("✅ All camera processing threads stopped")
