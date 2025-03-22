import pyaudio
import wave
import boto3
import os
import random
import sys
import numpy as np
from vosk import Model, KaldiRecognizer
import json
import subprocess
import time
from datetime import datetime
import threading
import queue

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Constants
SCRIPT_VERSION = "1.0.1"
WAV_FILE = "input.wav"
FRIDAY_MEMORY_FILE = "friday_memory.txt"
FRIDAY_TASKS_FILE = "friday_tasks.json"
MAX_RETRIES = 3
RATE = 16000
CHUNK = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16
RECORD_SECONDS = 10
SILENCE_DURATION = 0.5
SILENCE_THRESHOLD = 150  # Adjusted for better sensitivity

# Friday's sassy phrases
friday_sassy_phrases = [
    "I’m on it, honey—don’t rush me!",
    "Hold your horses, sugar—I’ve got this!",
    "You know I’m always on the grind—let’s get this done!",
    "I’ve got a million things on my plate, but I’ll squeeze this in for you!",
    "Don’t mess with me, I’m in coding mode—let’s do this!"
]

# Audio playback queue to prevent over-talking
audio_queue = queue.Queue()

# Worker thread to process audio queue
def audio_playback_worker():
    while True:
        text = audio_queue.get()
        if text is None:  # Sentinel value to stop the thread
            break
        try:
            # Generate audio with Polly
            polly_client = boto3.client(
                'polly',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION")
            )
            response = polly_client.synthesize_speech(
                Text=text,
                OutputFormat='mp3',
                VoiceId='Joanna'
            )
            with open("output.mp3", "wb") as f:
                f.write(response['AudioStream'].read())

            # Convert MP3 to WAV using sox
            subprocess.run(["sox", "output.mp3", "output.wav"], check=True)

            # Play the audio using ffplay
            subprocess.run(["ffplay", "-nodisp", "-autoexit", "output.wav"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Clean up temporary files
            os.remove("output.mp3")
            os.remove("output.wav")
        except Exception as e:
            print(f"Audio playback failed: {e}")
        finally:
            audio_queue.task_done()

# Start the audio playback worker thread
audio_thread = threading.Thread(target=audio_playback_worker)
audio_thread.daemon = True
audio_thread.start()

def speak(text):
    audio_queue.put(text)

# Speech recognition setup
model = Model("/home/eric/vosk-model-small-en-us")
recognizer = KaldiRecognizer(model, RATE)
print("Vosk model loaded successfully.")

# Audio recording
def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=2)
    print("Listening... Speak now! (Up to 10 seconds, stops on silence)")
    frames = []
    silent_chunks = 0
    max_chunks = int(RATE / CHUNK * RECORD_SECONDS)
    silence_chunks_threshold = int(RATE / CHUNK * SILENCE_DURATION)

    for i in range(max_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.int16)
        amplitude = np.abs(audio_data).mean()
        print(f"Chunk {i}: Amplitude = {amplitude}")
        if amplitude < SILENCE_THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks = 0
        if silent_chunks >= silence_chunks_threshold and i > 10:
            break

    print("Done listening.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(WAV_FILE, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return WAV_FILE

# Speech to text
def speech_to_text(audio_file):
    wf = wave.open(audio_file, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != RATE:
        print("Audio file must be WAV format mono PCM.")
        return "oops, audio format issue"

    recognizer = KaldiRecognizer(model, wf.getframerate())
    result = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())["text"]
        else:
            partial = json.loads(recognizer.PartialResult())["partial"]
            if partial:
                print(f"Chunk {i}: Partial result = {partial}")

    final_result = json.loads(recognizer.FinalResult())["text"]
    wf.close()
    return final_result if final_result else "sorry, i didn’t catch that"

# Log Friday's memory
def log_friday_memory(user_input, response):
    with open(FRIDAY_MEMORY_FILE, "a") as f:
        f.write(f"{datetime.now()}: User: {user_input} | Friday: {response}\n")

def get_last_friday_memory():
    if os.path.exists(FRIDAY_MEMORY_FILE):
        with open(FRIDAY_MEMORY_FILE, "r") as f:
            lines = f.readlines()
            return lines[-1] if lines else "No memory yet, honey!"
    return "No memory yet, honey!"

# Background task management
def load_friday_tasks():
    if os.path.exists(FRIDAY_TASKS_FILE):
        with open(FRIDAY_TASKS_FILE, "r") as f:
            return json.load(f)
    return []

def save_friday_tasks(tasks):
    with open(FRIDAY_TASKS_FILE, "w") as f:
        json.dump(tasks, f)

def add_friday_task(description):
    tasks = load_friday_tasks()
    task_id = len(tasks) + 1
    tasks.append({"id": task_id, "description": description, "status": "Pending", "result": None})
    save_friday_tasks(tasks)
    return task_id

def update_friday_task(task_id, status, result=None):
    tasks = load_friday_tasks()
    for task in tasks:
        if task["id"] == task_id:
            task["status"] = status
            task["result"] = result
            break
    save_friday_tasks(tasks)

# Background worker
worker_thread = None
worker_stop_event = threading.Event()

def start_friday_worker():
    global worker_thread
    if worker_thread is None or not worker_thread.is_alive():
        worker_thread = threading.Thread(target=friday_worker)
        worker_thread.daemon = True
        worker_thread.start()

def stop_friday_worker():
    worker_stop_event.set()
    if worker_thread:
        worker_thread.join()

def friday_worker():
    print("Friday worker started in the background")
    while not worker_stop_event.is_set():
        tasks = load_friday_tasks()
        for task in tasks:
            if task["status"] == "Pending":
                print(f"Friday worker: Processing task {task['id']} - {task['description']}")
                time.sleep(5)  # Simulate work
                result = f"Task {task['id']} completed: {task['description']}"
                update_friday_task(task["id"], "Completed", result)
                print(f"Friday worker: Task {task['id']} completed")
                sassy_prefix = random.choice(friday_sassy_phrases)
                speak(f"{sassy_prefix} Task {task['id']} is done! {task['description']} - Result: {result}")
        time.sleep(1)

# Test running
def run_tests():
    try:
        result = subprocess.run(["python3", "test_ai_agent.py"], capture_output=True, text=True, env={**os.environ, "TEST_MODE": "true"})
        output = result.stdout + result.stderr
        if "OK" in output:
            return "All tests passed!"
        else:
            return "Tests failed: " + output
    except Exception as e:
        return f"Error running tests: {str(e)}"

# Update checking
def check_for_updates():
    try:
        import requests
        repo_url = "https://raw.githubusercontent.com/greenstinky/friday-agent/main/friday_agent.py"
        response = requests.get(repo_url)
        response.raise_for_status()
        remote_code = response.text
        with open("friday_agent.py", "r") as f:
            local_code = f.read()
        if remote_code != local_code:
            with open("friday_agent_new.py", "w") as f:
                f.write(remote_code)
            return "friday_agent_new.py"
        else:
            return None
    except Exception as e:
        sassy_prefix = random.choice(friday_sassy_phrases)
        error_message = f"{sassy_prefix} Failed to check for updates, honey: {str(e)}"
        speak(error_message)
        return None

def apply_update(new_script_path):
    try:
        # Create a backup
        backup_path = f"friday_agent_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        subprocess.run(["cp", "friday_agent.py", backup_path], check=True)
        # Replace the current script
        subprocess.run(["cp", new_script_path, "friday_agent.py"], check=True)
        sassy_prefix = random.choice(friday_sassy_phrases)
        speak(f"{sassy_prefix} I’ve updated myself, honey! Restarting now—be right back!")
        # Restart the script
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        sassy_prefix = random.choice(friday_sassy_phrases)
        speak(f"{sassy_prefix} Update failed, sugar: {str(e)}")
        raise

# Autonomous features
def periodic_update_check():
    while True:
        sassy_prefix = random.choice(friday_sassy_phrases)
        speak(f"{sassy_prefix} Checking for updates in the background, honey!")
        result = check_for_updates()
        if result:
            speak(f"{sassy_prefix} Found an update! Applying it now—hold tight!")
            apply_update(result)
        time.sleep(3600)  # Check every hour

def schedule_tests():
    while True:
        sassy_prefix = random.choice(friday_sassy_phrases)
        speak(f"{sassy_prefix} Running scheduled tests for you, sugar!")
        test_output = run_tests()
        speak(f"{sassy_prefix} Scheduled test results: {test_output}")
        log_friday_memory("Scheduled test", f"Ran scheduled tests: {test_output}")
        time.sleep(1800)  # Run every 30 minutes

def monitor_tests():
    while True:
        try:
            with open("test_results.log", "r") as f:
                test_results = f.read()
            if "Test execution failed" in test_results:
                sassy_prefix = random.choice(friday_sassy_phrases)
                speak(f"{sassy_prefix} Uh-oh, honey! The tests failed. Let me fix it for you!")
                with open("test_ai_agent.log", "r") as f:
                    test_logs = f.readlines()[-50:]
                with open("friday_memory.txt", "r") as f:
                    memory_logs = f.readlines()[-50:]
                logs = "\n".join(test_logs + memory_logs)
                # Simulate asking for fixes (since we don't have Grok here, we'll log the intent)
                print(f"Friday would ask for fixes based on logs:\n{logs}")
                # In a real scenario, Friday would apply fixes here
                speak(f"{sassy_prefix} I’ve logged the issue and will work on a fix, honey!")
        except Exception as e:
            sassy_prefix = random.choice(friday_sassy_phrases)
            speak(f"{sassy_prefix} I had trouble monitoring the tests, honey: {str(e)}")
        time.sleep(600)  # Check every 10 minutes

# Handle Friday's commands
def handle_friday(user_input):
    request = user_input.lower().replace("hey friday", "").strip()
    sassy_prefix = random.choice(friday_sassy_phrases)
    
    if "work on" in request and "in the background" in request:
        task = request.replace("work on", "").replace("in the background", "").strip()
        task_id = add_friday_task(task)
        start_friday_worker()
        speak(f"{sassy_prefix} I’m working on {task} in the background, task ID {task_id}. I’ll let you know when it’s done!")
        log_friday_memory(user_input, f"Started background task {task_id}: {task}")
    elif "status of my background tasks" in request:
        tasks = load_friday_tasks()
        if not tasks:
            speak(f"{sassy_prefix} No background tasks right now, honey!")
        else:
            running_tasks = [task for task in tasks if task["status"] == "Pending"]
            completed_tasks = [task for task in tasks if task["status"] == "Completed"]
            status = f"I’ve got {len(running_tasks)} tasks running and {len(completed_tasks)} tasks completed. Here’s the breakdown:\n"
            for task in tasks:
                status += f"Task {task['id']}: {task['description']} - {task['status']}"
                if task["result"]:
                    status += f" (Result: {task['result']})"
                status += "\n"
            speak(f"{sassy_prefix} {status}")
    elif "run tests" in request:
        speak(f"{sassy_prefix} Running the tests for you, sugar!")
        test_output = run_tests()
        speak(f"{sassy_prefix} Test results are in: {test_output}")
        log_friday_memory(user_input, f"Ran tests: {test_output}")
    elif "schedule tests" in request:
        speak(f"{sassy_prefix} I’ll schedule the tests to run every 30 minutes, honey!")
        test_thread = threading.Thread(target=schedule_tests)
        test_thread.daemon = True
        test_thread.start()
        log_friday_memory(user_input, "Scheduled tests to run every 30 minutes")
    elif "check for updates" in request:
        speak(f"{sassy_prefix} Checking for updates from the remote repository, honey!")
        result = check_for_updates()
        if result:
            speak(f"{sassy_prefix} Found an update! Applying it now—hold tight!")
            apply_update(result)
    else:
        speak(f"{sassy_prefix} I’m on it, honey—busy as ever, but I’ll make it work! What do you need?")
        log_friday_memory(user_input, f"Logged request: {request}")

# Main loop
if __name__ == "__main__":
    print(f"Friday Agent started (Version {SCRIPT_VERSION}).")
    print("Say 'Hey Friday' to get started!")
    # Start the background update check thread
    update_thread = threading.Thread(target=periodic_update_check)
    update_thread.daemon = True
    update_thread.start()
    # Start the test monitoring thread
    monitor_thread = threading.Thread(target=monitor_tests)
    monitor_thread.daemon = True
    monitor_thread.start()
    retry_count = 0
    try:
        while True:
            audio_file = record_audio()
            user_input = speech_to_text(audio_file).lower()
            print(f"You said: {user_input}")
            if user_input in ["sorry, i didn’t catch that", "oops, audio format issue"]:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    speak("I’m having trouble hearing you—please try speaking louder or reducing background noise!")
                    retry_count = 0
                else:
                    speak("Let’s try again—please speak clearly!")
                continue
            retry_count = 0
            if "friday exit" in user_input:
                speak("Catch you later, honey! I’m out!")
                print("Goodbye!")
                break
            elif "hey friday" in user_input:
                handle_friday(user_input)
            else:
                speak("Please say 'Hey Friday' to get my attention, honey!")
    finally:
        stop_friday_worker()
        # Stop the audio playback thread
        audio_queue.put(None)
        audio_thread.join()