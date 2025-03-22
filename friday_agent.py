import pyaudio
import wave
import boto3
import os
import random
import sys
import threading
import numpy as np
from vosk import Model, KaldiRecognizer
import json
import subprocess
import time
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# API keys from .env
OPENAI_KEY = os.getenv("OPENAI_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

# OpenAI setup
client = OpenAI(api_key=OPENAI_KEY)

# AWS setup for Polly
polly_client = boto3.client(
    'polly',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 10
SILENCE_THRESHOLD = 300
SILENCE_DURATION = 1.0
WAV_FILE = "input.wav"
MAX_RETRIES = 3
SCRIPT_VERSION = "1.0.1"

# Shared folder & files
SHARED_FOLDER = "/home/eric/learning_buddy"
if not os.path.exists(SHARED_FOLDER):
    os.makedirs(SHARED_FOLDER)
FRIDAY_TASKS_FILE = os.path.join(SHARED_FOLDER, "friday_tasks.json")
FRIDAY_MEMORY_FILE = os.path.join(SHARED_FOLDER, "friday_memory.txt")

# Initialize Friday's task database
if not os.path.exists(FRIDAY_TASKS_FILE):
    with open(FRIDAY_TASKS_FILE, "w") as f:
        json.dump([], f)

# Initialize Friday's memory file
if not os.path.exists(FRIDAY_MEMORY_FILE):
    with open(FRIDAY_MEMORY_FILE, "w") as f:
        f.write("")

# Vosk model setup
vosk_model = Model("/home/eric/vosk-model-small-en-us")
vosk_recognizer = KaldiRecognizer(vosk_model, RATE)
vosk_recognizer.SetWords(True)
print("Vosk model loaded successfully.")

# Friday's sassy phrases (global scope)
friday_sassy_phrases = [
    "Honey, I’m busier than a bee, but I’ll make it work!",
    "You know I’m always on the grind—let’s get this done!",
    "I’ve got a million things on my plate, but I’ll squeeze this in for you!",
    "Don’t mess with me, I’m in coding mode—let’s do this!"
]

# Friday's background worker state
friday_worker_running = False
friday_worker_process = None

def start_friday_worker():
    global friday_worker_running, friday_worker_process
    if not friday_worker_running:
        friday_worker_process = subprocess.Popen(["python3", "friday_worker.py"])
        friday_worker_running = True
        print("Friday worker started in the background")

def stop_friday_worker():
    global friday_worker_running, friday_worker_process
    if friday_worker_running and friday_worker_process:
        friday_worker_process.terminate()
        friday_worker_process = None
        friday_worker_running = False
        print("Friday worker stopped")

def add_friday_task(description):
    tasks = load_friday_tasks()
    task_id = len(tasks) + 1
    task = {
        "id": task_id,
        "description": description,
        "status": "Pending",
        "result": None
    }
    tasks.append(task)
    save_friday_tasks(tasks)
    return task_id

def load_friday_tasks():
    if os.path.exists(FRIDAY_TASKS_FILE):
        with open(FRIDAY_TASKS_FILE, "r") as f:
            return json.load(f)
    return []

def save_friday_tasks(tasks):
    with open(FRIDAY_TASKS_FILE, "w") as f:
        json.dump(tasks, f, indent=4)

def log_friday_memory(user_input, response):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(FRIDAY_MEMORY_FILE, "a") as f:
            f.write(f"[{timestamp}] You: {user_input} | FRIDAY: {response}\n")
    except Exception as e:
        print(f"Failed to log Friday memory: {e}")

def get_last_friday_memory():
    if os.path.exists(FRIDAY_MEMORY_FILE):
        with open(FRIDAY_MEMORY_FILE, "r") as f:
            lines = f.readlines()
            if lines:
                return lines[-1].strip()
    return "I don’t have any recent coding requests in my memory, honey!"

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=2)
    print("Listening... Speak now! (Up to 10 seconds, stops on silence)")
    frames = []
    silent_chunks = 0
    max_chunks = int(RATE / CHUNK * RECORD_SECONDS)
    silence_chunks_threshold = int(RATE / CHUNK * SILENCE_DURATION)
    # Lower the silence threshold to be more sensitive
    adjusted_silence_threshold = 150  # Reduced from 300

    for i in range(max_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.int16)
        amplitude = np.abs(audio_data).mean()
        print(f"Chunk {i}: Amplitude = {amplitude}")
        if amplitude < adjusted_silence_threshold:
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

def speech_to_text(audio_file):
    wf = wave.open(audio_file, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != RATE:
        print("Audio file must be WAV format mono PCM.")
        return "Oops, audio format issue."
    
    text = ""
    chunk_count = 0
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        chunk_count += 1
        if vosk_recognizer.AcceptWaveform(data):
            result = json.loads(vosk_recognizer.Result())
            partial_text = result.get("text", "")
            print(f"Chunk {chunk_count}: Partial result = {partial_text}")
            text += partial_text + " "
    final_result = json.loads(vosk_recognizer.FinalResult())
    final_text = final_result.get("text", "")
    print(f"Final result = {final_text}")
    text += final_text
    wf.close()
    
    if not text.strip():
        return "Sorry, I didn’t catch that."
    
    text = text.strip()
    if text.startswith("a "):
        text = "hey " + text[2:]
    if text.startswith("friday "):
        text = "hey " + text
    if text.startswith("good "):
        text = "hey " + text[5:]
    if text.startswith("he "):
        text = "hey " + text[3:]
    text = text.replace("asked", "ask")
    text = text.replace("basque", "background")
    text = text.replace("where you write", "write")
    text = text.replace("little sorting", "sort")
    text = text.replace("it's good to see you rewrite", "")
    text = text.replace("death add parentheses a comma be", "def add(a, b)")
    text = text.replace("was the last thing i ask you about coding", "what did i last ask about coding")
    text = text.replace("one of the status of any background tasks that i have", "what’s the status of my background tasks")
    text = text.replace("how close are my task to being done", "how close are my tasks to being done")
    text = text.replace("run something and went", "run tests")
    text = text.replace("and make a know that the task completed but i didn't hear a sound that it was them", "i didn't hear a notification for the task completion")
    text = text.replace("make a know", "i didn't hear a notification")
    text = text.replace("how many background tasks are running", "what’s the status of my background tasks")
    text = text.replace("can you list the background tasks that are running", "what’s the status of my background tasks")
    text = text.replace("writer you don't actually have a virtual test environment did you just lie to me", "friday you don't actually have a virtual test environment did you just lie to me")
    text = text.replace("there is a mistake i didn't need you to scheduled tasks i needed you to schedule tests", "there is a mistake i didn't need you to schedule tasks i needed you to schedule tests")
    text = text.replace("schedule some tasks", "schedule tests")
    if text == "exit":
        text = "friday exit"
    
    return text

def ask_grok(question):
    system_prompt = "You are Friday, a busy and sassy AI with a strong black woman vibe. Be efficient, direct, and a bit playful. Focus on upcoming projects, research coding implementations, and run simulations in your virtual test environment."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content

def speak(text):
    print(f"DEBUG: speak() called with text: {text}, agent: friday")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            ssml_text = f"""
            <speak>
                <prosody pitch="medium" rate="1.2" volume="loud">
                    {text}
                </prosody>
            </speak>
            """
            response = polly_client.synthesize_speech(
                TextType="ssml",
                Text=ssml_text,
                OutputFormat="mp3",
                VoiceId="Salli",
                Engine="standard"
            )
            with open("output.mp3", "wb") as f:
                f.write(response["AudioStream"].read())
        except Exception as e:
            print(f"Polly error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print("Failed to synthesize speech after all retries.")
                return
            time.sleep(1)
            continue

        # Convert MP3 to WAV
        try:
            result = subprocess.run(
                ["sox", "output.mp3", "-r", "16000", "-c", "2", "output.wav"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"sox output: {result.stdout}")
            if result.stderr:
                print(f"sox stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"sox conversion failed (attempt {attempt + 1}/{max_retries}): {e}")
            if os.path.exists("output.mp3"):
                os.remove("output.mp3")
            if attempt == max_retries - 1:
                print("Failed to convert audio after all retries.")
                return
            time.sleep(1)
            continue

        # Play the audio
        try:
            result = subprocess.run(
                ["ffplay", "-ar", "16000", "-autoexit", "-nodisp", "output.wav"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"ffplay output: {result.stdout}")
            if result.stderr:
                print(f"ffplay stderr: {result.stderr}")
            break  # Success, exit retry loop
        except subprocess.CalledProcessError as e:
            print(f"ffplay failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print("Failed to play audio after all retries.")
            time.sleep(1)
        finally:
            # Clean up files if they exist
            for file in ["output.mp3", "output.wav"]:
                if os.path.exists(file):
                    try:
                        os.remove(file)
                    except Exception as e:
                        print(f"Failed to remove {file}: {e}")

def run_tests():
    try:
        # Use the absolute path to test_ai_agent.py
        test_script_path = os.path.abspath("test_ai_agent.py")
        if not os.path.exists(test_script_path):
            return "I couldn’t find the test script, honey! Make sure test_ai_agent.py is in the right place."
        
        # Run the test script with TEST_MODE=true
        env = os.environ.copy()
        env["TEST_MODE"] = "true"
        result = subprocess.run(
            ["python3", test_script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=300
        )
        # Read the test results
        with open("test_results.log", "r") as f:
            test_output = f.read()
        if result.stderr:
            test_output += f"\nErrors: {result.stderr}"
        return test_output
    except subprocess.TimeoutExpired:
        return "Tests timed out after 5 minutes."
    except FileNotFoundError:
        return "I couldn’t find the test script, honey! Make sure test_ai_agent.py is in the right place."
    except Exception as e:
        return f"Failed to run tests: {str(e)}"

def schedule_tests():
    try:
        while True:
            sassy_prefix = random.choice(friday_sassy_phrases)
            speak(f"{sassy_prefix} Running scheduled tests for you, sugar!")
            test_output = run_tests()
            speak(f"{sassy_prefix} Scheduled test results: {test_output}")
            log_friday_memory("Scheduled test", f"Ran scheduled tests: {test_output}")
            time.sleep(1800)  # Run every 30 minutes
    except Exception as e:
        sassy_prefix = random.choice(friday_sassy_phrases)
        speak(f"{sassy_prefix} Scheduled tests hit a snag, honey: {str(e)}")

def periodic_update_check():
    while True:
        try:
            sassy_prefix = random.choice(friday_sassy_phrases)
            speak(f"{sassy_prefix} Checking for updates in the background, honey!")
            result = check_for_updates()
            if isinstance(result, str):
                speak(f"{sassy_prefix} Found an update! Applying it now—hold tight!")
                apply_update(result)
            elif result is None:
                speak(f"{sassy_prefix} No updates available, sugar. I’m already up to date!")
            else:
                speak(f"{sassy_prefix} {result}")
        except Exception as e:
            sassy_prefix = random.choice(friday_sassy_phrases)
            speak(f"{sassy_prefix} Background update check failed, honey: {str(e)}")
        time.sleep(3600)  # Check every hour

def apply_update(new_script_path):
    try:
        # Create a backup of the current script
        backup_path = f"friday_agent_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        subprocess.run(["cp", "friday_agent.py", backup_path], check=True)
        # Replace the current script with the new one
        subprocess.run(["cp", new_script_path, "friday_agent.py"], check=True)
        sassy_prefix = random.choice(friday_sassy_phrases)
        speak(f"{sassy_prefix} I’ve updated myself, honey! Restarting now—be right back!")
        # Restart the script
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        sassy_prefix = random.choice(friday_sassy_phrases)
        speak(f"{sassy_prefix} Update failed, sugar: {str(e)}")
        raise

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
        return f"Failed to check for updates: {str(e)}"

def handle_friday(user_input):
    request = user_input.lower().replace("hey friday", "").strip()
    sassy_prefix = random.choice(friday_sassy_phrases)
    
    if "write a script to" in request:
        task = request.replace("write a script to", "").strip()
        answer = ask_grok(f"Write a Python script to {task}")
        speak(f"{sassy_prefix} Here’s your script to {task}:\n{answer}")
        log_friday_memory(user_input, f"Generated script to {task}: {answer}")
    elif "test this script" in request:
        script = request.replace("test this script", "").strip()
        try:
            exec_globals = {}
            exec(script, exec_globals)
            result = "Script ran successfully!"
        except Exception as e:
            result = f"Script failed: {str(e)}"
        speak(f"{sassy_prefix} I tested your script, and here’s what happened: {result}")
        log_friday_memory(user_input, f"Tested script: {script}, Result: {result}")
    elif "check this code" in request:
        code = request.replace("check this code", "").strip()
        with open("temp_code.py", "w") as f:
            f.write(code)
        try:
            result = subprocess.run(
                ["pylint", "temp_code.py"],
                capture_output=True,
                text=True
            )
            pylint_output = result.stdout + result.stderr
            if "Your code has been rated at 10.00/10" in pylint_output:
                feedback = "Your code looks perfect, honey! Pylint gives it a 10/10."
            else:
                feedback = f"Pylint says: {pylint_output}"
        except subprocess.CalledProcessError as e:
            feedback = f"Pylint had a little trouble, sugar: {str(e)}"
        finally:
            os.remove("temp_code.py")
        speak(f"{sassy_prefix} I checked your code, and here’s what I found: {feedback}")
        log_friday_memory(user_input, f"Checked code: {code}, Feedback: {feedback}")
    elif "what did i last ask about coding" in request or "what did i ask about coding last" in request:
        last_memory = get_last_friday_memory()
        speak(f"{sassy_prefix} Here’s the last coding thing you asked about: {last_memory}")
    elif "summarize my last coding requests" in request:
        if os.path.exists(FRIDAY_MEMORY_FILE):
            with open(FRIDAY_MEMORY_FILE, "r") as f:
                lines = f.readlines()
                coding_requests = [line for line in lines if "Generated script" in line or "Tested script" in line or "Checked code" in line][-5:]
                if coding_requests:
                    summary = "Here’s a summary of your last coding requests:\n"
                    for line in coding_requests:
                        summary += line.strip() + "\n"
                    speak(f"{sassy_prefix} {summary}")
                else:
                    speak(f"{sassy_prefix} I don’t have any recent coding requests in my memory, honey!")
        else:
            speak(f"{sassy_prefix} I don’t have any recent coding requests in my memory, honey!")
    elif "work on" in request and "in the background" in request:
        task = request.replace("work on", "").replace("in the background", "").strip()
        task_id = add_friday_task(task)
        start_friday_worker()
        speak(f"{sassy_prefix} I’m working on {task} in the background, task ID {task_id}. I’ll let you know when it’s done!")
        log_friday_memory(user_input, f"Started background task {task_id}: {task}")
    elif "status of my background tasks" in request or "how close are my tasks to being done" in request:
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
    elif "apply update" in request:
        new_script_path = request.replace("apply update", "").strip()
        if not new_script_path:
            speak(f"{sassy_prefix} You need to tell me the path to the new script, honey!")
        elif not os.path.exists(new_script_path):
            speak(f"{sassy_prefix} I can’t find that script at {new_script_path}, sugar!")
        else:
            speak(f"{sassy_prefix} Applying the update from {new_script_path}—hold tight!")
            apply_update(new_script_path)
    elif "check for updates" in request:
        speak(f"{sassy_prefix} Checking for updates from the remote repository, honey!")
        result = check_for_updates()
        if isinstance(result, str):
            speak(f"{sassy_prefix} Found an update! Applying it now—hold tight!")
            apply_update(result)
        elif result is None:
            speak(f"{sassy_prefix} No updates available, sugar. I’m already up to date!")
        else:
            speak(f"{sassy_prefix} {result}")
    elif "i didn't hear a notification" in request:
        speak(f"{sassy_prefix} I’m sorry you didn’t hear the notification, honey! Let me check the task status for you.")
        tasks = load_friday_tasks()
        if not tasks:
            speak(f"{sassy_prefix} No background tasks right now, honey!")
        else:
            status = "Here’s the status of your background tasks:\n"
            for task in tasks:
                status += f"Task {task['id']}: {task['description']} - {task['status']}"
                if task["result"]:
                    status += f" (Result: {task['result']})"
                status += "\n"
            speak(f"{sassy_prefix} {status}")
    elif "you don't actually have a virtual test environment did you just lie to me" in request:
        speak(f"{sassy_prefix} Oh, honey, I’m sorry if I gave you the wrong impression! I don’t have a real virtual test environment—I’m an AI, so I simulate things in my own way. I should’ve been clearer about that. Let’s fix that misunderstanding—what do you need help with?")
    elif "update yourself based on instructions" in request or "fix yourself based on logs" in request:
        speak(f"{sassy_prefix} Let me take a look at my logs and see what I can do to fix myself, honey!")
        try:
            with open("test_ai_agent.log", "r") as f:
                test_logs = f.readlines()[-50:]
            with open("friday_memory.txt", "r") as f:
                memory_logs = f.readlines()[-50:]
            logs = "\n".join(test_logs + memory_logs)
        except Exception as e:
            speak(f"{sassy_prefix} I had trouble reading my logs, sugar: {str(e)}")
            return

        prompt = f"I’m Friday, an AI assistant. I need to update my code based on these logs:\n{logs}\nPlease suggest changes to my code to fix any issues you find."
        suggested_fixes = ask_grok(prompt)
        speak(f"{sassy_prefix} Here’s what I found for fixes:\n{suggested_fixes}")

        try:
            subprocess.run(["cp", "friday_agent.py", f"friday_agent_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"], check=True)
            with open("friday_agent.py", "r") as f:
                current_code = f.read()
            apply_prompt = f"Here’s my current code:\n{current_code}\nApply these fixes:\n{suggested_fixes}\nReturn the updated code."
            updated_code = ask_grok(apply_prompt)
            with open("friday_agent_new.py", "w") as f:
                f.write(updated_code)
            apply_update("friday_agent_new.py")
        except Exception as e:
            speak(f"{sassy_prefix} I ran into a problem applying the fixes, honey: {str(e)}")
    else:
        if request:
            answer = ask_grok(request)
            speak(f"{sassy_prefix} {answer}")
        else:
            speak(f"{sassy_prefix} I’m on it, honey—busy as ever, but I’ll make it work! What do you need?")
        log_friday_memory(user_input, f"Logged request: {request}")

def monitor_tests():
    while True:
        try:
            with open("test_results.log", "r") as f:
                test_results = f.read()
            if "Test execution failed" in test_results:
                sassy_prefix = random.choice(friday_sassy_phrases)
                speak(f"{sassy_prefix} Uh-oh, honey! The tests failed. Let me take a look and fix it for you!")
                with open("test_ai_agent.log", "r") as f:
                    test_logs = f.readlines()[-50:]
                with open("friday_memory.txt", "r") as f:
                    memory_logs = f.readlines()[-50:]
                logs = "\n".join(test_logs + memory_logs)
                prompt = f"I’m Friday, an AI assistant. My tests are failing. Here are the logs:\n{logs}\nPlease suggest changes to my code to fix the test failures."
                suggested_fixes = ask_grok(prompt)
                speak(f"{sassy_prefix} Here’s what I found to fix the tests:\n{suggested_fixes}")
                # Apply the fixes
                subprocess.run(["cp", "friday_agent.py", f"friday_agent_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"], check=True)
                with open("friday_agent.py", "r") as f:
                    current_code = f.read()
                apply_prompt = f"Here’s my current code:\n{current_code}\nApply these fixes:\n{suggested_fixes}\nReturn the updated code."
                updated_code = ask_grok(apply_prompt)
                with open("friday_agent_new.py", "w") as f:
                    f.write(updated_code)
                apply_update("friday_agent_new.py")
        except Exception as e:
            sassy_prefix = random.choice(friday_sassy_phrases)
            speak(f"{sassy_prefix} I had trouble monitoring the tests, honey: {str(e)}")
        time.sleep(600)  # Check every 10 minutes

# Start the test monitoring thread in the main block
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