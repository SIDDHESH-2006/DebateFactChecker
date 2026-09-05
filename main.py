import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import os

from dotenv import load_dotenv
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from engine import verify_claims

# Loads the variables from your .env file into the system
load_dotenv()

# Initialize the global Deepgram client
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
deepgram = DeepgramClient(api_key=deepgram_api_key)

# 1. Define the Queues globally so all parts of the app can access them
audio_queue = asyncio.Queue()
text_queue = asyncio.Queue()
outbound_queue = asyncio.Queue()

# 2. Define the Background Workers (The "Chefs")
async def transcription_worker():
    """Pulls raw audio from the queue and streams it to Deepgram."""
    dg_connection = None
    transcript_buffer = ""
    current_speaker = None  # <--- Track who is currently holding the floor

    try:
        while True:
            audio_data = await audio_queue.get()
            
            if audio_data is None:
                if dg_connection is not None:
                    print("Closing Deepgram connection safely...")
                    try:
                        await asyncio.wait_for(dg_connection.finish(), timeout=2.0)
                    except Exception:
                        pass
                    dg_connection = None
                
                transcript_buffer = ""
                current_speaker = None
                audio_queue.task_done()
                continue

            if dg_connection is None:
                dg_connection = deepgram.listen.asyncwebsocket.v("1")
                
                async def on_message(self, result, **kwargs):
                    nonlocal transcript_buffer, current_speaker # <-- Bring memory into the function
                    
                    sentence = result.channel.alternatives[0].transcript
                    words = result.channel.alternatives[0].words

                    if not sentence or not words:
                        return

                    if result.is_final:
                        # 1. Who is speaking right now?
                        chunk_speaker = words[0].speaker

                        # 2. INTERRUPT DETECTED: If it's a new speaker, and the buffer isn't empty
                        if current_speaker is not None and chunk_speaker != current_speaker and transcript_buffer.strip():
                            complete_paragraph = transcript_buffer.strip()
                            dialogue_line = f"Speaker {current_speaker}: {complete_paragraph}"
                            
                            print(f"⚡ Interruption! Sending: {dialogue_line}")
                            await text_queue.put(dialogue_line)
                            
                            # Wipe the buffer clean for the new speaker
                            transcript_buffer = ""

                        # 3. Update the memory and add words to the buffer
                        current_speaker = chunk_speaker
                        transcript_buffer += sentence + " "
                        
                        # 4. SILENCE DETECTED: 2 seconds of silence has passed
                        if result.speech_final:
                            complete_paragraph = transcript_buffer.strip()
                            if complete_paragraph:
                                dialogue_line = f"Speaker {current_speaker}: {complete_paragraph}"
                                
                                print(f"📝 {dialogue_line}")
                                await text_queue.put(dialogue_line)
                                
                            # Reset completely for whoever speaks next
                            transcript_buffer = ""
                            current_speaker = None

                dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

                options = LiveOptions(
                    model="nova-2",
                    language="en-IN",
                    smart_format=True,
                    endpointing=1000,
                    diarize=True,
                )
                
                if await dg_connection.start(options) is False:
                    print("Failed to connect to Deepgram")
                    dg_connection = None
                    audio_queue.task_done()
                    continue

                print("Deepgram connection established!")

            await dg_connection.send(audio_data)
            audio_queue.task_done()

    except asyncio.CancelledError:
        print("Shutting down Deepgram connection (CTRL+C)...")
        if dg_connection:
            try:
                await asyncio.wait_for(dg_connection.finish(), timeout=2.0)
            except Exception:
                pass
        raise
            
    except Exception as e:
        print(f"Transcription worker error: {e}")
async def ai_logic_worker():
    """Pulls text, delegates to engine.py, and pushes the result to outbound_queue."""
    debate_history = []  # Store the memory of the ENTIRE debate session
    
    try:
        while True:
            text = await text_queue.get()
            
            # Handle session reset to wipe memory when "Stop Listening" is pressed
            if text is None:
                print("🧹 Stop Listening pressed: Wiping entire debate history for the next session.")
                debate_history.clear()
                text_queue.task_done()
                continue
                
            print(f"🧠 AI Worker evaluating: '{text}'...")

            try:
                # Pass the full history list to the engine
                result_json = await verify_claims(text, debate_history)
                print(f"✅ Verdict ready: {result_json['status']}")
                
                # Append this statement to the permanent session history
                debate_history.append(text)
                
                # (Notice we removed the 5-item limit! It now stores EVERYTHING until Stop is pressed)
                    
                await outbound_queue.put(result_json)
            except Exception as e:
                print(f"AI Worker error: {e}")
            finally:
                text_queue.task_done()

    except asyncio.CancelledError:
        print("Shutting down AI Worker (CTRL+C)...")
        raise

# 3. The Lifespan Manager: Boots the workers alongside the server
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the background workers when the server boots
    task1 = asyncio.create_task(transcription_worker())
    task2 = asyncio.create_task(ai_logic_worker())
    
    yield # The server runs and handles users here
    
    # Clean up when the server shuts down
    print("Cancelling background tasks...")
    task1.cancel()
    task2.cancel()
    
    # Force FastAPI to wait for the tasks to gracefully close their connections
    await asyncio.gather(task1, task2, return_exceptions=True)
    print("All background tasks shut down completely.")

# Initialize FastAPI with the lifespan
app = FastAPI(lifespan=lifespan)

@app.get("/")
async def get_ui():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(html_content)

# 4. The Gateway
@app.websocket("/listen")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    async def receive_audio():
        """Reads audio chunks from the client and feeds the audio queue."""
        try:
            while True:
                data = await websocket.receive_bytes()
                await audio_queue.put(data)
        except WebSocketDisconnect:
            print("Audio receive loop ended (client disconnected)")
            await audio_queue.put(None) # Kill signal to transcription_worker

    async def send_results():
        """Pulls completed fact-check cards and pushes them to the client."""
        try:
            while True:
                card_data = await outbound_queue.get()
                await websocket.send_json(card_data)
                outbound_queue.task_done()
        except WebSocketDisconnect:
            print("Send results loop ended")
        except asyncio.CancelledError:
            pass # Gracefully exit when the sibling task cancels this one

    # Create distinct tasks for both loops
    receive_task = asyncio.create_task(receive_audio())
    send_task = asyncio.create_task(send_results())

    # Wait for WHICHEVER task finishes/fails first
    done, pending = await asyncio.wait(
        [receive_task, send_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Instantly cancel the other task so it doesn't hang the server
    for task in pending:
        task.cancel()
        
    print("WebSocket disconnected completely")