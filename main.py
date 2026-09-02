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
    transcript_buffer = ""  # 1. Create a memory buffer for the paragraph

    try:
        while True:
            audio_data = await audio_queue.get()
            
            # Check for the kill signal (client disconnected)
            if audio_data is None:
                if dg_connection is not None:
                    print("Closing Deepgram connection safely...")
                    await dg_connection.finish()
                    dg_connection = None
                # Clear the buffer on disconnect
                transcript_buffer = ""
                audio_queue.task_done()
                continue

            # Initialize Deepgram ONLY when the first audio chunk arrives
            if dg_connection is None:
                dg_connection = deepgram.listen.asyncwebsocket.v("1")
                
                async def on_message(self, result, **kwargs):
                    nonlocal transcript_buffer  # 2. Let the handler modify our outside buffer
                    
                    sentence = result.channel.alternatives[0].transcript
                    if not sentence:
                        return

                    # 3. Only keep finalized text blocks
                    if result.is_final:
                        transcript_buffer += sentence + " "
                        
                        # 4. "speech_final" triggers when Deepgram detects the 800ms pause
                        if result.speech_final:
                            complete_paragraph = transcript_buffer.strip()
                            if complete_paragraph:
                                print(f"📝 Completed Paragraph: '{complete_paragraph}'")
                                await text_queue.put(complete_paragraph)
                            
                            # 5. Reset the buffer for the next time they speak
                            transcript_buffer = ""

                dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

                # 6. Add endpointing (silence detection) to your options
                options = LiveOptions(
                    model="nova-2",
                    language="en-US",
                    smart_format=True,
                    endpointing=800, # Wait for 800ms of silence before concluding the thought
                )
                
                if await dg_connection.start(options) is False:
                    print("Failed to connect to Deepgram")
                    dg_connection = None
                    audio_queue.task_done()
                    continue

                print("Deepgram connection established!")

            # Send the actual audio data
            await dg_connection.send(audio_data)
            audio_queue.task_done()

    except asyncio.CancelledError:
        print("Shutting down Deepgram connection (CTRL+C)...")
        if dg_connection:
            await dg_connection.finish()
            
    except Exception as e:
        print(f"Transcription worker error: {e}")

async def ai_logic_worker():
    """Pulls text, delegates to engine.py, and pushes the result to outbound_queue."""
    try:
        while True:
            # 1. Wait for text to arrive
            text = await text_queue.get()
            print(f"🧠 AI Worker evaluating: '{text}'...")

            # 2. Process the text in its own isolated error block
            try:
                result_json = await verify_claims(text)
                print(f"✅ Verdict ready: {result_json['status']}")
                await outbound_queue.put(result_json)
            except Exception as e:
                print(f"AI Worker error: {e}")
            finally:
                # Only mark the task done if we successfully grabbed it from the queue
                text_queue.task_done()

    except asyncio.CancelledError:
        # 3. Catch the CTRL+C shutdown signal and exit instantly
        print("Shutting down AI Worker (CTRL+C)...")

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