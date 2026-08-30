import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

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
    try:
        # 1. Create the Deepgram live connection
        dg_connection = deepgram.listen.asyncwebsocket.v("1")

        # 2. Define what happens when Deepgram returns text
        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if sentence:  # Only process if there are actual words
                print(f"📝 Deepgram heard: '{sentence}'")
                await text_queue.put(sentence)

        # Bind the handler to the Transcript event
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        # 3. Start the connection with the Nova-2 model
        options = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
        )
        
        if await dg_connection.start(options) is False:
            print("Failed to connect to Deepgram")
            return

        print("Deepgram connection established!")

        # 4. Loop forever: pull from our audio queue and send to Deepgram
        while True:
            audio_data = await audio_queue.get()
            await dg_connection.send(audio_data)
            audio_queue.task_done()

    except Exception as e:
        print(f"Transcription worker error: {e}")

async def ai_logic_worker():
    """Simulates the RAG pipeline checking a claim."""
    while True:
        # 1. Wait for text to arrive from the transcription worker
        text = await text_queue.get()
        print(f"🧠 AI Worker evaluating: '{text}'...")
        
        # 2. Simulate API processing time (the bottleneck we are trying to avoid)
        await asyncio.sleep(2)
        
        # 3. Create the fake result card
        fake_result = {
            "claim": text,
            "status": "FALSE",
            "reason": "Satellite imagery and physics confirm the Earth is an oblate spheroid."
        }
        
        # 4. Push to the final queue for the WebSocket to send!
        await outbound_queue.put(fake_result)
        text_queue.task_done()


# 3. The Lifespan Manager: Boots the workers alongside the server
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the background workers when the server boots
    task1 = asyncio.create_task(transcription_worker())
    task2 = asyncio.create_task(ai_logic_worker())
    
    yield # The server runs and handles users here
    
    # Clean up when the server shuts down
    task1.cancel()
    task2.cancel()

# Initialize FastAPI with the lifespan
app = FastAPI(lifespan=lifespan)
@app.get("/")
async def get_ui():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(html_content)

# 4. The Gateway (The "Waiter")
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

    async def send_results():
        """Pulls completed fact-check cards and pushes them to the client."""
        try:
            while True:
                # Wait for the AI worker to produce a verified card
                card_data = await outbound_queue.get()
                await websocket.send_json(card_data)
                outbound_queue.task_done()
        except WebSocketDisconnect:
            print("Send results loop ended")

    # Run both loops simultaneously on this connection
    try:
        await asyncio.gather(receive_audio(), send_results())
    except WebSocketDisconnect:
        print("WebSocket disconnected completely")