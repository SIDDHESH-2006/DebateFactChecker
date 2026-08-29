import asyncio
from fastapi import FastAPI, WebSocket
from contextlib import asynccontextmanager
# 1. Define the Queues globally so all parts of the app can access them
audio_queue = asyncio.Queue()
text_queue = asyncio.Queue()
outbound_queue = asyncio.Queue()

# 2. Define the Background Workers (The "Chefs")
async def transcription_worker():
    """Pulls from audio_queue, sends to Speech-to-Text, pushes to text_queue."""
    while True:
        # await audio_queue.get()
        # Process...
        await asyncio.sleep(0.1) # Prevents CPU locking for this skeleton

async def ai_logic_worker():
    """Pulls from text_queue, runs OpenAI fact-check, pushes to outbound_queue."""
    while True:
        # await text_queue.get()
        # Process...
        await asyncio.sleep(0.1)

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