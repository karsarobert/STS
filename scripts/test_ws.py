#!/home/karsa-robert/miniconda3/bin/python3
"""Test the STS WebSocket - fixed event handling."""
import asyncio, json, websockets, sys

async def test():
    uri = 'ws://localhost:8765/v1/realtime'
    print(f"Connecting to {uri}...", flush=True)
    async with websockets.connect(uri, max_size=10_000_000) as ws:
        msg = json.loads(await ws.recv())
        print(f"  <- {msg['type']}", flush=True)
        
        # Session update - minimal config
        await ws.send(json.dumps({
            'type': 'session.update',
            'session': {
                'modalities': ['text', 'audio'],
                'turn_detection': None,
            }
        }))
        resp = json.loads(await ws.recv())
        print(f"  <- {resp['type']}: {resp.get('error',{}).get('message','ok')}", flush=True)
        
        # Conversation item
        await ws.send(json.dumps({
            'type': 'conversation.item.create',
            'item': {
                'type': 'message',
                'role': 'user',
                'content': [{'type': 'input_text', 'text': 'Say hello in 5 words.'}]
            }
        }))
        
        # Create response
        await ws.send(json.dumps({
            'type': 'response.create',
            'response': {'modalities': ['text', 'audio']}
        }))
        
        audio_total = 0
        transcript = ""
        for i in range(30):
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
                t = msg['type']
                
                if t == 'response.audio_transcript.delta':
                    transcript += msg['delta']
                elif t == 'response.audio.delta':
                    audio_total += len(msg['delta'])
                elif t == 'response.audio.done':
                    pass
                elif 'audio' in t and 'delta' in t:
                    audio_total += len(msg.get('delta', ''))
                elif t in ('error',):
                    print(f"  <- {t}: {msg}", flush=True)
                elif t in ('response.done',):
                    print(f"  <- {t}", flush=True)
                    break
                elif t not in ('response.created','conversation.item.created',
                              'response.output_item.added','response.content_part.added',
                              'response.content_part.done','response.output_item.done',
                              'response.audio_transcript.done','response.audio.done',
                              'response.text.done','response.text.delta'):
                    print(f"  <- {t}", flush=True)
            except asyncio.TimeoutError:
                print("  TIMEOUT", flush=True)
                break
        
        print(f"\nTranscript: '{transcript}'", flush=True)
        print(f"Audio: {audio_total} bytes", flush=True)
        
        if audio_total > 500:
            print("SUCCESS: Pipeline works end-to-end!", flush=True)
            # Save audio to file
            return True
        else:
            print("WARNING: No audio received", flush=True)
            return False

success = asyncio.run(test())
sys.exit(0 if success else 1)
