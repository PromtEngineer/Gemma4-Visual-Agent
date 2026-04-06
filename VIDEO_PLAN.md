# Video Plan: Why Gemma Alone Isn't Enough

## Hook (0:00 - 0:30)
**Visual**: Side-by-side comparison result from the Compare tab
**Script angle**: "I asked two AI models the same question: how many cars are in this image? One said 'approximately 10 taxis.' The other said exactly 16, and showed me every single one. Here's what's different."

## Part 1: The Demo (0:30 - 3:00)
Show the app running live. Three quick examples:

1. **Dogs example** — "How many dogs and what breeds?"
   - Show the step-by-step: Falcon segments → Gemma reasons
   - Highlight the model badges at each step

2. **Street scene** — "Are there more cars than people?"
   - Show the 4-step pipeline: DETECT cars → DETECT people → COMPARE → ANSWER
   - This is the money shot — exact counts with colored masks

3. **Compare tab** — Same query, Gemma-only vs Falcon+Gemma
   - Left panel: Gemma hedges ("approximately 25-30")
   - Right panel: Falcon gives 31 exact with pixel masks
   - "This is why a VLM alone isn't enough"

## Part 2: The Problem (3:00 - 5:00)
**Visual**: Comparison table diagram

**Key points**:
- VLMs like Gemma are incredible at understanding scenes, describing context, identifying breeds
- But they fundamentally cannot do three things:
  1. **Exact counting** — They estimate, especially above 5 objects
  2. **Spatial grounding** — They can say "left" or "right" but can't give you pixel coordinates
  3. **Instance separation** — They can't distinguish overlapping objects
- These aren't bugs — it's architectural. VLMs output text tokens, not coordinates or masks
- For real applications (autonomous driving, inventory, surveillance, accessibility), you need precision

## Part 3: The Solution — Falcon Perception (5:00 - 8:00)
**Visual**: Falcon architecture diagram (Chain-of-Perception)

**Key points**:
- Falcon Perception from TII — 0.6B params, open source
- Single early-fusion Transformer (not the typical separate-encoder approach)
- Image patches and text tokens processed together from layer 1
- Chain-of-Perception decoding: for each object it outputs coord → size → seg
- The seg embedding gets dot-producted with upsampled image features for a full-resolution mask
- 0.6B params — tiny. Runs on a MacBook via MLX
- Show: what the raw output looks like (bounding boxes + masks)

## Part 4: The Architecture — Agentic Pipeline (8:00 - 12:00)
**Visual**: Full architecture diagram + agentic loop diagram

**Key points**:
- The insight: combine a perception model (Falcon) with a reasoning model (Gemma)
- Falcon provides the structured evidence, Gemma reasons over it
- The agent decides which tools to use based on the query:
  - Simple counting → DETECT → ANSWER
  - Comparison → DETECT A → DETECT B → COMPARE → ANSWER
  - Complex question → VLM → RE-PLAN → (Gemma decides next step)
  - Describe specific object → DETECT → CROP → VLM → ANSWER
- Walk through the agentic loop: how Gemma can trigger new detection steps
- Show the step-by-step UI rendering each step with model attribution

## Part 5: Code Walkthrough (12:00 - 16:00)
**Visual**: Code on screen

Key files to walk through:
1. **vision_studio.py** — The FastAPI app, SSE streaming, how events flow
2. **agent_studio.py** — The core pipeline:
   - `_detect()` — How Falcon is called
   - `_vlm()` — How Gemma is called
   - `initial_plan()` — How queries get routed to tool sequences
   - `execute_agent_events()` — The agentic loop
   - `_ask_gemma_for_next_step()` — Re-planning
3. **video_tracker.py** — IoU-based tracking across frames

## Part 6: Video Tracking (16:00 - 18:00)
**Visual**: Tracked video output

- Same Falcon model, but applied per-frame
- Simple IoU tracker maintains object IDs across frames
- Show the dogs video: 2 dogs tracked consistently across 20 frames
- Show the street video: cars tracked with persistent IDs
- Practical applications: traffic monitoring, wildlife counting, retail analytics

## Part 7: What This Means (18:00 - 20:00)
**Visual**: "When to use what" decision diagram

- VLMs are not enough for production vision applications that need precision
- But VLMs are essential for the reasoning layer — interpreting what the detections mean
- The combination is greater than the sum: Falcon provides evidence, Gemma provides understanding
- Both models run locally on a MacBook — no cloud needed
- This pattern (perception + reasoning) is the future of practical vision AI
- All code is open source: link to repo

## Closing (20:00 - 20:30)
- Link to GitHub repo
- "Try it yourself — one command to run"
- Subscribe / like
