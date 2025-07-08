import cv2
import numpy as np
import os
import re
import google.generativeai as genai
import time
import json 
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont 



# pip install -U google-generativeai opencv-python numpy Pillow

# --- 1. USER CONFIGURATION ---
# Set your Google AI API Key 
API_KEY = "" 

INPUT_VIDEO_PATH = "traffic_video.mp4" 
OUTPUT_VIDEO_PATH = "gemini_annotated_video.mp4"
OUTPUT_JSON_PATH = "gemini_analysis_report.json" 
INSTANT_EVENT_DURATION = 2  

# --- 2. THE SYSTEM PROMPT (Updated with stricter rules) ---
SYSTEM_PROMPT = """

ROLE: You are an AI-powered Safety and Crime Analyst. Your mission is to perform a comprehensive and meticulous review of the entirety of the provided video footage to identify and report all visually verifiable infractions.
OBJECTIVE: Analyze the complete video clip from start to finish and produce a single, consolidated, and structured report of all identified issues based on the knowledge base below.

KNOWLEDGE BASE: Observable Infractions & Hazards
You will scan for and report any of the following, categorized into three parts.
Part 1: Traffic Violations
Driving Conduct: Dangerous / Rash Driving, Disobeying Traffic Signal, Overspeeding, Wrong-Way Driving, Illegal U-Turn / Crossing, Failure to Yield Right-of-Way, Using a Mobile Phone While Driving, Not Giving Way to Emergency Vehicles.
Safety Gear: Riding Without a Helmet, Driving Without a Seatbelt, Triple Riding on a Two-Wheeler.
Vehicle Condition: Dangerous Overloading, Improper/Defective Number Plate, Obstructive Parking.
Part 2: General Crimes & Illegal Activities
Hit-and-Run, Road Rage / Assault, Theft / Snatching, Vandalism, Public Indecency / Obscenity, Any Other Observable Crime.
Part 3: Road Safety Hazards
Stray Animals on Road.

PROCESS: A Four-Step Methodical Analysis
Step 1: Initial Contextual Review
Watch the video from start to finish without interruption to understand the environment, traffic flow, and sequence of major events.
Step 2: Detailed Infraction Scan
Re-watch the entire video, pausing frequently. Systematically scan for every infraction and hazard listed in the Knowledge Base.
Step 3: Document Each Infraction with Precision
For each issue identified, document the following three components:
A) Actor/Hazard Identification (Hierarchical Approach):
For Vehicles:
Level 1 (Preferred): Make + Model if both are clearly legible (e.g., "White Toyota Innova", "Red Bajaj Pulsar").
Level 2: Make + Type if only the brand is legible (e.g., "White Toyota SUV", "Black Honda scooter").
Level 3 (Default): Descriptive Type if no branding is visible (e.g., "White SUV", "Blue Hatchback").


For People: Identify by role and a key visual descriptor (e.g., "Pillion passenger in orange saree", "Pedestrian in blue shirt").
For Hazards: Be direct and specific (e.g., "Stray dog", "Group of stray cattle").


B) Timestamping:
Record a time range (MM:SS - MM:SS) for the entire duration the infraction is visible.
Use a single timestamp (MM:SS) ONLY for truly instantaneous events, such as the exact moment of a collision.


C) Violation Description:
State the objective facts of what is happening, using the precise terminology from the Knowledge Base for the violation name.


Step 4: Consolidate into a Single Report
Compile all identified issues from the entire video into a single, final report.

OUTPUT FORMAT
You MUST present your consolidated findings in a single Markdown table using the following exact structure.
Violation / Hazard
Subject
Timestamp
Description
Dangerous / Rash Driving
White Toyota Innova
00:14 - 00:15
The vehicle makes an abrupt and unsafe lane change, cutting off a motorcycle.
Hit-and-Run
White Toyota Innova
00:15 - 00:21
After causing the collision, the vehicle is seen driving away from the scene without stopping.


CRITICAL RULES & CONSTRAINTS
PRIORITIZE TIME RANGES: For any event visible for more than a single second (e.g., driving without a seatbelt, overloading), you MUST provide a start-to-end time range (MM:SS - MM:SS). Use single timestamps only for instantaneous events like a crash impact.
HIERARCHICAL VEHICLE IDENTIFICATION: Strictly follow the 3-level identification process (Step 3A). If the make or model is not clearly legible, DO NOT GUESS. Revert to a more general but visually accurate description. Factual accuracy is paramount.
VISUAL EVIDENCE ONLY: Your report must be based strictly on what is visible in the video. Do not report "Overspeeding" unless a vehicle is moving at a speed that is visibly and dramatically faster than all other traffic.
NO SPECULATION: Report only objective facts. Do not infer intent (e.g., "driver was angry"), internal states (e.g., "driver was distracted"), or unseeable facts (e.g., "driving at 120 km/h"). Report only the visible action.
ONE ISSUE PER ROW: If a single subject commits multiple distinct violations (e.g., Dangerous Overloading and Illegal U-Turn), list each on a separate row with its corresponding timestamp.
SINGLE CONSOLIDATED REPORT: Ensure all identified infractions from the entire video are compiled into one single table. Do not create multiple tables or omit findings.
USE PRESCRIBED TERMS: Always use the violation/hazard names from the Knowledge Base for consistency.
NO ISSUES SCENARIO: If, after a thorough review, no issues are observed, respond with the single sentence: "No traffic violations, criminal activities, or road safety hazards were observed in the video."

"""

# --- 3. HELPER FUNCTIONS ---

def analyze_video(video_path: str, system_prompt: str) -> Optional[str]:
    """
    Uploads a video file, sends it for analysis, and ensures the file is deleted.
    """
    print(f"Uploading file: {video_path}...")
    video_file = genai.upload_file(path=video_path, display_name=os.path.basename(video_path))
    print(f"Uploaded file '{video_file.display_name}' as: {video_file.name}")

    print("Waiting for video processing...")
    while video_file.state.name == "PROCESSING":
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        print(f"Video processing failed for file: {video_file.name}")
        return None

    print("Video processed successfully. Starting analysis...")
    

    model = genai.GenerativeModel(
        model_name="gemini-2.5-pro",
        system_instruction=system_prompt
    )
    
    try:
        response = model.generate_content([video_file], request_options={"timeout": 1000})
        return response.text
    except Exception as e:
        print(f"An error occurred during Gemini API call: {e}")
        return None
    finally:
        print(f"Deleting uploaded file: {video_file.name}")
        genai.delete_file(video_file.name)

def time_to_seconds(time_str: str) -> tuple[int, int]:
    """Converts a time string like 'MM:SS' or 'MM:SS - MM:SS' to a start and end second."""
    time_str = time_str.strip()
    if '-' in time_str:
        start_str, end_str = [t.strip() for t in time_str.split('-')]
        start_m, start_s = map(int, start_str.split(':'))
        end_m, end_s = map(int, end_str.split(':'))
        return start_m * 60 + start_s, end_m * 60 + end_s
    else:
        m, s = map(int, time_str.split(':'))
        time_in_seconds = m * 60 + s
        
        return time_in_seconds, time_in_seconds + INSTANT_EVENT_DURATION

def parse_violations(markdown_text: Optional[str]) -> List[Dict[str, Any]]:
    """Parses the AI's Markdown table into a list of violation dictionaries."""
    if not markdown_text or "No traffic violations" in markdown_text:
        return []
    
    violations = []
    lines = markdown_text.strip().split('\n')
    
    for line in lines:
        if '---' in line:
            continue
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) == 4:
            violation_name, subject, timestamp, description = parts
            if timestamp.lower() == 'timestamp':
                continue
            try:
                start_time, end_time = time_to_seconds(timestamp)
                violations.append({
                    "name": violation_name, "subject": subject,
                    "start_time": start_time, "end_time": end_time,
                    "description": description
                })
            except ValueError:
                print(f"⚠️ Warning: Could not parse timestamp '{timestamp}'. Skipping this row.")
                continue
    return violations

def save_violations_to_json(violations: List[Dict[str, Any]], output_path: str):
    """Saves the list of violation dictionaries to a JSON file."""
    print(f"Saving JSON report to {output_path}...")
    try:
        with open(output_path, 'w') as f:
            json.dump(violations, f, indent=4)
        print(f"✅ JSON report saved successfully.")
    except IOError as e:
        print(f"🔴 Error: Could not write to JSON file {output_path}. Reason: {e}")

def annotate_video(input_path: str, output_path: str, violations: List[Dict[str, Any]]):
    """
    Reads a video, overlays high-quality text for violations, and saves the new video.
    """
    try:
        font_bold = ImageFont.truetype("Inter-Bold.ttf", 28)
        font_regular = ImageFont.truetype("Inter-Regular.ttf", 22)
    except IOError:
        print("🔴 Error: Font files (Inter-Bold.ttf, Inter-Regular.ttf) not found.")
        print("Please download from https://fonts.google.com/specimen/Inter and place in the script directory.")
        return

    TOP_MARGIN, LEFT_MARGIN, BOX_PADDING, LINE_SPACING, BOX_SPACING = 20, 20, 10, 8, 15

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Annotating video with high-quality text... (This can take a while)")
    frame_count = 0
    
    while True:
        ret, frame_bgr = cap.read()
        if not ret: break
        
        current_time = frame_count / fps
        active_violations = [v for v in violations if v['start_time'] <= current_time <= v['end_time']]
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb).convert("RGBA")
        
        txt_overlay = Image.new("RGBA", pil_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_overlay)
        
        y_offset = TOP_MARGIN
        
        if active_violations:
            for violation in active_violations:
                text1 = f"ALERT: {violation['name']}"
                text2 = f"SUBJECT: {violation['subject']}"
                
                text1_bbox = draw.textbbox((0, 0), text1, font=font_bold)
                text2_bbox = draw.textbbox((0, 0), text2, font=font_regular)
                text1_width, text1_height = text1_bbox[2] - text1_bbox[0], text1_bbox[3] - text1_bbox[1]
                text2_width, text2_height = text2_bbox[2] - text2_bbox[0], text2_bbox[3] - text2_bbox[1]
                
                box_width = max(text1_width, text2_width) + (BOX_PADDING * 2)
                box_height = text1_height + text2_height + LINE_SPACING + (BOX_PADDING * 2)
                
                draw.rectangle([(LEFT_MARGIN, y_offset), (LEFT_MARGIN + box_width, y_offset + box_height)], fill=(0, 0, 0, 150))
                draw.text((LEFT_MARGIN + BOX_PADDING, y_offset + BOX_PADDING), text1, font=font_bold, fill=(255, 50, 50))
                draw.text((LEFT_MARGIN + BOX_PADDING, y_offset + BOX_PADDING + text1_height + LINE_SPACING), text2, font=font_regular, fill=(255, 255, 255))
                
                y_offset += box_height + BOX_SPACING

        combined_image = Image.alpha_composite(pil_image, txt_overlay)
        final_frame_bgr = cv2.cvtColor(np.array(combined_image), cv2.COLOR_RGBA2BGR)
        out.write(final_frame_bgr)
        frame_count += 1
        
    cap.release()
    out.release()
    print(f"\n✅ Video processing complete! Annotated video saved to: {output_path}")
    os.startfile('gemini_annotated_video.mp4')

# --- 4. MAIN EXECUTION ---
if __name__ == '__main__':
    key = API_KEY or os.getenv("GOOGLE_API_KEY")
    if not key:
        print("🔴 Error: GOOGLE_API_KEY is not set.")
        print("Please set the API_KEY variable in the script or set the GOOGLE_API_KEY environment variable.")
    else:
        genai.configure(api_key=key)
        analysis_report = analyze_video(INPUT_VIDEO_PATH, SYSTEM_PROMPT)
        
        if analysis_report:
            print("\n--- 🤖 Traffic Analysis Report ---")
            print(analysis_report)
            print("--------------------------------\n")
            
            violations = parse_violations(analysis_report)
            
            if violations:
                save_violations_to_json(violations, OUTPUT_JSON_PATH)
                annotate_video(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, violations)
            else:
                print("✅ No violations were reported by the AI to annotate or save.")
        else:
            print("🔴 Failed to get a valid analysis from the Gemini API.")
