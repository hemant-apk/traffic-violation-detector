# Traffic Violation Detector

A tool that detects traffic violations from street camera footage using **Google's Gemini 2.5 Pro Model**.

---

## 🚦 What It Does

This tool reviews traffic videos and automatically flags violations such as:

- Rash or dangerous driving  
- Running red lights  
- Riding without a helmet  
- Triple riding on two-wheelers  
- Improper or defective number plates  
- Using a mobile phone while driving  
- Not giving way to emergency vehicles  
- Obstructive or illegal parking  
- Illegal U-turns or wrong-way driving  
- Hit-and-run incidents  
- Road rage or public fights  
- Overloaded vehicles  
- Stray animals on the road  
- ...and more — all based on what’s visible in the footage

---

## 🧠 How It Works

1. The system **watches the entire video** to understand the environment and traffic flow.
2. It then **revisits the video**, frame by frame, to detect possible violations.
3. For each detected violation, it logs:
   - What happened (violation type)
   - Who was involved (e.g., “red hatchback”, “pedestrian in black shirt”)
   - When it happened (timestamp)
4. It generates a structured report in:
   - `JSON` format for programmatic use

5. Real-time **violation alerts** are overlaid directly onto the video output.

---
