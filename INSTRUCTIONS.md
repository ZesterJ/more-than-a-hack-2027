# ⚽ Barca Vision Analyzer -- Quickstart & User Guide

Welcome to **Barca Vision Analyzer**! This is a tool for inspecting possession models against the ground truth, built for the **More Than A Hack 2027** ball possession challenge (FC Barcelona vs Athletic Club).

It provides a synchronized dual-view experience: the **official match broadcast video** runs in lockstep with a **2D tactical radar minimap** at 25 fps, and it overlays your model's predictions on the ground truth so you can see where the model goes wrong.

---

## 📁 1. Where to Place Your Data Files

Because raw match videos and tracking streams are large and gated under competition terms, they are **not included in this repository**. You only need to place your local files into the folders below:

```text
more-than-a-hack-2027/
├── data/
│   ├── tactical/
│   │   ├── BAR-ATH_tactical_atd.txt          <-- Place your tracking text file here
│   │   └── BAR-ATH_tactical_atd.xml          <-- (Optional metadata)
│   └── broadcast/
│       └── BAR-ATH_broadcast.mp4             <-- Place your broadcast video here
│                                                 (or place it in the root folder / data/)
├── labeler/                                  <-- Labeler source code & web app
├── run_labeler.bat                           <-- Windows 1-click launcher
└── run_labeler.sh                            <-- macOS / Linux 1-click launcher
```

> **Note**: If your video is stored in another folder or an external drive, you don't even need to move it! Simply click **`📂 Select Video`** in the top bar of the web app to open it directly from your computer with zero upload delay.

---

## 🚀 2. How to Run the Program

### On a Mac (macOS)
1. Open **Terminal** and navigate to the project directory:
   ```bash
   cd path/to/more-than-a-hack-2027
   ```
2. Install the 3 lightweight dependencies:
   ```bash
   pip install -r labeler/requirements.txt
   ```
3. Launch the app:
   ```bash
   chmod +x run_labeler.sh
   ./run_labeler.sh
   ```
   *(Or run directly with Python: `python3 labeler/run_labeler.py`)*

---

### On Windows
1. Simply double-click **`run_labeler.bat`** in the project folder!
   *(The script automatically verifies your Python installation and dependencies, starts the local server, and launches your default browser).*

---

## ⏱️ 3. Broadcast Video Kickoff Synchronization

- **The Sync Mechanism**:
  - In the competition data, tracking frames are indexed to the **tactical camera** (frame 11,185 = first-half kickoff).
  - In the **broadcast video** (`BAR-ATH_broadcast.mp4`), the kickoff whistle blows at `04:45` and the first boot-to-ball contact occurs at **`04:45.12`** (broadcast frame `7,128`).
- **Pre-Configured Offset (`4,013` frames)**:
  - The app comes pre-calibrated with a default offset of **`4,013` frames**.
  - When you jump to **Kickoff (frame 11,185)**, the video automatically lines up with the matching match moment.
  - When the video plays, the minimap tracking dots match player motion on screen.
- **Fine-Tuning Nudge**:
  - You can micro-adjust the alignment anytime using the **`[-1f]`** and **`[+1f]`** buttons, or by pressing **`Alt + ←`** / **`Alt + →`** on your keyboard.

---

## 🎮 4. How to Use the Analyzer

### A. Playback & Scrubbing
- **Spacebar**: Play / Pause both the broadcast video and the 2D pitch radar in exact 25 fps synchronization.
- **`←` / `→`**: Step single frames backward / forward.
- **`Shift + ←` / `→`**: Jump 1 second (25 frames).
- **Timeline**: Click anywhere on the bottom timeline to jump directly to that match moment.

### B. Load a Model and Compare It to the Ground Truth
1. Click **`📥 Load model CSV`** and choose a `frame,label` submission file (199,319 rows), or pick one of the files in `evaluation/results/` from the dropdown. Load several to compare them.
2. The timeline shows the ground truth, then each model, with a thin red row under each model marking where it is wrong.
3. The **Truth vs Predictions** card shows the true label and each model's label at the current frame.
4. The **Model Analysis** card shows the official metrics, where the errors fall (ball missing vs visible, near a handover vs mid-possession), what the model confuses, and a clickable list of its longest wrong stretches. Press **`E`** / **`Shift + E`** to jump to the next / previous error.

> The ground truth only covers the first 5 minutes (frames 11,185-18,684). Use the Analyzer to understand *why* a model fails, not to rank models.

---

## ⌨️ Keyboard Shortcuts

| Key | Action | Description |
| :--- | :--- | :--- |
| **`Spacebar`** | **Play / Pause** | Toggle 25 fps synchronized Video + Radar playback |
| **`←` / `→`** | **Step 1 Frame** | Precision single-frame stepping |
| **`Shift + ←` / `→`** | **Step 1 Second** | Jump 25 frames backward / forward |
| **`Ctrl + ←` / `→`** | **Step 5 Seconds** | Jump 125 frames backward / forward |
| **`E` / `Shift + E`** | **Next / previous error** | Jump to the analysed model's next / previous wrong stretch |
| **`Alt + ←` / `→`** | **Nudge Video Sync** | Micro-adjust video alignment by ±1 frame |
| **`Shift + Alt + ←` / `→`** | **Nudge Video Sync 10f** | Micro-adjust video alignment by ±10 frames |
