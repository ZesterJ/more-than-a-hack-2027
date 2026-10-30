# ⚽ Barca Vision Labeler -- Quickstart & User Guide

Welcome to **Barca Vision Labeler**! This is a tactical video and tracking annotation studio designed for the **More Than A Hack 2027** ball possession challenge (FC Barcelona vs Athletic Club).

It provides a synchronized dual-view experience: the **official match broadcast video** runs in lockstep with a **2D tactical radar minimap** at 25 fps, allowing you to quickly label possession spells and export machine-learning-ready datasets.

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
3. Launch the studio:
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

## 🎮 4. How to Use the Studio & Label Data

### A. Playback & Scrubbing
- **Spacebar**: Play / Pause both the broadcast video and the 2D pitch radar in exact 25 fps synchronization.
- **`←` / `→`**: Step single frames backward / forward.
- **`Shift + ←` / `→`**: Jump 1 second (25 frames).
- **Timeline Scrubber**: Click anywhere on the bottom possession ribbon to jump directly to that match moment.

### B. Fast 10x Spell / Range Labeling
In football, possessions average ~14 seconds (350 frames). Labeling frame-by-frame is tedious. Use the video-editor range workflow:
1. Scrub to where possession begins $\to$ Press **`I`** (Mark In-point).
2. Scrub to where possession ends $\to$ Press **`O`** (Mark Out-point).
3. Press **`H`** (Home / Barcelona), **`A`** (Away / Athletic), or **`D`** (Dead Ball / Stoppage).
- All 350+ frames are tagged in a single click with clean continuous boundaries!

### C. AI Pre-Annotation Assist
- The studio computes real-time ball proximity and player velocity decay on the fly.
- Press **`Enter`** to accept the AI suggestion for the active frame or marked range.

### D. Exporting Your Work
- **`📄 Export CSV`**: Exports the official 199,319-row competition submission file (`data/annotated_possession_labels.csv`).
- **`🚀 Export ML Set`**: Compiles all labeled frames and automatically merges them with the 31 engineered tactical features (`data/custom_training_dataset.csv`), ready to feed directly into machine learning models (`train_cnn_architectures.py`).

---

## ⌨️ Complete Keyboard Shortcuts Cheat Sheet

| Key | Action | Description |
| :--- | :--- | :--- |
| **`Spacebar`** | **Play / Pause** | Toggle 25 fps synchronized Video + Radar playback |
| **`←` / `→`** | **Step 1 Frame** | Precision single-frame stepping |
| **`Shift + ←` / `→`** | **Step 1 Second** | Jump 25 frames backward / forward |
| **`Ctrl + ←` / `→`** | **Step 5 Seconds** | Jump 125 frames backward / forward |
| **`H`** | **Label HOME** | Tag active frame or selection as Barcelona (`H`) |
| **`A`** | **Label AWAY** | Tag active frame or selection as Athletic Club (`A`) |
| **`D`** | **Label DEAD BALL** | Tag active frame or selection as Dead Ball / Stoppage (`D`) |
| **`X` / `Delete`** | **Clear Label** | Clear label for active frame or selection |
| **`I` / `[`** | **Mark In-Point** | Set start boundary for range labeling |
| **`O` / `]`** | **Mark Out-Point** | Set end boundary for range labeling |
| **`R`** | **Reset Range** | Clear active In/Out selection |
| **`Enter`** | **Accept AI** | Confirm AI pre-annotation suggestion |
| **`Alt + ←` / `→`** | **Nudge Video Sync** | Micro-adjust video alignment by $\pm 1$ frame |
| **`Shift + Alt + ←` / `→`** | **Nudge Video Sync 10f** | Micro-adjust video alignment by $\pm 10$ frames |
