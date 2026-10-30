# ⚽ Barca Vision Labeler -- Tactical & Video Annotation Studio

**Barca Vision Labeler** is a high-performance, browser-based tactical video annotation and tracking studio built specifically for the **More Than A Hack 2027** ball possession challenge (FC Barcelona vs Athletic Club).

It integrates the **full-match broadcast video** side-by-side with a **2D tactical radar minimap**, offering real-time 25 fps synchronization, physics telemetry, AI pre-annotations, and 10x-speed spell labeling hotkeys.

---

## 🚀 3-Minute Setup Guide for Collaborators

If you already have the competition match files (tracking text file and match video), getting started takes 3 simple steps:

### 1. Install Dependencies
Only 3 lightweight Python packages are required (no heavy deep learning libraries needed to run the labeler):
```bash
pip install -r labeler/requirements.txt
```
*(Or manually: `pip install flask numpy pandas`)*

### 2. Verify Your Match Data Location
Ensure your data assets are placed in either of the standard locations:
- **Tracking Data**:
  - `data/tactical/BAR-ATH_tactical_atd.txt` (and `.xml`)
- **Match Video (`.mp4`)**:
  - Place `BAR-ATH_broadcast.mp4` or `BAR-ATH_tactical.mp4` in the project root folder or inside `data/` or `data/tactical/`.
  - *(Alternatively, you can click **"📂 Select Video"** in the web app to load any `.mp4` file from your computer with zero upload delay).*

### 3. Launch the App
- **Windows**: Double-click `run_labeler.bat`
- **Mac / Linux**: Run `./run_labeler.sh` or:
```bash
python labeler/run_labeler.py
```
The studio will automatically:
1. Index all 199,319 tracking frames in ~0.2 seconds.
2. Boot the local server at `http://127.0.0.1:5000`.
3. Open your default web browser and auto-align to First-Half Kickoff!

---

## 📺 Broadcast Video Kickoff Synchronization (04:45.00)

The tracking data and official ground truth are indexed to the **tactical camera** (frame 11,185 = kickoff). The **broadcast feed** (`BAR-ATH_broadcast.mp4`) starts recording earlier.

- **Pre-Configured Offset**: The app comes pre-calibrated with an offset of **`4060` frames** ($162.4$ seconds).
- **Alignment Anchor**:
  - Broadcast video timestamp **`04:45.00`** ($285.0$s) corresponds exactly to Tactical Tracking **`Frame 11,185`**.
  - Broadcast video timestamp **`04:46.00`** ($286.0$s) corresponds exactly to Tactical Tracking **`Frame 11,210`** (ball rolled slightly into center circle).
- **Fine-Tuning**: If needed, you can adjust the offset anytime using the `[-1f]` and `[+1f]` buttons or click `Sync KO (4:45)`.

---

## 🌟 Key Features

1. **Main Cinema Video View (68% stage)**:
   - Clear view of player duels, ball touches, referee whistles, and stoppages.
   - 25 fps hardware-accelerated playback synced to tracking data.
2. **Tactical Radar Minimap (32% sidebar)**:
   - Real-time 2D pitch radar showing FC Barcelona (Blaugrana navy + gold, slots 1–23) and Athletic Club (Crimson + white, slots 24–46).
   - Dynamic velocity arrows and pulsing ball tracker with 12-frame trajectory trail.
   - Live distance lines to nearest Home and Away players with proximity margin ($d_{away} - d_{home}$).
   - Occlusion alert banner when the ball is missing (`NaN`) with centroid tracking fallback.
3. **10x Range / Spell Labeling**:
   - In football, possessions average ~14 seconds (350 frames). Labeling frame-by-frame is unnecessary:
     1. Scrub to possession start $\to$ Press **`I`** (Mark In-point).
     2. Scrub to possession end $\to$ Press **`O`** (Mark Out-point).
     3. Press **`H`** (Home), **`A`** (Away), or **`D`** (Dead Ball).
     - **350 frames labeled in 1 click!**
4. **AI Pre-Annotation Assist**:
   - Live machine learning heuristic calculates possession probabilities in real time.
   - Press **`Enter`** to accept the AI suggestion for the active frame or marked range.
5. **One-Click Export**:
   - **`📄 Export CSV`**: Saves competition submission format (`frame,label`) to `data/annotated_possession_labels.csv`.
   - **`🚀 Export ML Set`**: Compiles all labeled frames merged with the 31 collective features into `data/custom_training_dataset.csv`, ready to train PyTorch / LightGBM models!

---

## ⌨️ Keyboard Shortcuts

| Key | Action | Description |
| :--- | :--- | :--- |
| **`Spacebar`** | **Play / Pause** | Toggles 25 fps synchronized Video + Radar playback |
| **`←` / `→`** | **Step 1 Frame** | Precision frame-by-frame inspection |
| **`Shift + ←` / `→`** | **Step 1 Second** | Jump 25 frames backward / forward |
| **`Ctrl + ←` / `→`** | **Step 5 Seconds** | Jump 125 frames backward / forward |
| **`H`** | **Label HOME** | Tag active frame or selection as Barcelona (`H`) |
| **`A`** | **Label AWAY** | Tag active frame or selection as Athletic Club (`A`) |
| **`D`** | **Label DEAD BALL** | Tag active frame or selection as Stoppage / Out of Play (`D`) |
| **`X` / `Delete`** | **Clear Label** | Clear label for active frame or selection |
| **`I` / `[`** | **Mark In-Point** | Sets start boundary for range labeling |
| **`O` / `]`** | **Mark Out-Point** | Sets end boundary for range labeling |
| **`R`** | **Reset Range** | Clears active In/Out selection |
| **`Enter`** | **Accept AI** | Confirm AI suggestion for current frame/range |
