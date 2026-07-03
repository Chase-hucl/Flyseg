# LICS — Flow Zoometry 🪰🔬

**LICS** is the LabVIEW control software for the **Flow Zoometry** platform — a flow-based imaging system for whole *Drosophila* larvae.

Larvae are transported through a fluidic channel, detected on the fly by a laser / photodetector trigger, scanned with a Hamamatsu camera, and saved as `.dcimg` image stacks. The acquired images feed into the downstream [**FlySeg**](https://github.com/Chase-hucl/Flyseg) segmentation pipeline.

The application is a single-loop **state machine** (`fly dominator.vi`) coordinating the camera, motorized stage, flow pumps, laser, and data acquisition hardware.

---

## ✨ Features

- ✅ Real-time larva detection via laser / photodetector threshold triggering
- ✅ **Edge-hunting**: automatic localization of larva boundaries before a scan
- ✅ Acquisition on a Hamamatsu DCAM camera → `.dcimg` output
- ✅ Synchronized control of stage, flow pumps, laser, and camera through one state machine
- ✅ Automatic per-larva imaging, saving
- ✅ Live particle analysis (centroid, size, Feret diameter) for boundary finding
- ✅ Multi-flow control (sheath / sample / shower / jet)
- ✅ Email notification on experiment information via SMTP
- ✅ Output naming compatible with the FlySeg preprocessing pipeline

---

## 🧰 Requirements

### Software
- **LabVIEW 2021** (project saved as v21.0) — Windows
- **NI-DAQmx** driver
- **NI Vision Development Module** + **NI Vision Acquisition Software** (IMAQ / IMAQdx)
- **Hamamatsu DCAM-API** + the **Hamamatsu Video Capture** LabVIEW library (installed under `user.lib`)
- **Thorlabs Kinesis** (.NET) — `Thorlabs.MotionControl.Benchtop.StepperMotorCLI` (motorized filter wheel)
- **NI-VISA** — serial (RS-232) communication with the motorized stage

### Hardware
- Hamamatsu scientific camera (line-scan / TDI capable, DCAM-API)
- NI multifunction DAQ device (referenced in the project as **`Dev1`**)
- Thorlabs Kinesis benchtop stepper controller (motorized filter wheel)
- Motorized stepper stage (controlled over an RS-232 serial connection, `COM4`)
- Flow pumps driven through the DAQ digital lines (see channel map)
- 405 nm laser + photodetector for the trigger line

---

## 🚀 Getting Started

1. **Install** all drivers above, then confirm hardware names match the project:
   - DAQ device enumerates as `Dev1` in NI MAX
   - Stage RS-232 serial port is `COM4`
   - Camera is visible in the Hamamatsu / IMAQdx utility
2. **Open** `LICS.lvproj` in LabVIEW 2021.
3. **Open and run** the main VI: **`fly dominator.vi`**.
4. On the front panel, set the acquisition, edge-hunting, camera, flow, and save parameters (see below).
5. Start from the **`Chilling`** (idle/ready) state, then drive the workflow with the front-panel controls.

> ⚠️ **Operational rule (enforced by the software):** you must return to the **`Chilling`** state before performing any other operation. *"Please check the state frequently."*

---

## 🕹️ Main VIs

| VI | Role |
|----|------|
| **`fly dominator.vi`** | Main application & GUI. The state machine that orchestrates the full acquisition cycle. |
| **`Abyss-watcher.vi`** | Motion-detection monitor. Watches an ROI (`roix`, `roiy`) in the sample chamber and compares frame-to-frame changes against a threshold to detect whether any sample (larvae) remains in the chamber. |
| **`darksider.vi`** | Per-frame image analysis. Runs particle analysis (centroid, bounding rect, Feret diameter, equivalent ellipse) on `.dcimg` frames to measure larva size and find boundaries. |
| **`name.vi`** | File-naming utility. Builds the save path (including laser channel), checks/creates the target folder, and appends the `.dcimg` extension. |
| **`states.ctl`** | Typedef enum defining the state-machine states. |

---

## 🔌 DAQ Channel Map (`Dev1`)

Digital output line assignments, transcribed from the front-panel labels:

| Line | Label |
|------|-------|
| `port0/line0` | 405 nm laser |
| `port0/line1` | 488 nm laser |
| `port0/line2` | 532 nm laser |
| `port0/line3` | 594 nm laser |
| `port0/line4` | pinch valve 2 |
| `port0/line5` | pinch valve 3 |
| `port0/line6` | galvo system |
| `port0/line7` | pinch valve 1 |
| `port1/line0` | robotic arm (automated sample loading) |
| `port1/line2` | shower valve low |
| `port1/line3` | shower valve high |
| `port1/line4` | shower flow |
| `port1/line5` | sheath flow |
| `port1/line6` | sample flow |
| `port1/line7` | jet flow |

- **Analog in** `Dev1/ai0` — laser detector / trigger photodiode (`laser detector`)
- **Analog out** `Dev1/ao0` — laser output / power control
- **Counters** — camera line/frame triggering and timestamping 


---

## 📂 Output

- Image stacks saved as **`.dcimg`** in the configured **Save directory**
- Filenames built by `name.vi` from `Save base file name` + running index + `laser channel`
- These files are the input to **FlySeg** (after conversion to `.h5`)

---

## 📧 Email Notifications

The application can email a recipient list on key events via an SMTP client.

> Configure the SMTP server, sender account, and recipient list inside `fly dominator.vi`
> (`Set Recipients.vi`). **Do not commit real credentials or personal email addresses** —
> use placeholders such as `smtp.example.com` and `user@example.com`.

---

## 📁 Project Structure

```text
LICS/
├── LICS.lvproj            # LabVIEW 2021 project
├── fly dominator.vi       # Main application / state machine (GUI)
├── Abyss-watcher.vi       # Motion-detection monitor (sample-in-chamber check)
├── darksider.vi           # Per-frame particle analysis (larva sizing)
├── name.vi                # File-naming / folder-creation utility
├── states.ctl             # State-machine enum typedef
├── LICS.aliases           # Target machine alias
└── LICS.lvlps             # Project layout settings
```

Key dependencies (auto-listed under the project's **Dependencies**): Hamamatsu Video Capture
(`tm_*`), NI-DAQmx, NI Vision (IMAQ / IMAQdx), Thorlabs Kinesis DLLs, and the RS-232 stage VIs.

---

## 📬 Contact

For questions about the Flow Zoometry system, please contact hanqing-wang@g.ecc.u-tokyo.ac.jp.
