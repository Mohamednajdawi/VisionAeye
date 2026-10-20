"""
Real end-to-end streaming test: feeds the actual video1 frame JPEGs
(data/video1_frames/, 41 real images, 0.5s apart) through the full
VisionAeyeStream (live L1 API calls + L2 + arbitration), exactly as a
real camera stream would. This is the final proof the loop asked for:
not a simulation of recorded decisions, but the real pipeline run
against real frames end to end.
"""
import sys, os, base64, glob, json
sys.path.insert(0, os.path.dirname(__file__))
from stream import VisionAeyeStream

FRAME_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "video1_frames")

def main():
    frame_files = sorted(glob.glob(os.path.join(FRAME_DIR, "frame_*.jpg")))
    print(f"Found {len(frame_files)} frames in {FRAME_DIR}")

    stream = VisionAeyeStream()
    raw_seq = []
    stable_seq = []

    for i, path in enumerate(frame_files):
        t = round(i * 0.5, 1)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        result = stream.on_frame(b64, t)
        raw_seq.append(result["l1_raw_decision"])
        stable_seq.append(result["l2_stabilized_decision"])
        actions = [a.kind for a in result["audio_actions"]]
        print(f"t={t:5.1f}s raw={result['l1_raw_decision']:10s} "
              f"stable={result['l2_stabilized_decision']:10s} actions={actions}")

    raw_flips = sum(1 for a, b in zip(raw_seq, raw_seq[1:]) if a != b)
    stable_flips = sum(1 for a, b in zip(stable_seq, stable_seq[1:]) if a != b)
    print(f"\n=== REAL END-TO-END RESULT ===")
    print(f"raw L1 flips:        {raw_flips}")
    print(f"L2-stabilized flips: {stable_flips}")
    print(f"Final stable state:  {stable_seq[-1]}")

    with open(os.path.join(os.path.dirname(__file__), "..", "real_e2e_video1_result.json"), "w") as f:
        json.dump({"raw_seq": raw_seq, "stable_seq": stable_seq,
                    "raw_flips": raw_flips, "stable_flips": stable_flips}, f, indent=2)

if __name__ == "__main__":
    main()
