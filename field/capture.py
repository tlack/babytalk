"""Record field speech on the board: you read a prompt, then the laptop reads the same
prompt in a TTS voice, both captured by the board's mics. Repeat for --rounds prompts,
then move things around and start the next session.

    uv run capture.py --condition desk-quiet
    uv run capture.py --condition desk-quiet --sources terms --no-human --rounds 20
    uv run capture.py --condition desk-quiet --retake --no-human --rounds 20   # redo what went wrong
    uv run capture.py --condition desk-quiet --hard --no-human --rounds 20     # target problem words
    uv run capture.py --condition humvee-idle --noise "6.2L diesel idling" --distance 0.5 --gain 6
    uv run capture.py --condition humvee-driving --noise "diesel, 45 mph, windows up" --gain 4
    uv run capture.py --noise-bed 120 --condition humvee-driving --noise "diesel, 45 mph"

Hands free (TTS only, nothing to press: for recording while driving), from a board that
streams its mic over USB (atomvm/apps/mic_stream, e.g. the LilyGO T-LoRa Pager):
    uv run capture.py --link usb --board lilygo-t-lora-pager --mic es8311 --auto --minutes 45 \
        --condition truck-drive --noise "diesel, open doors, traffic, some music"

Two links to the board:
  mpy  (default) MicroPython (field/capture_board.py): commands over the USB console, audio
       over WiFi TCP (stt/main/wifi_secrets.h). Stereo: the ES7210's two mics.
  usb  the board streams its mic over the USB console nonstop (field/usb_stream.py); the
       laptop cuts each clip out of the stream and finds where its playback starts in it by
       cross-correlation. Mono. The whole stream is also saved (stream.wav).

--auto: no prompts. Records the noise floor, then TTS clips until --rounds or --minutes (or
Ctrl-C), with --bed-secs of background only every --bed-every clips (for noise augmentation).

Each session starts with a few seconds of silence (the room's noise floor). --noise-bed
records only background noise (no speech) for training-time augmentation.

Output (data/, git-ignored):
    data/field/sessions/<session>/NN-human.wav, NN-tts-<voice>.wav, noise.wav, session.json
    data/field/manifest.jsonl        one line per speech clip (text, split, levels, ...)
    data/field/noise.jsonl           one line per noise-only clip
WAVs are 16 kHz int16: stereo from the MicroPython link (the board's two mics; channel 0 =
the mic stt uses), mono from the usb link.
"""
import argparse
import datetime
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from mpremote.transport_serial import SerialTransport

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "datagen"))
from tts import Voices  # noqa: E402
from usb_stream import align  # noqa: E402

DATA = ROOT / "data/field"
PROMPTS = DATA / "prompts.jsonl"
MANIFEST = DATA / "manifest.jsonl"
NOISE = DATA / "noise.jsonl"
BOARD_FILES = {
    "capture_board.py": HERE / "capture_board.py",
    "io_ext.py": ROOT / "mpy/drivers/io_ext.py",
    "es7210.py": ROOT / "mpy/drivers/es7210.py",
}
RATE = 16000
BUCKET_ORDER = [1, 4, 8, 12, 20]
SOURCES = ["librispeech", "news", "wikipedia", "terms", "hardwords", "hackernews"]
HARD = DATA / "hard_words.json"
BOLD, DIM, RED, GREEN, YEL, RESET = "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"


# ---------------------------------------------------------------- board link
def wifi_creds():
    """From the git-ignored stt/main/wifi_secrets.h: passed to the board at run time only."""
    txt = (ROOT / "stt/main/wifi_secrets.h").read_text()
    ssid = re.search(r'WIFI_SSID\s+"([^"]*)"', txt).group(1)
    pw = re.search(r'WIFI_PASS\s+"([^"]*)"', txt).group(1)
    return ssid, pw


class Board:
    """Commands over the USB console (raw REPL); audio over a WiFi TCP socket."""

    def __init__(self, port, gain, tcp_port=5556):
        import socket
        self.t = SerialTransport(port, baudrate=115200)
        self.t.enter_raw_repl()
        for dest, src in BOARD_FILES.items():
            data = src.read_bytes()
            try:
                same = self.t.fs_hashfile(dest, "sha256") == hashlib.sha256(data).digest()
            except Exception:
                same = False
            if not same:
                self.t.fs_writefile(dest, data)
        self.t.exec(f"import capture_board; capture_board.setup({int(gain)})")
        ssid, pw = wifi_creds()
        out = self.t.exec(f"capture_board.net({ssid!r}, {pw!r}, {tcp_port})").decode()
        m = re.search(r"LISTEN (\S+) (\d+)", out)
        if not m:
            raise SystemExit(f"board did not come up on WiFi: {out.strip()}")
        self.ip = m.group(1)
        self.t.exec_raw_no_follow("capture_board.accept()")
        self.sock = socket.create_connection((self.ip, tcp_port), timeout=30)
        self.t.follow(10)

    def record(self, secs, on_start=None):
        """-> int16 array [n, 2]. on_start() runs when the board starts capturing."""
        state = {"buf": "", "pcm": b""}

        def consume(data):
            state["buf"] += data.decode(errors="replace")
            while "\n" in state["buf"]:
                line, state["buf"] = state["buf"].split("\n", 1)
                line = line.strip("\x04\r ")
                if line.startswith("REC"):
                    if on_start:
                        on_start()
                elif line.startswith("DATA"):
                    n = int(line.split()[1])
                    chunks, got = [], 0
                    while got < n:
                        c = self.sock.recv(min(65536, n - got))
                        if not c:
                            raise ConnectionError("board closed the audio socket")
                        chunks.append(c)
                        got += len(c)
                    state["pcm"] = b"".join(chunks)

        _, err = self.t.exec_raw(f"capture_board.record({secs:.2f})", timeout=secs + 30, data_consumer=consume)
        if err:
            raise RuntimeError("board: " + err.decode(errors="replace"))
        return np.frombuffer(state["pcm"], "<i2").reshape(-1, 2).copy()

    def close(self):
        try:
            self.sock.close()
            self.t.exit_raw_repl()
        finally:
            self.t.close()


# ---------------------------------------------------------------- laptop playback (WSL -> Windows)
def windows_temp():
    out = subprocess.check_output(["cmd.exe", "/c", "echo %TEMP%"], text=True, stderr=subprocess.DEVNULL).strip()
    return Path(subprocess.check_output(["wslpath", "-u", out], text=True).strip())


def stage(pcm, name):
    path = windows_temp() / name
    sf.write(path, pcm, RATE, subtype="PCM_16")
    return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()


def play_async(win_path):
    return subprocess.Popen(["powershell.exe", "-NoProfile", "-Command",
                             f"(New-Object Media.SoundPlayer '{win_path}').PlaySync()"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------- levels
def dbfs(x):
    x = x.astype(np.float64)
    return 20 * np.log10(max(1e-9, np.sqrt(np.mean(x * x))) / 32768)


def levels(pcm, noise_db):
    ch = pcm[:, 0].astype(np.float64)
    frames = ch[: len(ch) // 320 * 320].reshape(-1, 320)            # 20 ms frames
    fdb = 20 * np.log10(np.maximum(1e-9, np.sqrt((frames ** 2).mean(1))) / 32768)
    speech_db = float(np.mean(np.sort(fdb)[-max(1, len(fdb) // 5):]))   # loudest 20% of frames
    return {"speech_dbfs": round(speech_db, 1), "peak_dbfs": round(20 * np.log10(max(1, np.abs(ch).max()) / 32768), 1),
            "clipped_frac": round(float(np.mean(np.abs(ch) >= 32700)), 5),
            "snr_db": round(speech_db - noise_db, 1) if noise_db is not None else None}


def show_levels(lv):
    warn = ""
    if lv["clipped_frac"] > 0.001:
        warn = f"  {RED}CLIPPING ({lv['clipped_frac']:.1%}): lower --gain{RESET}"
    elif lv["speech_dbfs"] < -45:
        warn = f"  {YEL}quiet: raise --gain or move closer{RESET}"
    snr = f", SNR {lv['snr_db']} dB" if lv["snr_db"] is not None else ""
    print(f"{DIM}   level {lv['speech_dbfs']} dBFS, peak {lv['peak_dbfs']}{snr}{RESET}{warn}")


# ---------------------------------------------------------------- prompts
def load_prompts():
    if not PROMPTS.exists():
        sys.exit("no prompt pool: run `uv run prompts.py` first")
    return [json.loads(l) for l in open(PROMPTS)]


def shingles(ref, n=6):
    w = ref.split()
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def recorded_shingles():
    """6-word stretches of every recorded prompt: a prompt sharing one is a near-repeat."""
    if not MANIFEST.exists():
        return set()
    out = set()
    for l in open(MANIFEST):
        out |= shingles(json.loads(l)["ref"])
    return out


def fresh_only(prompts):
    """Drop prompts already recorded or overlapping recorded text (same article sentences)."""
    done, seen = recorded_ids(), recorded_shingles()
    return [p for p in prompts if p["id"] not in done and not (shingles(p["ref"]) & seen)]


def recorded_ids():
    if not MANIFEST.exists():
        return set()
    return {json.loads(l)["prompt_id"] for l in open(MANIFEST)}


def picker(prompts, seed, sources=None):
    """Staggered lengths (1/4/8/12/20 s), rotating sources, prompts not recorded yet first."""
    import random
    rng = random.Random(seed)
    srcs = [s for s in SOURCES if not sources or s in sources]
    prompts = [p for p in prompts if p["source"] in srcs]
    if not prompts:
        sys.exit(f"no prompts from sources {sources}")
    fresh = fresh_only(prompts) or prompts
    i = 0
    while True:
        b = BUCKET_ORDER[i % len(BUCKET_ORDER)]
        src = srcs[(i // len(BUCKET_ORDER)) % len(srcs)]
        pool = [p for p in fresh if p["bucket"] == b and p["source"] == src] or \
               [p for p in fresh if p["bucket"] == b] or fresh
        p = rng.choice(pool)
        sh = shingles(p["ref"])                           # no overlapping text within a session
        fresh = [q for q in fresh if q["id"] != p["id"] and not (shingles(q["ref"]) & sh)] or prompts
        i += 1
        yield p


def hard_list(top=80):
    """Vocabulary problem words (the float model missed them), from export/hard_words.py."""
    if not HARD.exists():
        sys.exit("no hard-word list: run `cd export && uv run field_eval.py --split all && uv run hard_words.py`")
    h = json.loads(HARD.read_text())
    # missed by the float model at least half the times it was said: real trouble, not noise
    ws = [d for d in h["words"] if d["kind"] == "vocab" and len(d["word"]) > 2
          and d.get("miss_float", 0) / max(1, d["seen"]) >= 0.5]
    return h, [d["word"] for d in ws][:top]


def retake_picker(prompts):
    """Prompts whose clips the float model got wrong, most errors first."""
    h, _ = hard_list()
    by_id = {p["id"]: p for p in prompts}
    errs = {}
    for l in open(MANIFEST):
        r = json.loads(l)
        e = h["clip_float_errors"].get(r["id"], 0)
        if e and not r.get("exclude") and r["prompt_id"] in by_id:
            errs[r["prompt_id"]] = max(errs.get(r["prompt_id"], 0), e)
    order = sorted(errs, key=lambda k: -errs[k])
    if not order:
        sys.exit("no prompts with errors to retake")
    print(f"{DIM}{len(order)} prompts had errors; retaking the worst first{RESET}")
    while True:
        for k in order:
            yield by_id[k]


def hard_picker(prompts):
    """Prompts containing problem words (not recorded yet first), most problem words first."""
    _, words = hard_list()
    ws = set(words)
    fresh = {p["id"] for p in fresh_only(prompts)}
    scored = [(len(ws & set(p["ref"].split())), p["id"] in fresh, p) for p in prompts]
    scored = [x for x in scored if x[0]]
    # unrecorded first, then the most problem words per spoken word (short, targeted sentences)
    scored.sort(key=lambda x: (not x[1], -x[0] / (5 + x[2]["words"])))
    if not scored:
        sys.exit("no prompts contain problem words: try `uv run prompts.py --hard`")
    print(f"{DIM}{len(scored)} prompts contain problem words ({sum(x[1] for x in scored)} new){RESET}")
    used = set()
    while True:
        for _, _, p in scored:
            sh = shingles(p["ref"])
            if sh & used:                                   # overlaps something this session
                continue
            used |= sh
            yield p
        used = set()


def est_secs(p):
    return min(25.0, p["words"] / 2.3 + 2.5)


# ---------------------------------------------------------------- session
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", required=True, help="short label: desk-quiet, tv-news, humvee-idle, ...")
    ap.add_argument("--noise", default="", help="what's making noise, in words")
    ap.add_argument("--distance", type=float, default=None, help="you -> board, meters")
    ap.add_argument("--speaker-distance", type=float, default=None, help="laptop speaker -> board, meters")
    ap.add_argument("--location", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--rounds", type=int, default=None, help="prompts to record (default 5; with --auto, no limit)")
    ap.add_argument("--gain", type=int, default=10, help="ES7210 mic gain 0-14 (~3 dB/step)")
    ap.add_argument("--noise-secs", type=float, default=6.0, help="noise-floor capture at session start")
    ap.add_argument("--noise-bed", type=float, default=0, help="record only background noise for this many seconds")
    ap.add_argument("--no-tts", action="store_true", help="only your voice")
    ap.add_argument("--no-human", action="store_true", help="only the laptop voices")
    ap.add_argument("--sources", nargs="*", choices=SOURCES, help="only prompts from these sources (e.g. terms)")
    ap.add_argument("--retake", action="store_true", help="re-record prompts the model got wrong (new voice/speed)")
    ap.add_argument("--hard", action="store_true", help="prompts containing words the model gets wrong")
    ap.add_argument("--test-only", action="store_true",
                    help="only test-split prompts: clips that are never trained on, to judge a fine-tuned "
                         "model (e.g. your own voice in a new place)")
    ap.add_argument("--include-test", action="store_true",
                    help="let --retake/--hard use test prompts too (default: train only, so the test set "
                         "doesn't drift toward the hardest material and scores stay comparable over time)")
    ap.add_argument("--tts-speed", default="0.75,1.05",
                    help="laptop voice speed, random per clip from this range (1 = Piper's normal, lower = slower)")
    ap.add_argument("--board", default="waveshare-s3cam")
    ap.add_argument("--mic", default="es7210")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--link", choices=["mpy", "usb"], default="mpy",
                    help="mpy: MicroPython board, audio over WiFi; usb: mic streamed over USB (mic_stream)")
    ap.add_argument("--auto", action="store_true", help="hands free: TTS only, no prompts")
    ap.add_argument("--minutes", type=float, default=None, help="stop after this long (with --auto)")
    ap.add_argument("--bed-every", type=int, default=10, help="--auto: background-only recording every N clips (0: never)")
    ap.add_argument("--bed-secs", type=float, default=8.0)
    a = ap.parse_args()
    if a.auto:
        a.no_human, a.no_tts = True, False
    if a.rounds is None:
        a.rounds = 10000 if a.auto else 5            # --auto: until --minutes or Ctrl-C
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sid = f"{stamp}-{re.sub(r'[^a-z0-9]+', '-', a.condition.lower()).strip('-')}"
    sdir = DATA / "sessions" / sid
    sdir.mkdir(parents=True, exist_ok=True)
    meta = {"session": sid, "condition": a.condition, "noise": a.noise, "distance_m": a.distance,
            "speaker_distance_m": a.speaker_distance, "location": a.location, "notes": a.notes,
            "board": a.board, "mic": a.mic, "gain": a.gain if a.link == "mpy" else None,
            "link": a.link, "started": stamp}
    voices = Voices()
    specs = voices.specs(n_multi=30)   # before the board: loading every voice takes seconds and can stall USB
    print(f"{BOLD}session {sid}{RESET}  {DIM}connecting to the board...{RESET}")
    if a.link == "usb":
        from usb_stream import UsbStreamBoard
        board = UsbStreamBoard(a.port, stream_wav=sdir / "stream.wav",
                               log=lambda m: print(f"{DIM}   board: {m}{RESET}"))
    else:
        board = Board(a.port, a.gain)
    n = 0
    try:
        if a.noise_bed:
            record_noise_bed(board, a, meta, sdir)
            return
        if not a.auto:
            input(f"\n{BOLD}Noise floor:{RESET} stay quiet for {a.noise_secs:g} s (leave the noise source running). Enter to start ")
        else:
            print(f"{BOLD}Noise floor:{RESET} {a.noise_secs:g} s")
        nz = board.record(a.noise_secs)
        sf.write(sdir / "noise.wav", nz, RATE, subtype="PCM_16")
        meta["noise_dbfs"] = round(dbfs(nz[:, 0]), 1)
        print(f"{DIM}   noise floor {meta['noise_dbfs']} dBFS{RESET}")
        append(NOISE, {**meta, "file": str((sdir / "noise.wav").relative_to(DATA)), "secs": a.noise_secs,
                       "kind": "floor"})

        vi = int(hashlib.sha1(sid.encode()).hexdigest(), 16) % len(specs)
        targeted = [p for p in load_prompts() if a.include_test or p["split"] == "train"]
        if a.retake:
            pick = retake_picker(targeted)
        elif a.hard:
            pick = hard_picker(targeted)
        else:
            pool = load_prompts()
            if a.test_only:
                pool = [p for p in pool if p["split"] == "test"]
            pick = picker(pool, seed=sid, sources=a.sources)
        import random
        srng = random.Random(sid + "speed")
        lo, hi = (float(x) for x in a.tts_speed.split(","))
        t_start, beds = time.monotonic(), 0
        for r in range(a.rounds):
            if a.minutes is not None and time.monotonic() - t_start > a.minutes * 60:
                print(f"{DIM}{a.minutes:g} minutes: done{RESET}")
                break
            if a.auto and a.bed_every and r and r % a.bed_every == 0:
                beds += 1
                record_bed(board, a, meta, sdir, beds, a.bed_secs)
            p = next(pick)
            print(f"\n{BOLD}[{r + 1}/{a.rounds}]{RESET} {DIM}{p['source']}, ~{p['bucket']} s, {p['split']}{RESET}")
            print(f"\n    {BOLD}{p['display']}{RESET}\n")
            if not a.no_human:
                while True:
                    k = input(f"  Enter, then read it aloud ({est_secs(p):.0f} s)   [s = skip this prompt] ").strip()
                    if k == "s":
                        break
                    pcm = board.record(est_secs(p), on_start=lambda: print(f"  {RED}● recording{RESET}", flush=True))
                    lv = levels(pcm, meta["noise_dbfs"])
                    show_levels(lv)
                    if input("  Enter = keep, r = redo ").strip() != "r":
                        n += 1
                        save(sdir, n, "human", pcm, p, meta, lv)
                        break
                if k == "s":
                    continue
            if not a.no_tts:
                spec = specs[vi % len(specs)]
                vi += 1
                voice = spec[0].replace("en_US-", "").replace("en_GB-", "").replace("-medium", "") + \
                        (f"-{spec[1]}" if spec[1] is not None else "")
                speed = round(srng.uniform(lo, hi), 2)
                tts = voices.say(p["display"], spec, speed=speed)
                win = stage(tts, "mm_field_tts.wav")
                print(f"  {DIM}laptop speaks ({voice}, speed {speed}, {len(tts) / RATE:.1f} s)...{RESET}")
                procs = []
                pcm = board.record(len(tts) / RATE + 1.5, on_start=lambda: procs.append(play_async(win)))
                for pr in procs:
                    pr.wait()
                extra = {}
                if a.link == "usb":                 # cut the clip to where the playback is
                    st, en, score = align(pcm[:, 0], tts)
                    extra = {"align_score": score, "lost_samples": int(board.last_lost),
                             "stream_at": int(board.last_at)}
                    if score >= 4:
                        pcm = pcm[st:en]
                        extra["stream_at"] += st
                        extra["lost_samples"] = int(board.span(extra["stream_at"], extra["stream_at"] + len(pcm))[1])
                    else:
                        print(f"  {YEL}couldn't find the playback in the recording (score {score}): kept whole{RESET}")
                    if extra["lost_samples"]:
                        print(f"  {YEL}{extra['lost_samples'] / RATE:.2f} s of this clip lost on the USB link{RESET}")
                lv = levels(pcm, meta["noise_dbfs"])
                show_levels(lv)
                n += 1
                save(sdir, n, f"tts-{voice}", pcm, p, meta, lv, tts_speed=speed, **extra)
    except KeyboardInterrupt:
        print(f"\n{DIM}stopped{RESET}")
    finally:
        meta["clips"] = n
        (sdir / "session.json").write_text(json.dumps(meta, indent=2))
        board.close()
    print(f"\n{GREEN}saved {n} clips -> {sdir.relative_to(ROOT)}{RESET}")
    print(f"{DIM}score it: cd export && uv run field_eval.py --session {sid}{RESET}")


def save(sdir, n, speaker, pcm, p, meta, lv, **extra):
    name = f"{n:02d}-{speaker}.wav"
    sf.write(sdir / name, pcm, RATE, subtype="PCM_16")
    append(MANIFEST, {"id": f"{meta['session']}/{n:02d}", "file": str((sdir / name).relative_to(DATA)),
                      "speaker": speaker, "prompt_id": p["id"], "ref": p["ref"], "display": p["display"],
                      "source": p["source"], "bucket": p["bucket"], "split": p["split"],
                      **({"loose": True} if p.get("loose") else {}),
                      "secs": round(len(pcm) / RATE, 2), **lv, **extra,
                      **{k: meta[k] for k in ("session", "condition", "noise", "distance_m", "speaker_distance_m",
                                              "board", "mic", "gain", "noise_dbfs")}})


def record_bed(board, a, meta, sdir, i, secs):
    """A few seconds of background only, between clips (--auto)."""
    pcm = board.record(secs)
    name = f"bed-{i:03d}.wav"
    sf.write(sdir / name, pcm, RATE, subtype="PCM_16")
    db = round(dbfs(pcm[:, 0]), 1)
    append(NOISE, {**meta, "file": str((sdir / name).relative_to(DATA)), "secs": secs, "kind": "bed", "dbfs": db})
    print(f"{DIM}   background only, {secs:g} s: {db} dBFS{RESET}")


def record_noise_bed(board, a, meta, sdir):
    total, i = 0.0, 0
    print(f"{BOLD}Noise bed:{RESET} {a.noise_bed:g} s of '{a.noise or a.condition}', in 20 s pieces. Ctrl-C stops early.")
    try:
        while total < a.noise_bed:
            secs = min(20.0, a.noise_bed - total)
            pcm = board.record(secs)
            i += 1
            name = f"bed-{i:03d}.wav"
            sf.write(sdir / name, pcm, RATE, subtype="PCM_16")
            db = round(dbfs(pcm[:, 0]), 1)
            append(NOISE, {**meta, "file": str((sdir / name).relative_to(DATA)), "secs": secs, "kind": "bed",
                           "dbfs": db})
            total += secs
            print(f"  {i}: {secs:g} s, {db} dBFS  ({total:.0f}/{a.noise_bed:g} s)")
    except KeyboardInterrupt:
        print("stopped")
    (sdir / "session.json").write_text(json.dumps({**meta, "noise_bed_secs": total}, indent=2))


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
