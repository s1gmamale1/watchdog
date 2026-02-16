from winops import set_dpi_awareness
set_dpi_awareness()

import time
import os
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

import win32gui
import win32con
import pyautogui
import numpy as np
import cv2
import subprocess
import csv
import io
import win32process
import win32ui
from ctypes import windll

from ocr import ocr_log_text
from utils import load_yaml, setup_logger
from window_connector import find_hwnd_by_title_substring
from layout import normalize_window_bottom_right
from pathlib import Path


def exe_dir() -> str:
    import sys
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE = exe_dir()
APP_CFG_PATH = os.path.join(BASE, "config", "app.yaml")
REGIONS_CFG_PATH = os.path.join(BASE, "config", "regions.yaml")


# BULLETPROOF: Ultra-flexible regex handles ALL realistic cases!
# Matches: "HH:MM | message", "HH:MM message", "H:M message", "HH:MMmessage"
ENTRY_RE = re.compile(
    r"(?P<hh>[01]?\d|2[0-3])\s*:\s*(?P<mm>[0-5]?\d)\s*[|\s]?\s*(?P<msg>.+?)(?=(?:\b[01]?\d|2[0-3])\s*:\s*[0-5]?\d|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# Pattern for "warm up"
WARM_WORD_RE = re.compile(r"\bwarm[\s\-]*up\b", re.IGNORECASE)

def normalize_for_match(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("]", " ").replace("[", " ").replace("–", "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_text_for_parsing(text: str) -> str:
    """
    BULLETPROOF: Normalize OCR text before parsing.
    Handles all common OCR misreads of pipe and timestamp characters.
    """
    if not text:
        return ""
    
    # Replace various pipe-like characters with standard pipe
    text = text.replace("｜", "|")  # Full-width pipe
    text = text.replace("¦", "|")   # Broken bar
    text = text.replace("丨", "|")  # CJK vertical line
    
    # IMPROVED: Handle "I" and "l" as pipe in timestamp context
    # Matches patterns like: "08:15I msg", "08:15 I msg", "8:5I msg"
    text = re.sub(r'(\d{1,2}:\d{1,2})\s*I\s+', r'\1 | ', text)  # I after timestamp
    text = re.sub(r'(\d{1,2}:\d{1,2})\s+l\s+', r'\1 | ', text)  # lowercase l
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def latest_msg_is_warm(msg: str) -> bool:
    m = normalize_for_match(msg)
    return bool(WARM_WORD_RE.search(m))

def minutes_since_hhmm(hh: int, mm: int) -> float:
    now = datetime.now()
    last = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if last > now:
        last -= timedelta(days=1)
    return (now - last).total_seconds() / 60.0

def client_origin_screen(hwnd: int) -> Tuple[int, int]:
    return win32gui.ClientToScreen(hwnd, (0, 0))

def capture_window_region_api(hwnd: int, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Capture window region using Windows API.
    
    CRITICAL FIX: Uses GetDC (client area) instead of GetWindowDC (entire window).
    This ensures coordinates are relative to CLIENT area, matching our percentage calculations.
    """
    try:
        # FIXED: Use GetDC (client area) not GetWindowDC (includes title bar)
        hwndDC = win32gui.GetDC(hwnd)  # ← Changed from GetWindowDC
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(saveBitMap)
        
        # BitBlt from client area (x,y are now correct!)
        saveDC.BitBlt((0, 0), (w, h), mfcDC, (x, y), win32con.SRCCOPY)
        
        bmpstr = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype=np.uint8)
        img.shape = (h, w, 4)
        
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    except Exception:
        return None


def capture_logbox_client(hwnd: int, log_region: dict) -> np.ndarray:
    """
    CRITICAL FIX: Properly handles both Windows API and screen capture coordinates.
    
    The log_region dict contains PIXEL coordinates (already calculated from percentage).
    These are CLIENT-relative coordinates (relative to window top-left).
    """
    x = int(log_region["x"])
    y = int(log_region["y"])
    w = int(log_region["w"])
    h = int(log_region["h"])
    
    # CRITICAL FIX: Always focus window before screenshot
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.3)
    except Exception as e:
        print(f"⚠️  Warning: Could not focus window: {e}")
    
    # Try Windows API first
    # capture_window_region_api expects CLIENT coordinates (x, y relative to window)
    img = capture_window_region_api(hwnd, x, y, w, h)
    if img is not None:
        return img
    
    # Fallback: pyautogui requires SCREEN coordinates
    # Convert client coords to screen coords
    cx, cy = client_origin_screen(hwnd)
    screen_left = cx + x
    screen_top = cy + y
    shot = pyautogui.screenshot(region=(screen_left, screen_top, w, h))
    return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)


def scroll_logbox_to_top(hwnd: int, regions: dict, verbose: bool = True) -> bool:
    """
    FIXED: Scroll logbox to TOP (where latest messages appear in FSM Panel).
    
    FSM Panel shows messages in REVERSE order:
    - TOP = Latest messages (newest)
    - BOTTOM = Old messages (oldest)
    
    This is opposite of normal logs!
    
    Double-clicks the scroll bar near TOP to jump there.
    
    Config in regions.yaml:
        log_scroll_point_pct:
            x: 0.95    # Right side (scroll bar location)
            y: 0.08    # TOP area (where latest messages are!)
    
    Args:
        hwnd: Window handle
        regions: Config dict with log_scroll_point_pct
        verbose: Print debug info (default True for visibility)
    
    Returns:
        True if scroll attempted, False if not configured
    """
    scroll_cfg = regions.get("log_scroll_point_pct")
    
    if not scroll_cfg:
        if verbose:
            print("⏭️  Auto-scroll disabled (log_scroll_point_pct not configured)")
        return False
    
    try:
        # Get window size
        cl, ct, cr, cb = win32gui.GetClientRect(hwnd)
        cw = cr - cl
        ch = cb - ct
        cx, cy = client_origin_screen(hwnd)
        
        # Calculate scroll position
        x_pct = float(scroll_cfg.get("x", 0.95))
        y_pct = float(scroll_cfg.get("y", 0.08))  # TOP for FSM Panel!
        
        x = cx + int(cw * x_pct)
        y = cy + int(ch * y_pct)
        
        print(f"📜 Scrolling to TOP (where latest messages are)...")
        print(f"   Position: ({x_pct:.2f}, {y_pct:.2f}) → screen ({x}, {y})")
        
        # Focus window first
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.3)
        
        # VISUAL: Move mouse slowly so you can see it
        print(f"   🖱️  Moving mouse to scroll bar...")
        pyautogui.moveTo(x, y, duration=0.5)  # Slow move (0.5s) for visibility
        time.sleep(0.2)  # Pause so you can see where it is
        
        # Double-click scroll bar to jump to TOP
        print(f"   🖱️  Double-clicking scroll bar...")
        pyautogui.doubleClick()
        
        # Wait for scroll animation
        print(f"   ⏳ Waiting for scroll animation...")
        time.sleep(0.5)  # Longer wait to see scroll happen
        
        print(f"   ✅ Scroll complete!")
        return True
        
    except Exception as e:
        print(f"   ❌ Scroll failed: {e}")
        return False

def run_panel_first_run_if_needed(hwnd: int, regions: dict, log=None, force: bool = False) -> bool:
    """
    Runs first-run onboarding clicks for panel if OCR detects first-run keywords
    OR if you want it unconditional after launch.
    
    Returns:
        True if clicks completed successfully, False if failed
    """
    panel = (regions or {}).get("panel") or {}
    fr = panel.get("first_run") or {}
    clicks = fr.get("clicks") or []
    if not clicks:
        return True  # No clicks needed = success

    # Wait AFTER launch so UI has time to show onboarding
    initial_wait = fr.get("initial_wait_seconds", 10)
    print(f"⏳ Waiting {initial_wait}s for first-run screen...")
    time.sleep(initial_wait)

    # CRITICAL FIX: Force focus with verification
    print("🎯 Forcing window to foreground...")
    for attempt in range(3):
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.5)
            
            # VERIFY focus was gained
            fg = win32gui.GetForegroundWindow()
            if fg == hwnd:
                print("   ✅ Window focused successfully")
                break
            else:
                print(f"   ⚠️  Focus attempt {attempt+1}/3 failed, retrying...")
                if attempt < 2:
                    win32gui.BringWindowToTop(hwnd)
                    time.sleep(0.3)
        except Exception as e:
            if log:
                log.warning("Focus attempt %d failed: %s", attempt+1, e)
            if attempt < 2:
                time.sleep(0.3)
    
    # Final verification
    if win32gui.GetForegroundWindow() != hwnd:
        print("❌ Failed to focus window after 3 attempts")
        if log:
            log.error("Could not focus panel window for first-run clicks")
        return False  # Failed
    
    time.sleep(0.5)  # Extra settle time

    # OPTIONAL (recommended): detect first-run screen by OCR
    detect = fr.get("detect_region")
    keywords = [k.lower() for k in (fr.get("keywords") or [])]

    should_click = force  # If force=True, always click
    
    if not force:
        # Only do OCR detection if not forced
        if detect and keywords:
            # convert pct region -> px region and OCR it
            cl, ct, cr, cb = win32gui.GetClientRect(hwnd)
            cw = cr - cl
            ch = cb - ct
            region_px = {
                "x": int(float(detect["x"]) * cw),
                "y": int(float(detect["y"]) * ch),
                "w": int(float(detect["w"]) * cw),
                "h": int(float(detect["h"]) * ch),
            }
            img = capture_logbox_client(hwnd, region_px)
            txt = (ocr_log_text(img) or "").lower()
            should_click = any(k in txt for k in keywords)

    if not should_click:
        if log: log.info("Panel first-run NOT detected, skipping clicks.")
        print("⏭️  First-run screen not detected, skipping clicks")
        return

    if log: log.info("Panel first-run detected (or forced). Running clicks...")
    print(f"🖱️  Running {len(clicks)} first-run click(s)...")

    # click sequence (pct coords)
    cl, ct, cr, cb = win32gui.GetClientRect(hwnd)
    cw = cr - cl
    ch = cb - ct
    cx, cy = client_origin_screen(hwnd)

    for i, step in enumerate(clicks, 1):
        # CRITICAL FIX: Re-focus if lost, don't abort
        fg = win32gui.GetForegroundWindow()
        if fg != hwnd:
            print(f"   ⚠️  Window lost focus before click {i}, re-focusing...")
            if log:
                log.warning("Panel lost focus before click %d, re-focusing", i)
            
            # Try to regain focus
            try:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                time.sleep(0.5)
                
                # Verify we got it back
                if win32gui.GetForegroundWindow() != hwnd:
                    print(f"   ❌ Failed to regain focus, aborting remaining clicks")
                    if log:
                        log.error("Could not regain focus, aborting first-run clicks")
                    return False  # Failed
                    
                print(f"   ✅ Focus regained, continuing...")
            except Exception as e:
                if log:
                    log.error("Exception regaining focus: %s", e)
                return False  # Failed
    
        x_pct = float(step.get("x_pct", step.get("x")))
        y_pct = float(step.get("y_pct", step.get("y")))
        wait_s = float(step.get("wait_s", step.get("wait", 0.8)))
    
        x = cx + int(cw * x_pct)
        y = cy + int(ch * y_pct)
    
        print(f"   Click {i}/{len(clicks)}: ({x_pct:.4f}, {y_pct:.4f}) -> screen ({x}, {y})")
        
        # ANTI-SPAM: Use jitter to prevent Windows ignoring repeated clicks
        pyautogui.moveRel(1, 0, duration=0)
        pyautogui.moveRel(-1, 0, duration=0)
        pyautogui.moveTo(x, y, duration=0.15)
        pyautogui.click()
        
        # Wait after click
        time.sleep(wait_s)
        
        # ANTI-SPAM: Minimum cooldown to prevent accidental double-clicks
        min_cooldown = 0.3
        if wait_s < min_cooldown:
            time.sleep(min_cooldown - wait_s)
    
    print("✅ First-run clicks complete!")
    return True  # Success!

def is_process_running(image_name: str) -> bool:
    """Windows-only: returns True if a process with this exact image name is running."""
    if not image_name:
        return False
    try:
        out = subprocess.check_output(
            ["tasklist", "/FO", "CSV", "/NH", "/FI", f"IMAGENAME eq {image_name}"],
            text=True,
            errors="ignore"
        )
        # If no tasks match, tasklist outputs: INFO: No tasks are running...
        if "No tasks are running" in out:
            return False
        # Parse CSV rows; first column is image name
        reader = csv.reader(io.StringIO(out))
        for row in reader:
            if row and row[0].strip('"').lower() == image_name.lower():
                return True
        return False
    except Exception:
        return False

def is_panel_running(panel_dir: str) -> bool:
    """
    Check if ANY .exe from the panel directory is currently running.
    This prevents relaunching when the panel is already open.
    """
    if not panel_dir:
        return False
    
    try:
        d = Path(panel_dir)
        if not d.exists():
            return False
        
        # Get all .exe filenames in the directory
        exes = [exe.name for exe in d.glob("*.exe")]
        
        if not exes:
            return False
        
        # Check if any of them are running
        for exe_name in exes:
            if is_process_running(exe_name):
                print(f"   ✅ Panel process already running: {exe_name}")
                return True
        
        return False
        
    except Exception as e:
        print(f"   ⚠️  Error checking panel process: {e}")
        return False


def launch_steam_route_if_configured(regions, log=None):
    """Launch Steam Route if configured and not already running."""
    cfg = (regions or {}).get("steam_route") or {}
    if not cfg.get("launch_with_panel", False):
        return

    proc = cfg.get("process_name")
    exe = cfg.get("exe")
    
    # Check if already running
    if proc and is_process_running(proc):
        print(f"✅ Steam Route already running ({proc}), skipping launch")
        if log: log.info("Steam Route already running (%s). Skipping launch.", proc)
        return
    
    # Validate exe path
    if not exe:
        if log: log.warning("steam_route.launch_with_panel true but steam_route.exe missing")
        print("⚠️  Steam Route exe not configured in regions.yaml")
        return
    
    if not os.path.exists(exe):
        if log: log.warning("Steam Route exe not found: %r", exe)
        print(f"⚠️  Steam Route exe not found: {exe}")
        return

    # Launch it
    try:
        print(f"🚀 Launching Steam Route: {exe}")
        
        # Get working directory for Steam Route
        exe_dir = os.path.dirname(exe)
        
        subprocess.Popen(
            [exe], 
            cwd=exe_dir,
            shell=False
        )
        
        if log: log.info("Launched Steam Route: %s", exe)
        print("✅ Steam Route launched successfully")
        
    except Exception as e:
        if log: log.error("Failed to launch Steam Route: %s", e)
        print(f"❌ Failed to launch Steam Route: {e}")


def find_latest_entry(text: str, debug=False) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[str], Optional[str]]:
    if not text:
        return None, None, None, None, None

    # IMPROVED: Normalize text first
    text = normalize_text_for_parsing(text)

    best_minutes = None
    best_hh = None
    best_mm = None
    best_line = None
    best_msg = None
    
    all_matches = []  # Track all matches for debugging

    for m in ENTRY_RE.finditer(text):
        hh = int(m.group("hh"))
        mm = int(m.group("mm"))
        msg = (m.group("msg") or "").strip()

        # IMPROVED: Skip empty messages
        if len(msg) < 2:
            continue

        mins = minutes_since_hhmm(hh, mm)
        
        # Debug: Track all matches
        if debug:
            all_matches.append((hh, mm, mins, msg[:50]))

        # VALIDATION: Skip timestamps that are too old (> 2 hours = 120 min)
        if mins > 120:
            if debug:
                print(f"   ⚠️  Skipping old timestamp: {hh:02d}:{mm:02d} ({mins:.1f} min ago)")
            continue

        if best_minutes is None or mins < best_minutes:
            best_minutes = mins
            best_hh = hh
            best_mm = mm
            compact_msg = re.sub(r"\s+", " ", msg).strip()
            best_line = f"{hh:02d}:{mm:02d} | {compact_msg}"
            best_msg = msg
    
    # Debug output
    if debug and all_matches:
        print(f"   📋 Found {len(all_matches)} timestamp(s):")
        for hh, mm, mins, msg in all_matches[:5]:  # Show first 5
            marker = "⭐" if (best_hh == hh and best_mm == mm) else "  "
            print(f"   {marker} {hh:02d}:{mm:02d} ({mins:.1f} min ago) - {msg}")

    return best_minutes, best_hh, best_mm, best_line, best_msg


def trigger_recovery_action(hwnd: int, log, app, reason: str):
    print(f"\n🚨 RECOVERY: {reason}")
    log.warning("Triggering recovery action. Reason: %s", reason)

    settle_click_ms = int(app["watchdog"].get("settle_after_click_ms", 2000))

    # Focus the target window
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    try:
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass
    time.sleep(0.3)

    # Get button coords
    regions = load_yaml(REGIONS_CFG_PATH)
    if "button_point_pct" in regions:
        cl, ct, cr, cb = win32gui.GetClientRect(hwnd)
        cw = cr - cl
        ch = cb - ct
        cx, cy = client_origin_screen(hwnd)

        b = regions["button_point_pct"]
        x = cx + int(cw * float(b["x"]))
        y = cy + int(ch * float(b["y"]))
    elif "button_point" in regions:
        b = regions["button_point"]
        cx, cy = client_origin_screen(hwnd)
        x = cx + int(b["x"])
        y = cy + int(b["y"])
    else:
        log.error("No button_point or button_point_pct in config. Cannot click.")
        return

    # Click
    print(f"   Clicking ({x}, {y})")
    pyautogui.moveTo(x, y, duration=0.15)
    pyautogui.click()

    time.sleep(settle_click_ms / 1000)
    print("✅ Recovery complete\n")
    log.info("Recovery click executed at (%d, %d)", x, y)


def resolve_panel_exe(regions: dict) -> Optional[str]:
    """
    IMPROVED: Smart panel exe detection with priority logic
    
    Priority:
    1. If "Panel.exe" exists → use it (most PCs)
    2. If not, find newest .exe in folder (PCs with changing names)
    
    Returns:
        Full path to panel exe, or None if not found
    """
    panel = (regions or {}).get("panel") or {}
    panel_dir = panel.get("dir")
    
    if not panel_dir:
        print("❌ panel.dir not configured in regions.yaml")
        return None

    d = Path(panel_dir)
    if not d.exists():
        print(f"❌ Panel directory does not exist: {panel_dir}")
        return None

    # PRIORITY 1: Check for Panel.exe specifically
    panel_exe_path = d / "Panel.exe"
    if panel_exe_path.exists():
        print(f"✅ Found Panel.exe")
        return str(panel_exe_path)
    
    # PRIORITY 2: Find any .exe files (for PCs with changing names)
    exes = list(d.glob("*.exe"))
    
    if not exes:
        print(f"❌ No .exe files found in: {panel_dir}")
        return None

    # Sort by modification time (newest first)
    exes.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    newest_exe = str(exes[0])
    
    # Show what we found
    print(f"⚠️  Panel.exe not found, using newest .exe:")
    print(f"   📁 Found {len(exes)} .exe file(s) in {panel_dir}")
    for i, exe in enumerate(exes[:3], 1):  # Show top 3
        mtime = datetime.fromtimestamp(exe.stat().st_mtime)
        marker = "⭐" if i == 1 else "  "
        print(f"   {marker} {exe.name} (modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
    if len(exes) > 3:
        print(f"   ... and {len(exes) - 3} more")
    
    print(f"✅ Will launch: {newest_exe}")
    
    return newest_exe


def _query_full_process_image_name(pid: int) -> str:
    """Return full exe path for a PID, or empty string."""
    import ctypes
    try:
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return ""
        buf = ctypes.create_unicode_buffer(2048)
        size = ctypes.c_ulong(len(buf))
        ok = ctypes.windll.kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        ctypes.windll.kernel32.CloseHandle(handle)
        return buf.value if ok else ""
    except Exception:
        return ""


def find_window_by_process_path(dir_substring: str) -> Tuple[Optional[int], Optional[str]]:
    """
    FALLBACK: Find window by process directory.
    Useful if title search fails for some reason.
    
    Args:
        dir_substring: Part of the directory path (e.g., "FSM_PANEL v.2.8.0")
    
    Returns:
        (hwnd, title) or (None, None)
    """
    dir_sub_lower = dir_substring.lower()
    matches = []
    
    def enum_handler(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            exe_path = _query_full_process_image_name(pid)
            if exe_path and dir_sub_lower in exe_path.lower():
                title = win32gui.GetWindowText(hwnd) or ""
                matches.append((hwnd, title))
        except Exception:
            pass
    
    win32gui.EnumWindows(enum_handler, None)
    return matches[0] if matches else (None, None)


def run_watchdog() -> None:
    log = setup_logger()
    app = load_yaml(APP_CFG_PATH)

    title_sub = app["window"]["title_substring"]
    width = int(app["layout"]["width"])
    height = int(app["layout"]["height"])
    margin_right = int(app["layout"].get("margin_right", 0))
    margin_bottom = int(app["layout"].get("margin_bottom", 0))

    poll = int(app["watchdog"]["poll_seconds"])
    warm_timeout = float(app["watchdog"].get("warm_timeout_minutes", 40))
    general_timeout = float(app["watchdog"].get("general_timeout_minutes", 60))
    debounce = int(app["watchdog"].get("action_debounce_seconds", 180))

    debug_print_ocr = bool(app["watchdog"].get("debug_print_ocr", False))
    save_last_log_image = bool(app["watchdog"].get("save_last_log_image", True))
    settle_norm_ms = int(app["watchdog"].get("settle_after_normalize_ms", 150))
    settle_focus_ms = int(app["watchdog"].get("settle_after_focus_ms", 200))

    # Initialize normalize flag based on config
    normalize_every = bool(app["watchdog"].get("normalize_every_loop", True))

    os.makedirs("logs", exist_ok=True)
    hwnd = None
    last_found_title = None
    last_action_ts = 0.0
    last_logged_latest_line = None
    steam_route_launched = False
    first_run_completed_pids = set()  # Track PIDs we've already run first-run on

    print("\n" + "="*70)
    print("🐕 APPLICATION WATCHDOG STARTED")
    print("="*70)
    print(f"Warm timeout: {warm_timeout} min (if msg contains 'warm')")
    print(f"General timeout: {general_timeout} min (otherwise)")
    print(f"Poll interval: {poll} sec")
    print("="*70 + "\n")
    
    log.info("Starting watchdog - Warm: %.1f min, General: %.1f min", warm_timeout, general_timeout)

    # OPTIMIZATION: Load regions once at startup (not every loop)
    regions = load_yaml(REGIONS_CFG_PATH)
    loop_count = 0
    
    while True:
        loop_count += 1
        
        # Check window
        if hwnd is None or not win32gui.IsWindow(hwnd):
            print("🔍 Searching for target window...")
            hwnd, last_found_title = find_hwnd_by_title_substring(title_sub)
            
            if not hwnd:
                print(f"⚠️  Window not found (looking for: '{title_sub}'). Launching panel.exe...")
                log.warning("Window not found. Launching panel.exe")

                try:
                    regions = load_yaml(REGIONS_CFG_PATH)
                    
                    # ENHANCED: Automatically find newest .exe in panel.dir
                    panel_exe = resolve_panel_exe(regions)

                    if not panel_exe:
                        log.error("Panel EXE not found in panel.dir")
                        print("❌ Panel EXE not found. Check regions.yaml -> panel.dir")
                        time.sleep(5)
                        continue
                    
                    log.warning("Launching Panel: %s", panel_exe)
                    print(f"\n🚀 Launching: {panel_exe}")
                    # Get the folder containing the .exe
                    exe_dir = os.path.dirname(panel_exe)
                    
                    print(f"📂 Working directory: {exe_dir}")
                    
                    # Launch with working directory set
                    process = subprocess.Popen(
                        [panel_exe],
                        cwd=exe_dir,  # ← THE CRITICAL FIX!
                        shell=False
                    )
                    
                    print(f"✅ Process PID: {process.pid}")

                    if not steam_route_launched:
                        launch_steam_route_if_configured(regions, log=log)
                        steam_route_launched = True

                except Exception as e:
                    log.exception("Failed to launch panel.exe: %s", e)
                    print(f"❌ Exception during panel launch: {e}")
                    time.sleep(5)
                    continue
                
                # ENHANCED: Progressive retry logic
                print("\n🔄 Waiting for panel window to appear...")
                print(f"   Looking for window title containing: '{title_sub}'")
                print("   Using progressive retry strategy: 3s, 5s, 8s, 12s delays")
                
                retry_delays = [3, 5, 8, 12]
                panel_dir = regions.get("panel", {}).get("dir", "")
                
                for attempt, delay in enumerate(retry_delays, 1):
                    print(f"\n   Attempt {attempt}/{len(retry_delays)}: waiting {delay}s...")
                    time.sleep(delay)
                    
                    # Primary: Try by title
                    hwnd, last_found_title = find_hwnd_by_title_substring(title_sub)
                    
                    if hwnd:
                        print(f"   ✅ Found by title: {last_found_title}")
                        break
                    
                    # Fallback: Try by process path
                    if panel_dir:
                        print(f"   Title search failed, trying by process path...")
                        hwnd, last_found_title = find_window_by_process_path(panel_dir)
                        if hwnd:
                            print(f"   ✅ Found by process path: {last_found_title}")
                            break
                    
                    if attempt < len(retry_delays):
                        print(f"   ⚠️  Window not found yet, will retry...")
                
                if not hwnd:
                    print("\n❌ Panel launched but window still not found after all retries.")
                    print("   Possible reasons:")
                    print("   1. Panel takes longer than 28 seconds to show window")
                    print(f"   2. Window title doesn't contain: '{title_sub}'")
                    print("   3. Panel.exe crashed or failed to start")
                    print(f"\n   💡 TIP: Check app.yaml -> window.title_substring = '{title_sub}'")
                    log.error("Panel window not found after progressive retries")
                    time.sleep(5)
                    continue
                
                print(f"\n✅ Window found after launch: {last_found_title} (hwnd={hwnd})")
                log.info("Window found after launch: hwnd=%s title=%r", hwnd, last_found_title)

                print("\n🔧 Running first-run setup if needed...")
                
                # ANTI-SPAM: Check if we've already run first-run on this panel instance
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    
                    if pid in first_run_completed_pids:
                        print(f"✅ First-run already completed for this panel instance (PID: {pid})")
                        log.info("First-run already completed for PID %d", pid)
                    else:
                        # Run first-run setup
                        success = run_panel_first_run_if_needed(hwnd, regions, log=log, force=True)
                        
                        # Only mark as completed if successful
                        if success:
                            first_run_completed_pids.add(pid)
                            print(f"✅ Marked PID {pid} as first-run completed")
                            log.info("First-run completed for PID %d", pid)
                        else:
                            print(f"⚠️  First-run clicks failed, will retry next time")
                            log.warning("First-run clicks failed for PID %d", pid)
                        
                except Exception as e:
                    log.warning("Could not get PID for first-run tracking: %s", e)
                    # Fall back to running it anyway
                    run_panel_first_run_if_needed(hwnd, regions, log=log, force=True)
                
                # Normalize window position AFTER first-run (always runs)
                print("\n📐 Normalizing panel window position...")
                x, y, moved = normalize_window_bottom_right(
                    hwnd, 
                    width=width, 
                    height=height,
                    margin_right=margin_right, 
                    margin_bottom=margin_bottom,
                )
                if moved:
                    print(f"   ✅ Window moved to bottom-right: ({x}, {y})")
                    log.info("Normalized window after first launch -> %d, %d", x, y)
                else:
                    print(f"   ℹ️  Window already at correct position: ({x}, {y})")
                
                time.sleep(settle_norm_ms / 1000)

            else:
                print(f"✅ Window found: {last_found_title} (hwnd={hwnd})\n")
                log.info("Found window: hwnd=%s title=%r", hwnd, last_found_title)

        # Normalize window (if allowed)
        if normalize_every:
            x, y, moved = normalize_window_bottom_right(
                hwnd, width=width, height=height,
                margin_right=margin_right, margin_bottom=margin_bottom,
            )
            if moved:
                log.info("Normalized window -> %d, %d", x, y)
            time.sleep(settle_norm_ms / 1000)
        
        try:
            # OPTIMIZATION: regions loaded at startup, not reloaded every loop
            # Use: regions (from outer scope)
            
            # NOTE: Window is focused inside capture_logbox_client() before screenshot
            # This ensures accurate OCR even if window was minimized/background

            cl, ct, cr, cb = win32gui.GetClientRect(hwnd)
            client_w = cr - cl
            client_h = cb - ct

            if "log_region_pct" in regions:
                r = regions["log_region_pct"]
                log_region = {
                    "x": int(r["x"] * client_w),
                    "y": int(r["y"] * client_h),
                    "w": int(r["w"] * client_w),
                    "h": int(r["h"] * client_h),
                }
            else:
                log_region = regions.get("log_region")
                if not log_region:
                    raise RuntimeError("Missing log_region in config.")

            # NEW: Scroll to bottom before capturing (if configured)
            scroll_logbox_to_top(hwnd, regions, verbose=True)  # Always show output

            img = capture_logbox_client(hwnd, log_region)
            
            # ALWAYS save images (create logs dir if missing)
            logs_dir = os.path.join(BASE, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            # Save main screenshot
            cv2.imwrite(os.path.join(logs_dir, "last_log.png"), img)
            if debug_print_ocr:
                print(f"💾 Saved: {os.path.join(logs_dir, 'last_log.png')}")

            text = (ocr_log_text(img) or "").strip()

            if debug_print_ocr:
                print(f"📄 OCR: {text[:100]}...")

        except Exception as e:
            print(f"❌ ERROR: {e}\n")
            log.exception("Capture failed")
            time.sleep(poll)
            continue

                # Logic Analysis
        # Logic Analysis (1st pass)
        minutes_ago, hh, mm, latest_line, latest_msg = find_latest_entry(text, debug=True)  # Always debug
        
        if minutes_ago is None:
            # Retry once using screen-capture fallback (still inside the same loop)
            try:
                print("⚠️  Parse failed. Retrying with screen-capture fallback...")
        
                # normalize common OCR pipe variants BEFORE parsing
                def _norm(s: str) -> str:
                    return (s or "").replace("｜", "|").replace("¦", "|").replace("丨", "|")
        
                # screen capture of the same client region
                cx, cy = client_origin_screen(hwnd)
                left = cx + int(log_region["x"])
                top  = cy + int(log_region["y"])
                w    = int(log_region["w"])
                h    = int(log_region["h"])
        
                shot = pyautogui.screenshot(region=(left, top, w, h))
                img2 = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
        
                # ALWAYS save fallback image
                logs_dir = os.path.join(BASE, "logs")
                os.makedirs(logs_dir, exist_ok=True)
                cv2.imwrite(os.path.join(logs_dir, "last_log_fallback.png"), img2)
                if debug_print_ocr:
                    print(f"💾 Saved: {os.path.join(logs_dir, 'last_log_fallback.png')}")
        
                text2 = _norm((ocr_log_text(img2) or "").strip())
        
                if debug_print_ocr:
                    print(f"📄 OCR(fallback): {text2[:100]}...")
        
                minutes_ago, hh, mm, latest_line, latest_msg = find_latest_entry(text2, debug=True)  # Always debug
        
            except Exception as e:
                log.warning("Fallback capture/OCR failed: %s", e)
        
        if minutes_ago is None:
            print("⚠️  No readable 'HH:MM | msg' found.\n")
            log.info("No parseable entry.")
            time.sleep(poll)
            continue
        

        is_warm = latest_msg_is_warm(latest_msg)

        if latest_line != last_logged_latest_line:
            warm_tag = "🔥" if is_warm else "  "
            print(f"{warm_tag} {latest_line} ({minutes_ago:.1f} min)")
            log.info("Log: '%s' (%.1f min ago) warm=%s", latest_line, minutes_ago, is_warm)
            last_logged_latest_line = latest_line

        # === FIXED DUAL LOGIC ===
        now_ts = time.time()
        since_last = now_ts - last_action_ts
        should_trigger = False
        trigger_reason = ""

        # 1. Determine which specific timeout applies
        if is_warm:
            effective_threshold = warm_timeout
            logic_type = "Warm-Up"
        else:
            effective_threshold = general_timeout
            logic_type = "General"

        # 2. Check the timeout
        if minutes_ago >= effective_threshold:
            should_trigger = True
            trigger_reason = (
                f"{logic_type} timeout exceeded "
                f"({minutes_ago:.1f} >= {effective_threshold} min). Msg: '{latest_msg}'"
            )
        if not should_trigger and minutes_ago >= general_timeout:
             should_trigger = True
             trigger_reason = (
                 f"Safety Net (General) timeout exceeded "
                 f"({minutes_ago:.1f} >= {general_timeout} min)."
             )

        if should_trigger:
            if since_last < debounce:
                remaining = debounce - since_last
                print(f"⏸️  Cooldown ({remaining:.0f}s). Reason: {trigger_reason}")
                log.warning("Cooldown active. Reason: %s", trigger_reason)
            else:
                # Disable normalization for next loop to prevent jumping during recovery
                normalize_every = False 
                
                trigger_recovery_action(hwnd, log, app, trigger_reason)
                last_action_ts = now_ts
        else:
            # Re-enable normalization if we are healthy (and config says so)
            normalize_every = bool(app["watchdog"].get("normalize_every_loop", True))

        # PRODUCTION: Check if SteamRoute crashed and restart if needed
        if steam_route_launched:  # Check every loop
            cfg = regions.get("steam_route", {})
            proc = cfg.get("process_name")
            if proc and not is_process_running(proc):
                print("⚠️  SteamRoute process died! Relaunching...")
                log.warning("SteamRoute crashed, relaunching")
                launch_steam_route_if_configured(regions, log=log)
        
        # PRODUCTION: Cleanup PID tracking to prevent memory growth
        if len(first_run_completed_pids) > 20:
            print("🧹 Cleaning up old PID tracking...")
            # Keep only recent PIDs (last 10)
            if len(first_run_completed_pids) > 10:
                first_run_completed_pids = set(list(first_run_completed_pids)[-10:])
        
        time.sleep(poll)

if __name__ == "__main__":
    run_watchdog()