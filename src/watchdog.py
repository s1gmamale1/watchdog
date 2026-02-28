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
import psutil
import win32console
import win32process
import win32ui
from ctypes import windll, byref, wintypes

from ocr import ocr_log_text
from utils import load_yaml, setup_logger
from window_connector import find_hwnd_by_title_substring
from layout import normalize_window_bottom_right
from winops import _query_full_process_image_name
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
    
    # CRITICAL FIX: OCR reads colon as period
    text = re.sub(r'\b(\d{1,2})\.(\d{1,2})\b', r'\1:\2', text)
    
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
        
        print(f"📜 Scrolling to top ({x}, {y})...")

        # Focus window first
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.3)

        pyautogui.moveTo(x, y, duration=0.5)
        time.sleep(0.2)
        pyautogui.doubleClick()
        time.sleep(0.5)

        print(f"   ✅ Scrolled")
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
    print(f"⏳ Waiting {initial_wait}s for first-run UI...")
    time.sleep(initial_wait)

    # CRITICAL FIX: Force focus with verification
    print("🎯 Focusing window...")
    for attempt in range(3):
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.5)

            # VERIFY focus was gained
            fg = win32gui.GetForegroundWindow()
            if fg == hwnd:
                print("   ✅ Focused")
                break
            else:
                print(f"   ⚠️  Focus {attempt+1}/3 failed, retrying...")
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
        print("❌ Focus failed after 3 attempts")
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
        print("⏭️  First-run not detected, skipping")
        return

    if log: log.info("Panel first-run detected (or forced). Running clicks...")
    print(f"🖱️  Running {len(clicks)} first-run click(s)")

    # click sequence (pct coords)
    cl, ct, cr, cb = win32gui.GetClientRect(hwnd)
    cw = cr - cl
    ch = cb - ct
    cx, cy = client_origin_screen(hwnd)

    for i, step in enumerate(clicks, 1):
        # CRITICAL FIX: Re-focus if lost, don't abort
        fg = win32gui.GetForegroundWindow()
        if fg != hwnd:
            print(f"   ⚠️  Focus lost before click {i}, re-focusing...")
            if log:
                log.warning("Panel lost focus before click %d, re-focusing", i)

            # Try to regain focus
            try:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                time.sleep(0.5)

                # Verify we got it back
                if win32gui.GetForegroundWindow() != hwnd:
                    print(f"   ❌ Could not regain focus, aborting")
                    if log:
                        log.error("Could not regain focus, aborting first-run clicks")
                    return False  # Failed

                print(f"   ✅ Focus regained")
            except Exception as e:
                if log:
                    log.error("Exception regaining focus: %s", e)
                return False  # Failed
    
        x_pct = float(step.get("x_pct", step.get("x")))
        y_pct = float(step.get("y_pct", step.get("y")))
        wait_s = float(step.get("wait_s", step.get("wait", 0.8)))
    
        x = cx + int(cw * x_pct)
        y = cy + int(ch * y_pct)
    
        print(f"   Click {i}/{len(clicks)} → screen ({x}, {y})")
        
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
    
    print("✅ First-run clicks done")
    return True  # Success!

def is_process_running(image_name: str) -> bool:
    """Returns True if a process with this exact image name is running."""
    if not image_name:
        return False
    name_lower = image_name.lower()
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'] and proc.info['name'].lower() == name_lower:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
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
        
        # Check Panel.exe specifically first — avoids false positives from
        # unrelated exes (installers, updaters) that share the same directory.
        panel_exe_path = d / "Panel.exe"
        if panel_exe_path.exists():
            if is_process_running("Panel.exe"):
                print(f"   ✅ Panel.exe running")
                return True
            return False

        # Panel.exe not present — fall back to checking any exe in the directory
        exes = [exe.name for exe in d.glob("*.exe")]
        if not exes:
            return False

        for exe_name in exes:
            if is_process_running(exe_name):
                print(f"   ✅ {exe_name} running")
                return True

        return False

    except Exception as e:
        print(f"   ⚠️  Panel check error: {e}")
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
        print(f"✅ SteamRoute already running ({proc})")
        if log: log.info("Steam Route already running (%s). Skipping launch.", proc)
        return

    # Validate exe path
    if not exe:
        if log: log.warning("steam_route.launch_with_panel true but steam_route.exe missing")
        print("⚠️  SteamRoute exe not configured")
        return

    if not os.path.exists(exe):
        if log: log.warning("Steam Route exe not found: %r", exe)
        print(f"⚠️  SteamRoute exe not found: {exe}")
        return

    # Launch it
    try:
        print(f"🚀 Launching SteamRoute: {exe}")

        exe_dir = os.path.dirname(exe)

        subprocess.Popen(
            [exe],
            cwd=exe_dir,
            shell=False
        )

        if log: log.info("Launched Steam Route: %s", exe)
        print("✅ SteamRoute launched")

    except Exception as e:
        if log: log.error("Failed to launch Steam Route: %s", e)
        print(f"❌ SteamRoute launch failed: {e}")


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
                print(f"   skip {hh:02d}:{mm:02d} ({mins:.0f} min, too old)")
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
        print(f"   📋 {len(all_matches)} match(es):")
        for hh, mm, mins, msg in all_matches[:5]:  # Show first 5
            marker = "⭐" if (best_hh == hh and best_mm == mm) else "  "
            print(f"   {marker} {hh:02d}:{mm:02d} {mins:.1f}m  {msg}")

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
        return False

    # Click
    print(f"   Clicking ({x}, {y})")
    pyautogui.moveTo(x, y, duration=0.15)
    pyautogui.click()

    time.sleep(settle_click_ms / 1000)
    print("✅ Recovery done\n")
    log.info("Recovery click at (%d, %d)", x, y)
    return True


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
        print("❌ panel.dir not set in regions.yaml")
        return None

    d = Path(panel_dir)
    if not d.exists():
        print(f"❌ Panel dir not found: {panel_dir}")
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

    print(f"⚠️  Panel.exe not found — using newest .exe:")
    print(f"   {len(exes)} .exe(s) in {panel_dir}:")
    for i, exe in enumerate(exes[:3], 1):  # Show top 3
        mtime = datetime.fromtimestamp(exe.stat().st_mtime)
        marker = "⭐" if i == 1 else "  "
        print(f"   {marker} {exe.name}  ({mtime.strftime('%Y-%m-%d %H:%M')})")
    if len(exes) > 3:
        print(f"   ...+{len(exes) - 3} more")

    print(f"✅ Will launch: {newest_exe}")
    
    return newest_exe


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
    if not matches:
        return (None, None)
    if len(matches) > 1:
        print(f"   ⚠️  {len(matches)} windows match '{dir_substring}'; using '{matches[0][1]}'")
    return matches[0]




def restart_explorer(log=None):
    """Restart Windows Explorer when 'Cannot add' is detected"""
    print("🔄 Restarting Explorer...")
    if log:
        log.warning("Restarting explorer.exe - 'Cannot add' detected")
    
    try:
        subprocess.run(
            ["taskkill", "/F", "/IM", "explorer.exe"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10
        )
        time.sleep(1)
        
        subprocess.Popen(["explorer.exe"], shell=False)
        time.sleep(2)
        
        print("✅ Explorer restarted")
        if log:
            log.info("Explorer restarted")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        if log:
            log.error("Explorer restart failed: %s", e)


def count_cs2_instances():
    """Count how many CS2 (cs2.exe) instances are running"""
    count = 0
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'] and proc.info['name'].lower() == 'cs2.exe':
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return count


def check_cs2_instance_count(hwnd, regions, expected=4, log=None):
    """
    Check if exactly 4 CS2 instances running.
    If not, click kill_all_cs2 button and re-run first-run clicks.
    """
    count = count_cs2_instances()
    print(f"🎮 CS2: {count}/{expected}")
    
    if count == expected:
        print(f"   ✅ OK")
        if log:
            log.info("CS2 count correct: %d", count)
        return True

    print(f"   ❌ Expected {expected}, got {count}")
    if log:
        log.warning("CS2 count wrong: %d (expected %d)", count, expected)

    print("   🔧 Clicking kill_all_cs2...")
    kill_button = regions.get("kill_all_cs2_point_pct")
    if not kill_button:
        print("   ❌ kill_all_cs2_point_pct not configured")
        if log:
            log.error("kill_all_cs2_point_pct not configured")
        return False
    
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.5)
        
        cl, ct, cr, cb = win32gui.GetClientRect(hwnd)
        cx, cy = client_origin_screen(hwnd)
        x_pct = float(kill_button["x"])
        y_pct = float(kill_button["y"])
        x = cx + int((cr - cl) * x_pct)
        y = cy + int((cb - ct) * y_pct)
        
        pyautogui.moveTo(x, y, duration=0.15)
        pyautogui.click()
        print("   ✅ Clicked")

        print("   ⏳ Waiting 10s...")
        time.sleep(10)

        print("   🔄 Re-running first-run")
        run_panel_first_run_if_needed(hwnd, regions, log=log, force=True)
        print("   ✅ Done")
        
        return False
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        if log:
            log.error("CS2 fix failed: %s", e)
        return False


def reposition_console_window():
    """Reposition console to bottom-left corner, above the taskbar"""
    try:
        console_hwnd = win32console.GetConsoleWindow()
        if not console_hwnd:
            return

        # Use work area so the console sits above the taskbar
        work_area = wintypes.RECT()
        windll.user32.SystemParametersInfoW(0x30, 0, byref(work_area), 0)

        console_width = 800
        console_height = 400

        x = work_area.left
        y = work_area.bottom - console_height

        print(f"📐 Console → ({x}, {y})")

        win32gui.SetWindowPos(
            console_hwnd,
            win32con.HWND_TOP,
            x, y,
            console_width, console_height,
            win32con.SWP_SHOWWINDOW
        )

        print("✅ Console repositioned\n")

    except Exception as e:
        print(f"⚠️  Console reposition: {e}\n")

def run_watchdog() -> None:
    log = setup_logger()
    
    # Reposition console BEFORE launching anything
    reposition_console_window()
    
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
    settle_norm_ms = int(app["watchdog"].get("settle_after_normalize_ms", 150))

    # Initialize normalize flag based on config
    normalize_every = bool(app["watchdog"].get("normalize_every_loop", True))

    os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)
    hwnd = None
    last_found_title = None
    last_action_ts = time.time()  # Prevents immediate trigger on startup
    last_logged_latest_line = None
    steam_route_launched = False
    first_run_completed_pids = []  # Ordered list so we keep the most recent on cleanup
    cs2_check_timestamp = None
    cs2_check_done = False

    print("\n" + "="*70)
    print("🐕 APPLICATION WATCHDOG STARTED")
    print("="*70)
    print(f"Warm timeout:    {warm_timeout} min")
    print(f"General timeout: {general_timeout} min")
    print(f"Poll interval:   {poll} sec")
    print("="*70 + "\n")
    
    log.info("Starting watchdog - Warm: %.1f min, General: %.1f min", warm_timeout, general_timeout)

    # OPTIMIZATION: Load regions once at startup (not every loop)
    regions = load_yaml(REGIONS_CFG_PATH)

    # Tracks consecutive "process running but window absent" cycles to prevent infinite loop
    _panel_no_window_iters = 0

    while True:
        # Check window
        if hwnd is None or not win32gui.IsWindow(hwnd):
            print("🔍 Searching for window...")
            hwnd, last_found_title = find_hwnd_by_title_substring(title_sub)

            if not hwnd:
                print(f"⚠️  Window not found ('{title_sub}')")
                log.warning("Window not found.")

                # Guard: if panel process is already running but window is just hidden/loading,
                # don't spawn a second instance — wait and search again.
                panel_dir_check = regions.get("panel", {}).get("dir", "")
                if panel_dir_check and is_panel_running(panel_dir_check):
                    _panel_no_window_iters += 1
                    if _panel_no_window_iters <= 12:
                        print(f"⚠️  Panel running but window not found. Waiting 5s... ({_panel_no_window_iters}/12)")
                        log.warning("Panel running (no window). Waiting 5s. (%d/12)", _panel_no_window_iters)
                        time.sleep(5)
                        continue
                    else:
                        print(f"⚠️  Panel running without window for ~{_panel_no_window_iters * 5}s — forcing new launch")
                        log.warning("Forcing new panel launch after %d no-window retries", _panel_no_window_iters)
                        _panel_no_window_iters = 0
                        # Fall through to launch

                print("Launching panel...")
                log.warning("Launching panel")

                try:
                    regions = load_yaml(REGIONS_CFG_PATH)

                    # ENHANCED: Automatically find newest .exe in panel.dir
                    panel_exe = resolve_panel_exe(regions)

                    if not panel_exe:
                        log.error("Panel EXE not found in panel.dir")
                        print("❌ Panel EXE not found (check regions.yaml → panel.dir)")
                        time.sleep(5)
                        continue

                    log.info("Panel: %s", panel_exe)
                    print(f"\n🚀 Launching: {panel_exe}")
                    exe_dir = os.path.dirname(panel_exe)
                    print(f"   cwd: {exe_dir}")

                    # Launch with working directory set
                    process = subprocess.Popen(
                        [panel_exe],
                        cwd=exe_dir,
                        shell=False
                    )

                    print(f"✅ PID {process.pid}")

                    if not steam_route_launched:
                        launch_steam_route_if_configured(regions, log=log)
                        steam_route_launched = True

                except Exception as e:
                    log.exception("Panel launch failed: %s", e)
                    print(f"❌ Launch failed: {e}")
                    time.sleep(5)
                    continue
                
                # ENHANCED: Progressive retry logic
                print(f"\n🔄 Waiting for window ('{title_sub}')  3s→5s→8s→12s")

                retry_delays = [3, 5, 8, 12]
                panel_dir = regions.get("panel", {}).get("dir", "")

                for attempt, delay in enumerate(retry_delays, 1):
                    print(f"   [{attempt}/{len(retry_delays)}] Waiting {delay}s...")
                    time.sleep(delay)

                    # Primary: Try by title
                    hwnd, last_found_title = find_hwnd_by_title_substring(title_sub)

                    if hwnd:
                        print(f"   ✅ Found by title: '{last_found_title}'")
                        break

                    # Fallback: Try by process path
                    if panel_dir:
                        print(f"   Title not found, trying by path...")
                        hwnd, last_found_title = find_window_by_process_path(panel_dir)
                        if hwnd:
                            print(f"   ✅ Found by path: '{last_found_title}'")
                            break

                    if attempt < len(retry_delays):
                        print(f"   ⚠️  Not found, retrying...")

                if not hwnd:
                    print(f"\n❌ Window not found after all retries.")
                    print(f"   1. Panel >28s to show  2. Title missing '{title_sub}'  3. Crashed on start")
                    print(f"   💡 Check: app.yaml → window.title_substring")
                    log.error("Panel window not found after progressive retries")
                    time.sleep(5)
                    continue
                
                _panel_no_window_iters = 0
                print(f"\n✅ Window: '{last_found_title}' (hwnd={hwnd})")
                log.info("Window found after launch: hwnd=%s title=%r", hwnd, last_found_title)

                print("🔧 First-run check...")
                
                # ANTI-SPAM: Check if we've already run first-run on this panel instance
                _first_run_failed = False
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    
                    if pid in first_run_completed_pids:
                        print(f"✅ First-run done for PID {pid}")
                        log.info("First-run done (PID %d)", pid)
                    else:
                        # Run first-run with retry logic (3 attempts)
                        success = False
                        max_attempts = 3
                        
                        for attempt in range(1, max_attempts + 1):
                            print(f"\n🔄 First-run attempt {attempt}/{max_attempts}")
                            log.info("First-run attempt %d/%d for PID %d", attempt, max_attempts, pid)
                            
                            success = run_panel_first_run_if_needed(hwnd, regions, log=log, force=True)
                            
                            if success:
                                first_run_completed_pids.append(pid)
                                
                                # Schedule CS2 check 5 min from now (reset on each relaunch)
                                cs2_check_timestamp = time.time() + (5 * 60)
                                cs2_check_done = False
                                print(f"⏰ CS2 check in 5 min")
                                if log:
                                    log.info("CS2 check scheduled for 5 min")

                                print(f"✅ First-run ok (attempt {attempt})")
                                log.info("First-run ok (PID %d)", pid)
                                break
                            else:
                                print(f"⚠️  Attempt {attempt} failed")
                                log.warning("Attempt %d failed for PID %d", attempt, pid)

                                if attempt < max_attempts:
                                    print(f"   ⏳ Retrying in 5s...")
                                    time.sleep(5)
                                else:
                                    print(f"❌ All {max_attempts} attempts failed!")
                                    log.error("All attempts failed for PID %d", pid)
                                    _first_run_failed = True

                except Exception as e:
                    log.warning("Could not get PID for first-run tracking: %s", e)
                    # Fall back to running it anyway
                    run_panel_first_run_if_needed(hwnd, regions, log=log, force=True)

                if _first_run_failed:
                    print("⚠️  First-run failed — will retry next loop")
                    log.warning("First-run failed all attempts; invalidating hwnd to force re-check")
                    hwnd = None
                    time.sleep(5)
                    continue

                # Normalize window position AFTER first-run (always runs)
                print("📐 Normalizing window...")
                x, y, moved = normalize_window_bottom_right(
                    hwnd,
                    width=width,
                    height=height,
                    margin_right=margin_right,
                    margin_bottom=margin_bottom,
                )
                if moved:
                    print(f"   ✅ Moved to ({x}, {y})")
                    log.info("Window normalized -> %d, %d", x, y)
                else:
                    print(f"   ℹ️  Already at ({x}, {y})")
                
                time.sleep(settle_norm_ms / 1000)

            else:
                _panel_no_window_iters = 0
                print(f"✅ Window: '{last_found_title}' (hwnd={hwnd})\n")
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

            # Scroll to top before capturing (if log_scroll_point_pct is configured)
            scroll_logbox_to_top(hwnd, regions, verbose=False)

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
            print(f"❌ Capture error: {e}\n")
            log.exception("Capture failed")
            time.sleep(poll)
            continue

        # Logic Analysis (1st pass)
        minutes_ago, hh, mm, latest_line, latest_msg = find_latest_entry(text, debug=True)


        # Check for "Cannot add" error (restart explorer)
        if latest_msg and "cannot add" in latest_msg.lower():
            print(f"⚠️  'Cannot add' detected: {latest_msg[:50]}")
            restart_explorer(log=log)
        
        if minutes_ago is None:
            # Retry once using screen-capture fallback (still inside the same loop)
            try:
                print("⚠️  Parse failed — trying screen fallback...")

                # Screen capture of the same client region
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
        
                # normalize_text_for_parsing handles all pipe variants + colon/I/l fixes
                text2 = normalize_text_for_parsing((ocr_log_text(img2) or "").strip())
        
                if debug_print_ocr:
                    print(f"📄 OCR(fallback): {text2[:100]}...")
        
                minutes_ago, hh, mm, latest_line, latest_msg = find_latest_entry(text2, debug=True)
        
            except Exception as e:
                log.warning("Fallback capture/OCR failed: %s", e)
        
        if minutes_ago is None:
            print("⚠️  No parseable entry found.\n")
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

        if should_trigger:
            if since_last < debounce:
                remaining = debounce - since_last
                print(f"⏸️  Cooldown ({remaining:.0f}s). Reason: {trigger_reason}")
                log.warning("Cooldown active. Reason: %s", trigger_reason)
            else:
                # Disable normalization for next loop to prevent jumping during recovery
                normalize_every = False

                if trigger_recovery_action(hwnd, log, app, trigger_reason):
                    last_action_ts = now_ts
                else:
                    log.warning("Recovery action failed; cooldown not applied")
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
        
        # Cleanup PID tracking to prevent memory growth (keep 10 most recent)
        if len(first_run_completed_pids) > 20:
            first_run_completed_pids = first_run_completed_pids[-10:]
        

        # Check CS2 count (5 min after first-run)
        if cs2_check_timestamp is not None and not cs2_check_done:
            if time.time() >= cs2_check_timestamp:
                print("\n🎮 CS2 INSTANCE CHECK")
                try:
                    check_cs2_instance_count(hwnd, regions, expected=4, log=log)
                finally:
                    cs2_check_done = True
                print("")
        
        time.sleep(poll)

if __name__ == "__main__":
    run_watchdog()