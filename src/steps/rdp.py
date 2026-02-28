"""
rdp.py - RDPClient Automation with Working Directory Fix

FIXED: 
- Loads configuration directly from regions.yaml
- Sets working directory when launching (finds JSON config)
- Handles confirmation dialogs
- NEW: Positions RDP game windows to screen corners after launch
- NEW: Logging to logs/rdp_YYYYMMDD.log
"""
import os
import sys
from pathlib import Path
import time
import subprocess
import ctypes
import ctypes.wintypes
import win32gui
import win32con
import win32api
import logging
from datetime import datetime

# Ensure ../ (src) is on sys.path
_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from winops import (
    set_dpi_awareness,
    find_window,
    wait_for_window,
    force_foreground,
    assert_foreground,
    pct_to_screen_xy,
    safe_double_click,
)
from utils import load_yaml

TITLE_SUB = "RDP Session Manager"
RDP_EXE_DEFAULT = r"C:\Users\Recruiter\Downloads\RDPClient\RDPClient.exe"

# UI wait times
UI_STABILIZATION_WAIT_S = 5.0
DIALOG_WAIT_S = 1.0

def setup_logging():
    """Setup logging to logs/rdp_YYYYMMDD.log"""
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = f"rdp_{datetime.now().strftime('%Y%m%d')}.log"
    log_path = os.path.join(logs_dir, log_file)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def close_confirmation_dialog(hwnd=None, verbose=True):
    """
    Close confirmation dialog by pressing Enter.
    If hwnd is supplied, skips the press when focus has moved to an unrelated
    window — prevents accidentally hitting UI elements in other apps.
    """
    if hwnd is not None:
        fg = win32gui.GetForegroundWindow()
        if fg != hwnd:
            fg_title = win32gui.GetWindowText(fg) if fg else ""
            # Allow child dialogs (their title won't match TITLE_SUB, but they
            # belong to the same process).  Only skip if an entirely different
            # top-level app has grabbed focus.
            if TITLE_SUB.lower() not in fg_title.lower():
                if verbose:
                    print(f"   ⚠️  Focus on unrelated window '{fg_title}' — skipping Enter press")
                return

    if verbose:
        print("   🔘 Pressing Enter to close dialog...")

    import pyautogui
    pyautogui.press('enter')
    time.sleep(0.2)
    pyautogui.press('enter')  # Press twice to be sure

    if verbose:
        print("   ✅ Sent Enter keypress")


def launch_rdp_with_workdir(exe_path, verbose=True):
    """
    Launch RDPClient with its directory as working directory.
    This ensures it finds config files (like user accounts JSON).
    """
    if not os.path.exists(exe_path):
        raise RuntimeError(f"RDP exe not found: {exe_path}")
    
    # Get the directory containing RDPClient.exe
    rdp_dir = os.path.dirname(os.path.abspath(exe_path))
    
    if verbose:
        print(f"🚀 Launching RDPClient")
        print(f"   Exe: {exe_path}")
        print(f"   Cwd: {rdp_dir}")
    
    # Launch with working directory set
    subprocess.Popen([exe_path], cwd=rdp_dir)


# ============================================================================
# NEW FUNCTIONS - Window Positioning (added without changing existing logic)
# ============================================================================

def find_rdp_game_windows(title_substring="SinFermera"):
    """
    Find RDP game windows (SinFermera15, SinFermera16, etc.)
    
    Args:
        title_substring: Text to search for in window titles
    
    Returns:
        List of (hwnd, title) tuples
    """
    windows = []
    
    def enum_callback(hwnd, results):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title and title_substring.lower() in title.lower():
                results.append((hwnd, title))
    
    win32gui.EnumWindows(enum_callback, windows)
    return windows


def position_window_to_corner(hwnd, corner, window_size=(640, 480)):
    """
    Position a window at a corner of the screen.
    
    Args:
        hwnd: Window handle to position
        corner: "top-left" or "bottom-right"
        window_size: (width, height) tuple in pixels
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get usable work area (excludes taskbar regardless of its position)
        work_area = ctypes.wintypes.RECT()
        ctypes.windll.user32.SystemParametersInfoW(0x30, 0, ctypes.byref(work_area), 0)

        width, height = window_size

        # Calculate position based on corner, clamped to the work area
        if corner == "top-left":
            x = work_area.left
            y = work_area.top
        elif corner == "bottom-right":
            x = work_area.right - width
            y = work_area.bottom - height
        else:
            print(f"   ⚠️  Unknown corner: {corner}")
            return False
        
        # Restore window if minimized
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        
        # Move and resize window
        win32gui.SetWindowPos(
            hwnd,
            win32con.HWND_TOP,
            x, y,
            width, height,
            win32con.SWP_SHOWWINDOW
        )
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to position window: {e}")
        return False

def focus_rdp_game_windows(windows_list, verbose=True):
    """Focus both RDP game windows after positioning"""
    if not windows_list or len(windows_list) < 2:
        if verbose:
            print("⚠️  Not enough windows to focus")
        return False
    
    if verbose:
        print("\n🎯 Focusing both RDP game windows...")
    
    for i, (hwnd, title) in enumerate(windows_list[:2], 1):
        try:
            if verbose:
                print(f"   Window {i}: '{title}'")
            
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            
            if verbose:
                print(f"      ✅ Focused")
            
            time.sleep(0.3)
            
        except Exception as e:
            if verbose:
                print(f"      ❌ Failed: {e}")
            return False
    
    return True

def position_rdp_game_windows(config, verbose=True):
    """
    Find and position RDP game windows to screen corners.
    
    Args:
        config: Dict with window positioning settings
        verbose: Print status messages
    
    Returns:
        True if both windows positioned successfully
    """
    # Get config values
    title_search = config.get("title_search", "SinFermera")
    window_width = config.get("width", 640)
    window_height = config.get("height", 480)
    max_wait = config.get("max_wait", 30)
    
    window_size = (window_width, window_height)
    
    if verbose:
        screen_width = win32api.GetSystemMetrics(0)
        screen_height = win32api.GetSystemMetrics(1)
        
        print(f"\n📺 Screen: {screen_width}x{screen_height}")
        print(f"🔍 Looking for '{title_search}' windows...")
    
    # Wait for windows to appear
    start_time = time.time()
    windows = []
    
    while time.time() - start_time < max_wait:
        windows = find_rdp_game_windows(title_search)
        
        if len(windows) >= 2:
            if verbose:
                print(f"✅ Found {len(windows)} window(s)")
            break
        else:
            if verbose:
                print(f"⏳ Waiting... ({len(windows)}/2)")
            time.sleep(2)
    
    if len(windows) < 2:
        if verbose:
            print(f"⚠️  Only found {len(windows)}/2 windows")
        return False
    
    if verbose:
        print(f"\n📐 Positioning {window_width}x{window_height} windows...")
    
    # Position Window 1: Top-left
    hwnd1, title1 = windows[0]
    if verbose:
        print(f"\n   '{title1}' → Top-left (0, 0)")
    
    success1 = position_window_to_corner(hwnd1, "top-left", window_size)
    if verbose:
        print(f"      {'✅' if success1 else '❌'}")
    
    time.sleep(0.3)
    
    # Position Window 2: Bottom-right
    hwnd2, title2 = windows[1]
    if verbose:
        x2 = win32api.GetSystemMetrics(0) - window_width
        y2 = win32api.GetSystemMetrics(1) - window_height
        print(f"\n   '{title2}' → Bottom-right ({x2}, {y2})")
    
    success2 = position_window_to_corner(hwnd2, "bottom-right", window_size)
    if verbose:
        print(f"      {'✅' if success2 else '❌'}")
    focus_rdp_game_windows(windows, verbose=verbose)
        
    return success1 and success2



# ============================================================================
# END NEW FUNCTIONS
# ============================================================================


def run(config=None, context=None):
    """
    Launch RDPClient and double-click User1 and User2 entries.
    
    Automatically loads config from regions.yaml if not provided.
    
    Args:
        config: Optional config dict (for testing)
        context: Optional context (unused)
    
    Returns:
        True if successful
    """
    log = setup_logging()
    log.info("=" * 70)
    log.info("RDP AUTOMATION STARTED")
    log.info("=" * 70)
    
    set_dpi_awareness()

    # Load regions.yaml if config not provided
    if config is None:
        log.info("Loading configuration from regions.yaml")
        try:
            regions = load_yaml("config/regions.yaml")
            rdp_config = regions.get("rdp", {})
            paths = regions.get("paths", {})
            
            # Get exe path from paths section
            rdpclient_exe_list = paths.get("rdpclient_exe", [])
            if rdpclient_exe_list:
                exe_path = rdpclient_exe_list[0]
            else:
                exe_path = RDP_EXE_DEFAULT
            
            # Build complete config
            config = {
                "exe_path": exe_path,
                "user1_point_pct": rdp_config.get("user1_point_pct"),
                "user2_point_pct": rdp_config.get("user2_point_pct"),
                "ui_wait_s": rdp_config.get("ui_wait_s", UI_STABILIZATION_WAIT_S),
                "dialog_wait_s": rdp_config.get("dialog_wait_s", DIALOG_WAIT_S),
            }
            
            log.info(f"Loaded config - Exe: {config['exe_path']}")
            log.info(f"User1: {config['user1_point_pct']}, User2: {config['user2_point_pct']}")
            log.info(f"UI wait: {config['ui_wait_s']}s")
            
            print(f"✅ Config loaded:")
            print(f"   Exe:     {config['exe_path']}")
            print(f"   User1:   {config['user1_point_pct']}")
            print(f"   User2:   {config['user2_point_pct']}")
            print(f"   UI wait: {config['ui_wait_s']}s")
            
        except Exception as e:
            log.error(f"Failed to load regions.yaml: {e}")
            raise RuntimeError(f"Failed to load regions.yaml: {e}")
    
    cfg = config or {}
    exe_path = cfg.get("exe_path", RDP_EXE_DEFAULT)
    
    # Get user click positions from config (REQUIRED!)
    user1 = cfg.get("user1_point_pct")
    user2 = cfg.get("user2_point_pct")
    
    if not user1 or not user2:
        raise RuntimeError(
            "RDP: user1_point_pct and user2_point_pct are required!\n"
            "Add them to regions.yaml under 'rdp' section:\n"
            "rdp:\n"
            "  user1_point_pct:\n"
            "    x: 0.1902\n"
            "    y: 0.2754\n"
            "  user2_point_pct:\n"
            "    x: 0.2366\n"
            "    y: 0.3300\n"
        )
    
    ui_wait_s = cfg.get("ui_wait_s", UI_STABILIZATION_WAIT_S)
    dialog_wait_s = cfg.get("dialog_wait_s", DIALOG_WAIT_S)

    print("\n" + "=" * 70)
    print("🖥️  RDPCLIENT AUTOMATION")
    print("=" * 70)

    # 1) Find or launch RDPClient
    m = find_window(TITLE_SUB, require_visible=True)
    if not m:
        log.info("RDPClient window not found, launching...")
        print("⚠️  RDPClient not found, launching...")

        # Launch with working directory set (CRITICAL FIX!)
        launch_rdp_with_workdir(exe_path, verbose=True)

        log.info("Waiting for RDPClient window (25s timeout)")
        print("⏳ Waiting for window (25s)...")
        m = wait_for_window(TITLE_SUB, timeout_s=25.0, require_visible=True)

    if not m:
        log.error("RDPClient window not found after launch")
        print("❌ Window not found after launch")
        raise RuntimeError("RDP: window not found after launch.")

    log.info(f"Found RDPClient Window (HWND: {m.hwnd}, Title: {m.title})")
    print(f"\n✅ RDPClient window found")
    print(f"   HWND:  {m.hwnd}")
    print(f"   Title: {m.title}")

    # 2) Wait for UI and user list to fully load
    log.info(f"Waiting {ui_wait_s}s for UI and user list to load")
    print(f"\n⏳ Waiting {ui_wait_s}s for UI...")
    time.sleep(ui_wait_s)

    # 3) Force focus before first click
    log.info("Focusing RDPClient window")
    print("\n🎯 Focusing RDPClient...")
    if not force_foreground(m.hwnd):
        log.error("Failed to focus RDPClient")
        print("❌ Failed to focus RDPClient")
        raise RuntimeError("RDP: could not foreground (safety stop).")

    assert_foreground(m.hwnd)
    log.info("Window focused")
    print("✅ Focused")
    
    time.sleep(0.3)

    # 4) Double-click User1
    x1, y1 = pct_to_screen_xy(m.hwnd, float(user1["x"]), float(user1["y"]))
    
    log.info(f"Double-clicking User1 at ({user1['x']:.4f}, {user1['y']:.4f}) → screen ({x1}, {y1})")
    print(f"\n🖱️  Double-clicking User1  ({user1['x']:.4f}, {user1['y']:.4f}) → ({x1}, {y1})")

    safe_double_click(x1, y1)
    log.info("User1 double-clicked")
    print("✅ User1 clicked")

    # 4a) Close confirmation dialog
    log.info(f"Waiting {dialog_wait_s}s for User1 confirmation dialog")
    print(f"⏳ Waiting {dialog_wait_s}s for dialog...")
    time.sleep(dialog_wait_s)
    close_confirmation_dialog(hwnd=m.hwnd, verbose=True)
    time.sleep(0.5)

    # 5) Refocus RDPClient for second click
    log.info("Re-focusing RDPClient window")
    print("\n🎯 Re-focusing RDPClient...")
    if not force_foreground(m.hwnd):
        log.error("Failed to re-focus RDPClient after User1")
        print("❌ Re-focus failed after User1")
        raise RuntimeError("RDP: could not refocus after user1 (safety stop).")

    assert_foreground(m.hwnd)
    log.info("Window re-focused")
    print("✅ Re-focused")
    
    time.sleep(0.3)

    # 6) Double-click User2
    x2, y2 = pct_to_screen_xy(m.hwnd, float(user2["x"]), float(user2["y"]))
    
    log.info(f"Double-clicking User2 at ({user2['x']:.4f}, {user2['y']:.4f}) → screen ({x2}, {y2})")
    print(f"\n🖱️  Double-clicking User2  ({user2['x']:.4f}, {user2['y']:.4f}) → ({x2}, {y2})")

    safe_double_click(x2, y2)
    log.info("User2 double-clicked")
    print("✅ User2 clicked")

    # 6a) Close confirmation dialog
    log.info(f"Waiting {dialog_wait_s}s for User2 confirmation dialog")
    print(f"⏳ Waiting {dialog_wait_s}s for dialog...")
    time.sleep(dialog_wait_s)
    close_confirmation_dialog(hwnd=m.hwnd, verbose=True)
    time.sleep(0.5)

    log.info("RDPCLIENT AUTOMATION COMPLETE - Both RDP sessions opening")
    print("\n" + "=" * 70)
    print("✅ RDPCLIENT AUTOMATION COMPLETE")
    print("=" * 70)
    print("Both RDP sessions should now be opening\n")

    # ========================================================================
    # NEW: Position RDP game windows (added without changing above logic)
    # ========================================================================
    try:
        regions = load_yaml("config/regions.yaml")
        rdp_windows_cfg = regions.get("rdp_windows", {})
        
        if rdp_windows_cfg:
            log.info("Starting game window positioning")
            print("=" * 70)
            print("📐 POSITIONING GAME WINDOWS")
            print("=" * 70)
            
            success = position_rdp_game_windows(rdp_windows_cfg, verbose=True)
            if success:
                log.info("Game windows positioned successfully")
                print("✅ Positioned correctly")
            else:
                log.warning("Game window positioning incomplete")
                print("⚠️  Positioning incomplete")
    except Exception as e:
        log.warning(f"Window positioning failed: {e}")
        print(f"\n⚠️  Window positioning failed: {e}")
        print("   (RDP sessions opened, windows not repositioned)\n")
    # ========================================================================

    log.info("=" * 70)
    log.info("RDP AUTOMATION FINISHED")
    log.info("=" * 70)
    
    return True


if __name__ == "__main__":
    # Test - will automatically load from regions.yaml
    run()