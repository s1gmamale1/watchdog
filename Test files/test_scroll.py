"""
Test script to verify scroll function is working
Run this to SEE the scroll happen in real-time
"""
import pyautogui
import time
import win32gui
import win32con

def test_scroll():
    print("=" * 60)
    print("SCROLL FUNCTION TEST")
    print("=" * 60)
    print()
    print("This will:")
    print("1. Find FSM Panel window")
    print("2. Move mouse to scroll bar (you'll SEE it move)")
    print("3. Double-click to scroll to TOP")
    print()
    print("Make sure FSM Panel is OPEN and VISIBLE!")
    print()
    
    # Find window
    title_substring = "524AAD7FA11896EC"  # Your window title
    
    def enum_callback(hwnd, results):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title_substring.lower() in title.lower():
                results.append((hwnd, title))
    
    windows = []
    win32gui.EnumWindows(enum_callback, windows)
    
    if not windows:
        print(f"❌ Window not found with title containing: {title_substring}")
        print("   Make sure FSM Panel is running!")
        return
    
    hwnd, title = windows[0]
    print(f"✅ Found window: {title}")
    print()
    
    # Get window position and size
    rect = win32gui.GetClientRect(hwnd)
    cx, cy = win32gui.ClientToScreen(hwnd, (0, 0))
    cw = rect[2] - rect[0]
    ch = rect[3] - rect[1]
    
    print(f"Window info:")
    print(f"  Screen position: ({cx}, {cy})")
    print(f"  Client size: {cw}x{ch}")
    print()
    
    # Configure scroll position (from regions.yaml)
    x_pct = 0.95  # Right side (scroll bar)
    y_pct = 0.08  # TOP area (latest messages)
    
    x = cx + int(cw * x_pct)
    y = cy + int(ch * y_pct)
    
    print(f"Scroll target:")
    print(f"  Percentage: ({x_pct}, {y_pct})")
    print(f"  Screen coords: ({x}, {y})")
    print()
    
    # Focus window
    print("Step 1: Focusing window...")
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(0.5)
    print("  ✅ Window focused")
    print()
    
    # Move mouse (SLOWLY so you can see it)
    print("Step 2: Moving mouse to scroll bar...")
    print(f"  👀 WATCH THE MOUSE - it will move to ({x}, {y})")
    time.sleep(2)  # Pause so you can look
    
    pyautogui.moveTo(x, y, duration=1.0)  # Slow move (1 second)
    print(f"  ✅ Mouse moved")
    print()
    
    # Pause so you can verify position
    print("Step 3: Pausing 2 seconds...")
    print("  👀 CHECK: Is mouse on the scroll bar near TOP?")
    print("     If YES: Great! Scroll will work!")
    print("     If NO: Adjust x_pct and y_pct in regions.yaml")
    time.sleep(2)
    print()
    
    # Double-click
    print("Step 4: Double-clicking scroll bar...")
    print("  👀 WATCH FSM PANEL - log box should scroll to TOP!")
    pyautogui.doubleClick()
    time.sleep(0.5)
    print("  ✅ Double-click executed")
    print()
    
    # Wait for scroll
    print("Step 5: Waiting for scroll animation...")
    time.sleep(1)
    print("  ✅ Done!")
    print()
    
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print()
    print("Questions to verify:")
    print("1. Did you SEE the mouse move to scroll bar? YES/NO")
    print("2. Was mouse position CORRECT (on scroll bar)? YES/NO")
    print("3. Did log box SCROLL to TOP after double-click? YES/NO")
    print()
    print("If all YES → Scroll is working! ✅")
    print("If any NO → Adjust coordinates in regions.yaml")
    print()
    print("Recommended coordinates for FSM Panel:")
    print("  x: 0.95  (scroll bar on right)")
    print("  y: 0.08  (TOP area where latest messages are)")
    print()

if __name__ == "__main__":
    test_scroll()