"""Launcher entry point for the CyUI workspace."""
import sys
import traceback

from settings import DATA


def main():
    try:
        from updates import apply_on_startup
        apply_on_startup()
        from studio_ui import main as run
        run()
    except Exception:
        report = DATA / "logs" / "startup-error.log"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(traceback.format_exc(), encoding="utf-8")
        if getattr(sys, "stderr", None) is not None:
            traceback.print_exc()
        elif sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, f"ImageStudio could not open.\n{report}", "CyTools_ImageStudio", 0x10)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
