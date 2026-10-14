#!/usr/bin/env python3
"""
Main Pipeline Runner - Easy access to the complete tone calibration system
"""

import sys
import os
from pathlib import Path

# Add scripts directory to Python path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

def main():
    """Run the complete tone calibration pipeline."""
    
    print("🚀 TONE CALIBRATION SYSTEM")
    print("=" * 50)
    print("Available commands:")
    print("  python run_pipeline.py          - Run complete honest pipeline (standalone)")
    print("  python run_pipeline.py web      - Start web application")
    print("=" * 50)
    print("")
    print("📱 WEB APPLICATION (Primary Interface):")
    print("  Frontend: http://localhost:3000")
    print("  Backend API: http://localhost:8000")
    print("  - Real user data collection")
    print("  - Live calibration with Supabase")
    print("  - Enhanced content generation")
    print("")
    print("🖥️  STANDALONE PIPELINE:")
    print("  - Interactive command-line tool")
    print("  - For testing and development")
    print("=" * 50)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "web":
            print("�� Starting web application...")
            print("Run these commands in separate terminals:")
            print("  Terminal 1: cd frontend && npm start")
            print("  Terminal 2: cd residualized_tone_space/api && python main.py")
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: web")
    else:
        # Default: run the honest pipeline
        print("🔄 Running complete honest pipeline (standalone)...")
        from scripts.core.complete_pipeline_honest import main as honest_main
        honest_main()

if __name__ == "__main__":
    main()
