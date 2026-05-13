import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Absolute Package Imports
from controller.config import ARCHIVE_DIR, CSV_DIR
from controller.core.processor import process_archive_file, check_and_trigger_batch
from controller.core.logger import log

class FolderHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.processing = set() # Track files to prevent duplicate processing

    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Avoid processing temporary or hidden files
        if file_path.name.startswith((".", "~")) or file_path.suffix.lower() != '.csv':
            return

        # Simple thread-safe lock to prevent duplicate triggers
        if str(file_path) in self.processing:
            return
            
        self.processing.add(str(file_path))

        try:
            # TRIGGER 1: File lands in archive -> Immediate Evaluation
            if file_path.parent == ARCHIVE_DIR:
                # 1-second sleep is critical for Windows OS file writing completion 
                time.sleep(1.0) 
                if file_path.exists():
                    process_archive_file(file_path)
            
            # TRIGGER 2: Manual drop or move to CSV -> Check 7-day Preprocessing
            elif file_path.parent == CSV_DIR:
                log(f"Staging update detected: {file_path.name}") 
                check_and_trigger_batch()
        finally:
            # Allow the file to be tracked again if it reappears later
            time.sleep(0.5)
            self.processing.discard(str(file_path))

def process_existing_files():
    """
    Sweeps folders for stranded files on startup.
    Order is important: Clean the staging first, then handle the intake. 
    """
    
    # 1. Sweep CSV staging first (Check for 7-day Preprocessing)
    log("Startup Sweep: Checking [csv] staging for existing 7-day buffer...")
    check_and_trigger_batch()
            
    # 2. Sweep the intake ARCHIVE folder (Check for 1-file Evaluation)
    existing_archive = sorted([f for f in ARCHIVE_DIR.iterdir() if f.is_file() and f.suffix.lower() == '.csv'])
    if existing_archive:
        log(f"Startup Sweep: Found {len(existing_archive)} stranded CSV(s) in [archive].")
        for csv_path in existing_archive:
            if csv_path.exists():
                process_archive_file(csv_path) 
                time.sleep(0.5)

# controller/core/monitor.py
def start_monitor():
    """Initializes the live watcher and performs the startup sweep."""
    process_existing_files()

    event_handler = FolderHandler()
    observer = Observer()
    
    observer.schedule(event_handler, str(ARCHIVE_DIR), recursive=False)
    observer.schedule(event_handler, str(CSV_DIR), recursive=False)
    
    observer.start()
    # Removed the rocket emoji to prevent UnicodeEncodeError
    log("LIVE MONITOR ACTIVE: Watching for 5-minute drops in [archive]...") 
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        log("Monitor shutting down...")
    observer.join()