#!/usr/bin/env python3
"""
DJ Music Analyzer - GUI Application
A desktop interface for the DJMusicCleaner script
"""

import os
import sys
import json
import threading
import queue
import time
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime

# Import drag and drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DRAG_DROP_SUPPORT = True
except ImportError:
    print("Warning: tkinterdnd2 not found, drag & drop will be disabled")
    print("To enable, install with: pip install tkinterdnd2")
    DRAG_DROP_SUPPORT = False

# Import the DJMusicCleaner class from the original script
try:
    from djmusiccleaner.dj_music_cleaner import DJMusicCleaner
    from djmusiccleaner.gui_settings import load_settings, save_settings
except ImportError:
    try:
        # Try relative import for when running directly
        from dj_music_cleaner import DJMusicCleaner
        from gui_settings import load_settings, save_settings
    except ImportError:
        # Fallback if script is in same directory
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from dj_music_cleaner import DJMusicCleaner
        
        # For settings module
        try:
            from gui_settings import load_settings, save_settings
        except ImportError:
            # Define fallback settings functions if module not found
            def load_settings():
                return {}
                
            def save_settings(settings):
                return False


class RedirectText:
    """Redirect stdout to a queue for display in the GUI"""
    
    def __init__(self, text_queue):
        self.text_queue = text_queue
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, string):
        self.text_queue.put(string)
        # Also write to original stdout for console logs
        self.stdout.write(string)
        
    def flush(self):
        self.stdout.flush()
        self.stderr.flush()


class DJMusicAnalyzerGUI:
    """Main GUI application for DJ Music Analyzer"""
    
    def __init__(self, root):
        self.root = root
        
        # Load saved settings
        self.settings = load_settings()
        self.root.title("DJ Music Analyzer")
        self.root.minsize(800, 600)
        
        # Set theme
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use a modern looking theme
        
        # Configure colors
        self.root.configure(bg="#f0f0f0")  # Light gray background
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", background="#4a7abc", foreground="white")
        self.style.configure("TLabel", background="#f0f0f0")
        self.style.configure("TCheckbutton", background="#f0f0f0")
        
        # Initialize variables with saved settings if available
        self.input_folder = tk.StringVar(value=self.settings.get("input_folder", ""))
        self.output_folder = tk.StringVar(value=self.settings.get("output_folder", ""))
        
        # Option category toggles
        self.online_enhancement = tk.BooleanVar(value=self.settings.get("online_enhancement", False))
        self.dj_analysis = tk.BooleanVar(value=self.settings.get("dj_analysis", True))
        self.high_quality_only = tk.BooleanVar(value=self.settings.get("high_quality_only", False))
        
        # Online Enhancement options
        self.acoustid_api_key = tk.StringVar(value=self.settings.get("acoustid_api_key", ""))
        self.cache_path = tk.StringVar(value=self.settings.get("cache", ""))
        self.skip_id3 = tk.BooleanVar(value=self.settings.get("skip_id3", False))
        
        # DJ Analysis options
        self.detect_key = tk.BooleanVar(value=self.settings.get("detect_key", True))
        self.detect_cues = tk.BooleanVar(value=self.settings.get("detect_cues", True))
        self.calculate_energy = tk.BooleanVar(value=self.settings.get("calculate_energy", True))
        self.detect_bpm = tk.BooleanVar(value=self.settings.get("detect_bpm", True))
        self.normalize_tags = tk.BooleanVar(value=self.settings.get("normalize_tags", True))
        self.normalize_loudness = tk.BooleanVar(value=self.settings.get("normalize_loudness", False))
        self.target_lufs = tk.DoubleVar(value=self.settings.get("target_lufs", -14.0))
        
        # Rekordbox options
        self.rekordbox_xml = tk.StringVar(value=self.settings.get("rekordbox_xml", ""))
        self.export_rekordbox = tk.StringVar(value=self.settings.get("export_rekordbox", ""))
        self.rekordbox_preserve = tk.BooleanVar(value=self.settings.get("rekordbox_preserve", False))
        
        # High Quality options
        self.analyze_audio = tk.BooleanVar(value=self.settings.get("analyze_audio", True))
        self.generate_report = tk.BooleanVar(value=self.settings.get("report", True))
        self.detailed_report = tk.BooleanVar(value=self.settings.get("detailed_report", True))
        self.json_report = tk.BooleanVar(value=self.settings.get("json_report", True))
        self.csv_report = tk.BooleanVar(value=self.settings.get("csv_report", False))
        self.html_report_path = tk.StringVar(value=self.settings.get("html_report_path", ""))
        self.year_in_filename = tk.BooleanVar(value=self.settings.get("year_in_filename", False))
        self.dry_run = tk.BooleanVar(value=self.settings.get("dry_run", False))
        self.workers = tk.IntVar(value=self.settings.get("workers", 0))
        self.find_duplicates = tk.BooleanVar(value=self.settings.get("find_duplicates", False))  # Disabled by default
        self.show_priorities = tk.BooleanVar(value=self.settings.get("show_priorities", True))
        
        # Processing state variables
        self.processing = False
        self.process_thread = None
        self.text_queue = queue.Queue()
        self.progress_value = tk.DoubleVar(value=0)
        self.current_file = tk.StringVar(value="Ready")
        self.processed_count = tk.IntVar(value=0)
        self.total_count = tk.IntVar(value=0)
        self.error_count = tk.IntVar(value=0)
        self.cli_command = tk.StringVar(value="")
        
        # Update option states when category toggles change
        self.dj_analysis.trace_add("write", self.update_dj_analysis_options)
        self.high_quality_only.trace_add("write", self.update_high_quality_options)
        self.online_enhancement.trace_add("write", self.update_online_enhancement_options)
        
        # Add traces to update CLI command when any option changes
        for var_name in dir(self):
            var = getattr(self, var_name)
            if isinstance(var, (tk.BooleanVar, tk.StringVar, tk.IntVar, tk.DoubleVar)):
                var.trace_add("write", self.update_cli_display)
        
        # Load API key from environment if available
        api_key = os.environ.get("ACOUSTID_API_KEY", "")
        self.acoustid_api_key.set(api_key)
        
        # Create layout
        self.create_widgets()
        
        # Configure drag and drop
        self.setup_drag_drop()
        
        # Set up queue checker for output text
        self.check_queue()
        
        # Set up window close event handler to save settings
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_widgets(self):
        """Create all widgets for the application"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # App title and description
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = ttk.Label(title_frame, text="DJ Music Analyzer", font=("Helvetica", 16, "bold"))
        title_label.pack(side=tk.LEFT)
        
        # Folder selection section
        folder_frame = ttk.LabelFrame(main_frame, text="File Locations", padding="10")
        folder_frame.pack(fill=tk.X, pady=10)
        
        # Input folder with drag-drop indicator
        input_label = ttk.Label(folder_frame, text="Input Folder:")
        input_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        input_entry = ttk.Entry(folder_frame, textvariable=self.input_folder, width=50)
        input_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # If drag & drop is supported, add special styling
        if DRAG_DROP_SUPPORT:
            drop_text = ttk.Label(folder_frame, text="(drag & drop folder here)", foreground="gray")
            drop_text.grid(row=0, column=1, sticky=tk.E, padx=(0, 30))
            # Make the entry a drop target
            input_entry.drop_target_register(DND_FILES)
            input_entry.dnd_bind("<<Drop>>", self.handle_drop)
            
        ttk.Button(folder_frame, text="Browse", command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)
        
        # Output folder
        ttk.Label(folder_frame, text="Output Folder:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(folder_frame, textvariable=self.output_folder, width=50).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(folder_frame, text="Browse", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)
        
        # Options section
        options_frame = ttk.LabelFrame(main_frame, text="Processing Options", padding="10")
        options_frame.pack(fill=tk.X, pady=10)
        
        # Checkboxes
        ttk.Checkbutton(options_frame, text="Online Enhancement", variable=self.online_enhancement).grid(
            row=0, column=0, sticky=tk.W, padx=20, pady=5)
        ttk.Checkbutton(options_frame, text="DJ Analysis", variable=self.dj_analysis).grid(
            row=0, column=1, sticky=tk.W, padx=20, pady=5)
        ttk.Checkbutton(options_frame, text="High Quality Only", variable=self.high_quality_only).grid(
            row=0, column=2, sticky=tk.W, padx=20, pady=5)
        
        # CLI Command Display section
        cli_frame = ttk.LabelFrame(main_frame, text="CLI Command", padding="10")
        cli_frame.pack(fill=tk.X, pady=10)
        
        # Command display (editable text widget with scrollbar)
        cli_display = tk.Text(cli_frame, wrap=tk.WORD, height=3)
        cli_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        cli_display.insert(tk.END, "Command will appear here and can be edited before processing")
        
        # Add instructions for editing
        edit_label = ttk.Label(cli_frame, text="✏️ You can edit this command before processing")
        edit_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5)
        
        # Store reference to the text widget
        self.cli_display = cli_display
        
        # Scrollbar for command display
        cli_scrollbar = ttk.Scrollbar(cli_frame, command=cli_display.yview)
        cli_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        cli_display.config(yscrollcommand=cli_scrollbar.set)
        
        # API Key section (collapsible)
        self.api_frame = ttk.LabelFrame(main_frame, text="API Settings (Optional)", padding="10")
        self.api_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(self.api_frame, text="AcoustID API Key:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.api_frame, textvariable=self.acoustid_api_key, width=50).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Label(self.api_frame, text="Get key at https://acoustid.org/api-key").grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=5)
        
        # Process buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.process_button = ttk.Button(button_frame, text="Process Files", command=self.process)
        self.process_button.pack(side=tk.LEFT, padx=10)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_value, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=10, padx=5)
        
        # Current file label
        self.file_label = ttk.Label(progress_frame, textvariable=self.current_file)
        self.file_label.pack(fill=tk.X, pady=5, padx=5)
        
        # Status counters
        status_frame = ttk.Frame(progress_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(status_frame, text="Processed:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Label(status_frame, textvariable=self.processed_count).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(status_frame, text="Total:").grid(row=0, column=2, sticky=tk.W, padx=20)
        ttk.Label(status_frame, textvariable=self.total_count).grid(row=0, column=3, sticky=tk.W)
        
        ttk.Label(status_frame, text="Errors:").grid(row=0, column=4, sticky=tk.W, padx=20)
        ttk.Label(status_frame, textvariable=self.error_count).grid(row=0, column=5, sticky=tk.W)
        
        # Reports link
        self.report_link = ttk.Label(progress_frame, text="")
        self.report_link.pack(fill=tk.X, pady=5, padx=5)
        
        # Log output
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Add scrollbars to the log
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for different message types
        self.log_text.tag_configure("error", foreground="red")
        self.log_text.tag_configure("success", foreground="green")
        self.log_text.tag_configure("warning", foreground="orange")
        self.log_text.tag_configure("feature", foreground="blue")
        
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_drag_drop(self):
        """Set up drag and drop for the input folder field"""
        if DRAG_DROP_SUPPORT:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind("<<Drop>>", self.handle_drop)
    
    def handle_drop(self, event):
        """Handle drag and drop for folder selection"""
        if not DRAG_DROP_SUPPORT:
            return
            
        # Get the dropped files/folders
        data = event.data
        
        # Handle multiple paths (space-separated)
        paths = data.split(" ")
        data = paths[0]  # Just use the first one
        
        # Remove curly braces if present (from some platforms)
        if data.startswith("{") and data.endswith("}"):
            data = data[1:-1]
            
        # Remove quotes if present
        if data.startswith('"') and data.endswith('"'):
            data = data[1:-1]
        
        # Check if it's a valid directory
        path = Path(data)
        if path.is_dir():
            self.input_folder.set(str(path))
            self.status_var.set(f"Input folder set to: {path}")
        else:
            messagebox.showwarning("Invalid Drop", "Please drop a folder, not a file.")
    
    def browse_input(self):
        """Browse for input folder"""
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_folder.set(folder)
    
    def browse_output(self):
        """Browse for output folder"""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder.set(folder)
    
    def check_queue(self):
        """Check for messages in the queue and display them"""
        try:
            # Process up to 50 messages at once to prevent UI freezing with large logs
            messages_processed = 0
            while messages_processed < 50:
                # Get message without waiting
                message = self.text_queue.get_nowait()
                
                # Add to log with formatting based on message type
                if "Error:" in message or "❌" in message:
                    self.log_text.insert(tk.END, message, "error")
                elif "✓" in message or "✅" in message or "Success" in message:
                    self.log_text.insert(tk.END, message, "success")
                elif "⚠️" in message or "Warning" in message:
                    self.log_text.insert(tk.END, message, "warning")
                elif any(marker in message for marker in ["🎵", "🎹", "⚡", "🎧", "🎚", "🎛"]):
                    self.log_text.insert(tk.END, message, "feature")
                else:
                    self.log_text.insert(tk.END, message)
                    
                self.log_text.see(tk.END)  # Scroll to the end
                
                # PROGRESS: Track file processing
                if "Analyzing" in message and ".mp3" in message:
                    # Extract current file name
                    try:
                        filename = message.split("Analyzing")[1].strip()
                        self.current_file.set(f"Processing: {filename}")
                        # We'll use the PROGRESS messages to update the processed count instead of incrementing here
                        # This avoids double-counting files
                        
                        # Update progress bar
                        total = max(self.total_count.get(), 1)  # Avoid division by zero
                        progress = min((self.processed_count.get() / total) * 100, 100)  # Cap at 100%
                        self.progress_value.set(progress)
                    except Exception:
                        # If extraction fails, ignore - don't crash the UI
                        pass
                
                # PROGRESS: Detect both old-style and new JSON-formatted progress messages
                if "PROGRESS:" in message or "PROGRESS {" in message:
                    try:
                        # Handle old-style progress format (PROGRESS: 7/23)
                        if "PROGRESS:" in message:
                            parts = message.split("PROGRESS:")
                            if len(parts) > 1:
                                progress_info = parts[1].strip()
                                if "/" in progress_info:
                                    current, total = map(int, progress_info.split("/"))
                                    progress = (current / max(total, 1)) * 100
                                    self.progress_value.set(progress)
                                    self.processed_count.set(current)
                                    self.total_count.set(total)
                        # Handle JSON progress format (PROGRESS {"file": "...", "idx": 7, "total": 23, "phase": "analyze"})
                        elif "PROGRESS {" in message:
                            import json
                            json_start = message.find("PROGRESS ") + 9
                            json_str = message[json_start:].strip()
                            progress_data = json.loads(json_str)
                            if "idx" in progress_data and "total" in progress_data:
                                current = progress_data["idx"]
                                total = progress_data["total"]
                                progress = (current / max(total, 1)) * 100
                                self.progress_value.set(progress)
                                self.processed_count.set(current)
                                self.total_count.set(total)
                                # Also update current file if available
                                if "file" in progress_data:
                                    filename = os.path.basename(progress_data["file"])
                                    self.current_file.set(f"Processing: {filename}")
                                # Update status with phase if available
                                if "phase" in progress_data:
                                    self.status_var.set(f"Phase: {progress_data['phase']} ({current}/{total})")
                    except Exception:
                        # If extraction fails, ignore - don't crash the UI
                        pass
                
                # REPORTS: Detect report path
                if "report generated:" in message or "Report available at:" in message:
                    try:
                        # Extract path, accounting for both formats
                        if "report generated:" in message:
                            report_path = message.split("report generated:")[1].strip()
                        else:
                            report_path = message.split("Report available at:")[1].strip()
                            
                        self.report_link.config(text=f"Report available at: {report_path}", 
                                               foreground="blue", cursor="hand2")
                        self.report_link.bind("<Button-1>", lambda e, path=report_path: self.open_report(path))
                    except Exception:
                        # If extraction fails, ignore
                        pass
                
                # ERRORS: Count errors for reporting
                if "Error:" in message or "❌" in message:
                    self.error_count.set(self.error_count.get() + 1)
                    self.status_var.set(f"Error encountered: {self.error_count.get()} total errors")
                
                # COMPLETION: Check if processing is done
                if "Done!" in message or "Processing complete" in message or "✨ Processing complete" in message:
                    self.processing_done()
                
                messages_processed += 1
                
        except queue.Empty:
            # No more messages in queue
            pass
        finally:
            # Always update the UI during processing
            if self.processing:
                self.root.update_idletasks()
            
            # Continue checking every 50ms for smoother updates
            self.root.after(50, self.check_queue)
    
    def open_report(self, path):
        """Open the generated report"""
        import webbrowser
        webbrowser.open(f"file://{path}")
    
    def update_ui_state(self):
        """Update UI state based on current processing status"""
        if self.processing:
            self.process_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_var.set("Processing...")
        else:
            self.process_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_var.set("Ready")
    
    def process(self):
        """Process files - run DJ Music Cleaner"""
        if self.processing:
            messagebox.showinfo("Processing", "Processing already in progress!")
            return
        
        input_folder = self.input_folder.get()
        output_folder = self.output_folder.get()
        
        if not input_folder or not os.path.isdir(input_folder):
            messagebox.showerror("Error", "Please select a valid input folder")
            return
            
        if not output_folder or not os.path.isdir(output_folder):
            messagebox.showerror("Error", "Please select a valid output folder")
            return
        
        # Save settings before processing
        self.save_current_settings()
        
        # Update UI
        self.processing = True
        self.update_ui_state()
        self.current_file.set("Starting...")
        self.processed_count.set(0)
        self.total_count.set(0)
        self.error_count.set(0)
        
        # Build command with all options and update display if it's empty
        if not self.cli_display.get(1.0, tk.END).strip():
            cmd = self.update_cli_command()
            self.cli_display.delete(1.0, tk.END)
            self.cli_display.insert(tk.END, " ".join(cmd))
        else:
            # Get the command from the editable text box
            cmd_text = self.cli_display.get(1.0, tk.END).strip()
            cmd = cmd_text.split()
        
        # Log the command to the console for debugging
        print("Executing command:", " ".join(cmd))
        if hasattr(self, 'append_to_log'):
            self.append_to_log(f"Command: {' '.join(cmd)}\n")
        
        # Start the process
        self.start_process_thread(cmd)
    
    def start_process_thread(self, cmd):
        """Start the processing in a separate thread"""
        # Redirect stdout to capture log output
        self.stdout_backup = sys.stdout
        sys.stdout = RedirectText(self.text_queue)
        
        # Process in a separate thread
        self.process_thread = threading.Thread(target=self.run_process, args=(cmd,))
        self.process_thread.daemon = True
        self.process_thread.start()
        
    def run_process(self, cmd):
        """Run the process using subprocess"""
        try:
            # Run the command and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Process output in real-time
            for line in process.stdout:
                print(line, end='')
                
            # Wait for process to complete
            process.wait()
            
            # Display completion message
            print("\n✨ Processing complete!")
            print(f"Command executed with exit code: {process.returncode}")
            
        except Exception as e:
            print(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Reset stdout
            sys.stdout = self.stdout_backup
            # Mark processing as done
            self.root.after(0, self.processing_done)
    
    def processing_done(self):
        """Called when processing is finished"""
        # Reset processing state
        self.processing = False
        self.process_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.current_file.set("Done")
        self.status_var.set("Processing complete")
        
        # Ensure progress bar shows 100%
        self.progress_value.set(100)
        
        # Show completion message
        messagebox.showinfo("Processing Complete", 
                           f"Processed {self.processed_count.get()} files with {self.error_count.get()} errors.")
    
    def stop_processing(self):
        """Stop the current processing"""
        if not self.processing or not self.process_thread:
            return
        
        # There's no direct way to stop DJMusicCleaner once started
        # But we can indicate processing should stop
        self.processing = False
        messagebox.showinfo("Stopping", "Trying to stop processing. Please wait for current operation to finish.")
        self.status_var.set("Stopping...")
    
    def save_current_settings(self):
        """Save current GUI settings to file"""
        settings = {
            # Basic folders and API key
            "input_folder": self.input_folder.get(),
            "output_folder": self.output_folder.get(),
            "acoustid_api_key": self.acoustid_api_key.get(),
            "last_used_date": datetime.now().isoformat(),
            
            # Option categories
            "online_enhancement": self.online_enhancement.get(),
            "dj_analysis": self.dj_analysis.get(),
            "high_quality_only": self.high_quality_only.get(),
            
            # Online Enhancement options
            "cache": self.cache_path.get(),
            "skip_id3": self.skip_id3.get(),
            
            # DJ Analysis options
            "detect_key": self.detect_key.get(),
            "detect_cues": self.detect_cues.get(),
            "calculate_energy": self.calculate_energy.get(),
            "detect_bpm": self.detect_bpm.get(),
            "normalize_tags": self.normalize_tags.get(),
            "normalize_loudness": self.normalize_loudness.get(),
            "target_lufs": self.target_lufs.get(),
            
            # Rekordbox options
            "rekordbox_xml": self.rekordbox_xml.get(),
            "export_rekordbox": self.export_rekordbox.get(),
            "rekordbox_preserve": self.rekordbox_preserve.get(),
            
            # High Quality options
            "analyze_audio": self.analyze_audio.get(),
            "report": self.generate_report.get(),
            "detailed_report": self.detailed_report.get(),
            "json_report": self.json_report.get(),
            "csv_report": self.csv_report.get(),
            "html_report_path": self.html_report_path.get(),
            "year_in_filename": self.year_in_filename.get(),
            "dry_run": self.dry_run.get(),
            "workers": self.workers.get(),
            "find_duplicates": self.find_duplicates.get(),
            "show_priorities": self.show_priorities.get()
        }
        
        success = save_settings(settings)
        if success:
            print("Settings saved successfully")
            self.settings = settings
        else:
            print("Failed to save settings")
    
    def update_dj_analysis_options(self, *args):
        """Update DJ Analysis options based on toggle state"""
        if self.dj_analysis.get():
            # Set default TRUE options
            self.detect_key.set(True)
            self.detect_cues.set(True)
            self.calculate_energy.set(True)
            self.detect_bpm.set(True)
            self.normalize_tags.set(True)
        else:
            # Clear all options when disabled
            self.detect_key.set(False)
            self.detect_cues.set(False)
            self.calculate_energy.set(False)
            self.detect_bpm.set(False)
            self.normalize_tags.set(False)
            self.normalize_loudness.set(False)
    
    def update_high_quality_options(self, *args):
        """Update High Quality options based on toggle state"""
        if self.high_quality_only.get():
            # Set default TRUE options
            self.analyze_audio.set(True)
            self.generate_report.set(True)
            self.detailed_report.set(True)
            self.json_report.set(True)
            self.find_duplicates.set(True)
            self.show_priorities.set(True)
        else:
            # Clear options when disabled
            self.analyze_audio.set(False)
            self.generate_report.set(False)
            self.detailed_report.set(False)
            self.json_report.set(False)
            self.find_duplicates.set(False)
            self.show_priorities.set(False)
    
    def update_online_enhancement_options(self, *args):
        """Update Online Enhancement options based on toggle state"""
        # No default options to set for online enhancement currently
        pass
        
    def update_cli_command(self):
        """Update the CLI command display"""
        # Start with the base command
        cmd = ["python", "-m", "djmusiccleaner.dj_music_cleaner"]
        
        # Add input and output folders
        input_folder = self.input_folder.get()
        output_folder = self.output_folder.get()
        
        if input_folder:
            cmd.extend(["--input", f"\"{input_folder}\""]) 
        if output_folder:
            cmd.extend(["--output", f"\"{output_folder}\""]) 
            
        # Update display with current command
        cmd_text = " ".join(cmd)
        
        # Process grouped options
        if self.online_enhancement.get():
            cmd.append("--online")
            if self.acoustid_api_key.get():
                cmd.extend(["--api-key", self.acoustid_api_key.get()])
            if self.cache_path.get():
                cmd.extend(["--cache", f"\"{self.cache_path.get()}\""])
            if self.skip_id3.get():
                cmd.append("--skip-id3")
        
        # DJ Analysis options
        if not self.dj_analysis.get():
            cmd.append("--no-dj")
        else:
            if self.detect_key.get():
                cmd.append("--detect-key")
            else:
                cmd.append("--no-key")
                
            if self.detect_cues.get():
                # No explicit flag needed, it's default behavior
                pass
            else:
                cmd.append("--no-cues")
                
            if self.calculate_energy.get():
                cmd.append("--calculate-energy")
            else:
                cmd.append("--no-energy")
                
            if self.detect_bpm.get():
                cmd.append("--detect-bpm")
                
            if self.normalize_tags.get():
                cmd.append("--normalize-tags")
                
            if self.normalize_loudness.get():
                cmd.append("--normalize")
                cmd.extend(["--lufs", str(self.target_lufs.get())])
        
        # Rekordbox options
        if self.rekordbox_xml.get():
            cmd.extend(["--import-rekordbox", f"\"{self.rekordbox_xml.get()}\""])
        if self.export_rekordbox.get():
            cmd.extend(["--export-rekordbox", f"\"{self.export_rekordbox.get()}\""])
        if self.rekordbox_preserve.get():
            cmd.append("--rekordbox-preserve")
        
        # High Quality options
        if not self.analyze_audio.get():
            cmd.append("--no-quality")
        if self.high_quality_only.get():
            cmd.append("--high-quality")
        if self.find_duplicates.get():
            cmd.append("--find-duplicates")
        if self.show_priorities.get():
            cmd.append("--priorities")
        if self.year_in_filename.get():
            cmd.append("--year")
        if self.dry_run.get():
            cmd.append("--dry-run")
        if self.workers.get() != 0:
            cmd.extend(["--workers", str(self.workers.get())])
        
        # Report options
        if not self.generate_report.get():
            cmd.append("--no-report")
        elif self.html_report_path.get():
            cmd.extend(["--html-report", f"\"{self.html_report_path.get()}\""])
        else:
            cmd.append("--report")
            
        if not self.detailed_report.get():
            cmd.append("--no-detailed-report")
        else:
            cmd.append("--detailed-report")
            
        # JSON report path - automatically set to reports directory in output folder
        if self.json_report.get():
            json_report_path = os.path.join(self.output_folder.get(), "reports", "processed_files.json")
            cmd.extend(["--json-report", json_report_path])
            
        # CSV report path - automatically set to reports directory in output folder
        if self.csv_report.get():
            csv_report_path = os.path.join(self.output_folder.get(), "reports", "processed_files.csv")
            cmd.extend(["--csv-report", csv_report_path])
        
        # Store the command for reference
        self.cli_command.set(" ".join(cmd))
        return cmd
    
    def update_cli_display(self, *args):
        """Update the CLI command display when any option changes"""
        # Only update if GUI is fully initialized
        if hasattr(self, 'cli_display'):
            cmd = self.update_cli_command()
            # Update the text widget
            self.cli_display.config(state=tk.NORMAL)
            self.cli_display.delete(1.0, tk.END)
            self.cli_display.insert(tk.END, " ".join(cmd))
            # Keep text widget editable - removed the disabled state
    
    def on_closing(self):
        """Handle window close event"""
        # Save settings on exit
        self.save_current_settings()
        self.root.destroy()


def main():
    """Main entry point for the GUI application"""
    # Use TkinterDnD.Tk for drag & drop support
    if DRAG_DROP_SUPPORT:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = DJMusicAnalyzerGUI(root)
    
    # Set window icon if available
    try:
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
        if os.path.exists(icon_path):
            img = tk.PhotoImage(file=icon_path)
            root.iconphoto(True, img)
    except Exception:
        pass
    
    # Center window on screen
    window_width = 800
    window_height = 700
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = int((screen_width / 2) - (window_width / 2))
    y = int((screen_height / 2) - (window_height / 2))
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()
