# PyInstaller spec file for DJ Music Cleaner GUI
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Add additional hidden imports
hidden_imports = [
    'tkinter',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'tkinter.ttk',
    'json',
    'subprocess',
    'threading',
    'pathlib',
    'os',
    'sys',
    'platform',
    'datetime',
    're'
]

a = Analysis(
    ['gui_app/app.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # Changed to onedir mode
    name='DJ Music Cleaner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,  # Changed to False
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='gui_app/resources/icon.ico'  # Icon file path
)

# Add binaries and data files in onedir mode
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DJ Music Cleaner',
)

# For macOS, create a .app bundle
app = BUNDLE(
    coll,  # Use the collection instead of exe for onedir mode
    name='DJ Music Cleaner.app',
    icon='gui_app/resources/icon.icns',  # Icon file path for macOS
    bundle_identifier='com.djmusiccleaner.gui',
    info_plist={
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
        'NSHighResolutionCapable': 'True',
        'LSBackgroundOnly': 'False',
        'NSRequiresAquaSystemAppearance': 'False',  # For dark mode support
        'LSEnvironment': {'PYTHONHASHSEED': '1'},  # Prevent recursion issues
        'LSMultipleInstancesProhibited': 'True',  # Prevent multiple instances
    },
)
