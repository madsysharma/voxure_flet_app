import os
import shutil
import subprocess
import sys
import psutil

def clean_build_dirs():
    """Clean build and dist directories if they exist."""
    for dir_name in ['build', 'dist']:
        if os.path.exists(dir_name):
            print(f"Cleaning {dir_name} directory...")
            shutil.rmtree(dir_name)

def ensure_directories():
    """Ensure necessary directories exist."""
    os.makedirs('storage/icons', exist_ok=True)

def check_memory():
    """Check if there's enough memory available."""
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024 * 1024 * 1024)
    print(f"Available memory: {available_gb:.2f} GB")
    if available_gb < 2:  # Require at least 2GB free
        print("Warning: Low memory available. Consider closing other applications.")
        return False
    return True

def build_executable():
    """Build the executable using PyInstaller."""
    try:
        # Ensure we're in the project root directory
        project_root = os.path.abspath(os.path.dirname(__file__))
        os.chdir(project_root)
        print(f"Working directory: {os.getcwd()}")
        
        # Clean previous builds
        clean_build_dirs()
        
        # Ensure directories exist
        ensure_directories()
        
        # Check memory
        if not check_memory():
            print("Proceeding with build despite low memory...")
        
        # Build command with memory optimization flags
        cmd = [
            'pyinstaller',
            'main.spec',
            '--clean',
            '--noconfirm',
            '--log-level=INFO',
            '--workpath=build',
            '--distpath=dist',
            '--upx-dir=upx'  # Optional: specify UPX directory if you have it
        ]
        
        # Run PyInstaller with increased timeout
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            cwd=project_root  # Ensure we're in the correct directory
        )
        
        # Monitor the build process
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Get return code and any error output
        return_code = process.poll()
        _, stderr = process.communicate()
        
        if return_code != 0:
            print(f"Build failed with return code: {return_code}")
            print("Error output:")
            print(stderr)
            return False
            
        print("Build completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during build: {str(e)}")
        return False

if __name__ == "__main__":
    # Install required packages if not present
    try:
        import psutil
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
    
    # Run the build
    success = build_executable()
    sys.exit(0 if success else 1) 