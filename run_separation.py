import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Add ultimatevocalremovergui to Python path
uvg_path = project_root / 'ultimatevocalremovergui'
if str(uvg_path) not in sys.path:
    sys.path.append(str(uvg_path))

from process_audio import AudioProcessor

def main():
    parser = argparse.ArgumentParser(description='Process audio files (wav, mp3, m4a, flac) for vocal separation and enhancement.')
    parser.add_argument('input_dir', help='Directory containing input audio files (wav, mp3, m4a, flac).')
    parser.add_argument('output_dir', help='Directory for output files')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    # Get list of input files
    supported_extensions = ["*.wav", "*.mp3", "*.m4a", "*.flac"]
    input_files = []
    for ext in supported_extensions:
        input_files.extend(Path(input_dir).glob(ext))

    if not input_files:
        print(f"No supported audio files found in {input_dir}. Supported formats: wav, mp3, m4a, flac.")
        return
    
    print(f"\nFound {len(input_files)} audio files in {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process all files
    processor = AudioProcessor()
    processor.process_audio_files(input_files, output_dir)
    
    print("\nProcessing complete!")
    print("\nOutput directories:")
    print(f"- Instrumentals: {os.path.join(output_dir, 'instrumentals')}")
    print(f"- Initial Vocals: {os.path.join(output_dir, 'intermediate_vocals')}")
    print(f"- HP-Karaoke Vocals: {os.path.join(output_dir, 'hp_vocals')}")
    print(f"- Final De-echoed Vocals: {os.path.join(output_dir, 'final_vocals')}")

if __name__ == "__main__":
    main() 