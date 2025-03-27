# md_utils.py

import os
import pyttsx3

def format_filename(filename):
    """Format the filename to uppercase with spaces."""
    name_without_extension = os.path.splitext(filename)[0]
    formatted_name = name_without_extension.replace('_', ' ').replace('-', ' ').upper()
    return formatted_name

def get_md_files(directory='./'):
    """Get all markdown (.md) files in the specified directory."""
    md_files = [f for f in os.listdir(directory) if f.endswith('.md')]
    return md_files

def read_file_content(file_path):
    """Read the contents of a markdown file and filter to display numbers and letters."""
    with open(file_path, 'r') as file:
        content = file.read()

    # Filtering content to only show numbers and letters
    filtered_content = ''.join(char for char in content if char.isalnum() or char.isspace())
    return filtered_content

def text_to_speech(text):
    """Convert the provided text to speech."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def display_md_content(file, directory='./'):
    """Display the .md file's content and TTS."""
    print(f"\nDisplaying content of: {file}")
    content = read_file_content(os.path.join(directory, file))
    print("\nContent (Numbers and Letters only):\n")
    print(content)
    text_to_speech(content)  # Play TTS for the content
