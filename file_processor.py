import os
import shutil
import string
from typing import List, Dict, Any
import logging

log = logging.getLogger(__name__)

class FileProcessor:
    """Handles MD file processing and cleanup"""

    @staticmethod
    def clean_filename(filename: str) -> str:
        """Clean filename by removing unwanted characters"""
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        cleaned_name = ''.join(c for c in filename if c in valid_chars)
        return cleaned_name

    @staticmethod
    def sanitize_directory(directory_path: str) -> None:
        """Sanitize all filenames in the directory"""
        for root, dirs, files in os.walk(directory_path):
            # Clean directory names
            for dir_name in dirs:
                clean_dir_name = FileProcessor.clean_filename(dir_name)
                if dir_name != clean_dir_name:
                    old_path = os.path.join(root, dir_name)
                    new_path = os.path.join(root, clean_dir_name)
                    shutil.move(old_path, new_path)

            # Clean file names
            for filename in files:
                if filename.endswith('.md'):
                    clean_name = FileProcessor.clean_filename(filename)
                    if filename != clean_name:
                        old_path = os.path.join(root, filename)
                        new_path = os.path.join(root, clean_name)
                        shutil.move(old_path, new_path)

    @staticmethod
    def read_md_files(directory_path: str) -> List[Dict[str, Any]]:
        """Read all MD files from directory and subdirectories"""
        documents = []

        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                            # Create relative path from base directory
                            rel_path = os.path.relpath(file_path, directory_path)

                            # Create document with metadata
                            doc = {
                                "text": content,
                                "metadata": {
                                    "source": rel_path,
                                    "filename": file,
                                    "folder": os.path.basename(root)
                                }
                            }
                            documents.append(doc)
                    except Exception as e:
                        log.error(f"Error reading file {file_path}: {e}")

        return documents
