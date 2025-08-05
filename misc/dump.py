import os

def merge_py_files_for_llm(directory, output_file, include_headers=True):
    """
    Merges all .py files in the specified directory into a single text file with clear separators.

    Parameters:
    - directory (str): Path to the directory containing .py files.
    - output_file (str): Path to the output .txt file.
    - include_headers (bool): Whether to include file headers as separators.
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in sorted(os.listdir(directory)):
            if filename.endswith('.py'):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    if include_headers:
                        # Enhanced Separator for LLMs
                        separator = (
                            f"\n\n---\n"
                            f"# File: {filename}\n"
                            f"---\n\n"
                        )
                        outfile.write(separator)
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        # Optionally, wrap code in markdown code blocks for better formatting
                        code_block = f"```python\n{content}\n```"
                        outfile.write(code_block)
                        outfile.write('\n')  # Add a newline for separation
    print(f"All .py files have been merged into {output_file} with clear separators for LLMs.")

if __name__ == "__main__":
    # Specify the directory containing .py files
    target_directory = 'nexus'  # Current directory; change as needed, e.g., '/path/to/directory'

    # Specify the output file path
    output_txt_file = 'merged_files_for_llm.txt'

    # Call the merge function
    merge_py_files_for_llm(target_directory, output_txt_file, include_headers=True)

