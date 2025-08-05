import os

def merge_py_files_for_llm(file_list, output_file, include_headers=True, prompt_string=""):
    """
    Merges specified .py files into a single text file with clear separators.

    Parameters:
    - file_list (list of str): List of file paths to .py files.
    - output_file (str): Path to the output .txt file.
    - include_headers (bool): Whether to include file headers as separators.
    - prompt_string (str): String to be added at the end of the output file.
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filepath in file_list:
            if os.path.isfile(filepath) and filepath.endswith('.py'):
                #filename = os.path.basename(filepath)
                if include_headers:
                    separator = f"\n\n# =====  File: {filepath} =====\n\n"
                    outfile.write(separator)
                with open(filepath, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    # Optionally, wrap code in markdown code blocks for better formatting
                    code_block = f"```python\n{content}\n```"
                    outfile.write(code_block)
                    outfile.write('\n')  # Add a newline for separation
        if prompt_string:
            outfile.write('\n' + prompt_string)
    print(f"Specified .py files have been merged into {output_file} with clear separators for LLMs.")

if __name__ == "__main__":
    # List of file paths to include
    file_list = [
        'nexus/abc.py',
        'nexus/asset.py',
        'nexus/event.py',
        'nexus/execution.py',
        'nexus/position.py',
        'nexus/portfolio.py',
        'nexus/backtest.py',
        'nexus/helpers.py',
        'nexus/performance.py',
        'nexus/indicators.py',
        'feed/t6.py',
        #'strategy/ma_cross.py',
        #'backtests/ma_cross.py'
        'blackbook/alice1a.py'
        # Add more file paths as needed
    ]

    # Specify the output file path
    output_txt_file = 'merged_files_for_llm.txt'

    # Prompt string to add at the end
    prompt_string = "This is the actual source code of algorithmic event trading framework codenamed NEXUS."

    # Call the merge function
    merge_py_files_for_llm(file_list, output_txt_file, include_headers=True, prompt_string=prompt_string)
