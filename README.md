## Installation Guide

Follow these steps to install TA-Lib and necessary dependencies for the project:

### 1. Install TA-Lib

1. **Download TA-Lib**:
   - Download the `ta-lib x64.zip` file from the [TA-Lib GitHub Repository](https://github.com/afnhsn/TA-Lib_x64).
   - Extract the zip file to `C:\`, so it looks like this: `C:\ta-lib`.

2. **Install Visual C++ Build Tools**:
   - Download Visual C++ Build Tools 2022 from [here](https://aka.ms/vs/17/release/vs_buildtools.exe).

3. **Install Visual C++ Build Tools**:
   - Follow the instructions in this [Stack Overflow article](https://stackoverflow.com/a/54136652/10997732) to install Visual C++ Build Tools.

4. **Install TA-Lib**:
   - Run the following command to install TA-Lib using pip:
     ```bash
     pip install ta-lib
     ```

5. **Manual Installation (if needed)**:
   - If the above installation fails, download the appropriate `.whl` file from [this link](https://github.com/cgohlke/talib-build/releases).
   - Run the command below in the command prompt (adjust the path as necessary):
     ```bash
     "C:\Users\singh\AppData\Local\Programs\Python\Python310\python.exe" -m pip install C:\Users\singh\Downloads\TA_Lib-0.4.32-cp310-cp310-win_amd64.whl
     ```
   - Ensure that the version of Python and the `.whl` file are the same.

### 2. Install Other Dependencies

To install the necessary dependencies for the project, run the following command:
```bash
pip install pandas numpy yfinance matplotlib
```
verify by running  the below command
```bash
python -c "import talib; import pandas; import yfinance; import numpy; import matplotlib
```

### Additional Tips:
- Make sure to update the paths and Python version as necessary based on your specific environment.
- You might want to add sections for usage examples or additional configuration if relevant to your project. 

Feel free to modify any part of the instructions to better fit your project or audience!


