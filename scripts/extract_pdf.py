
import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import pypdf
except ImportError:
    print("pypdf not found, installing...")
    try:
        install("pypdf")
        import pypdf
    except Exception as e:
        print(f"Failed to install pypdf: {e}")
        sys.exit(1)

def extract_text(pdf_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        # Read only the first 10 pages or so, assuming BAB 3 is within reach or read all?
        # It's a "BAB 3" file, so likely just that chapter.
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    pdf_path = r"d:\Data Main\Portfolio_afnan\CNN_Batik_Parang\Muhammad Afnan Risandi BAB 3_2301021.pdf"
    content = extract_text(pdf_path)
    # Print the content to stdout so the agent can read it
    print(content)
