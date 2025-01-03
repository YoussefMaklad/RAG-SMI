# from pathlib import Path
# import logging

# logging.basicConfig(level=logging.DEBUG, format='%(message)s', filemode='a', filename='evaluation.log')

# pdf_path = str(Path(__file__).parent.parent.parent) + '\knowledge\\rfc2326.pdf'
# print(pdf_path, type(pdf_path))

# pdf_path = str(Path(__file__).parent.parent.parent / 'knowledge/rfc2326.pdf')
# print(pdf_path, type(pdf_path))


# logging.debug("Hello there test")
# with open('../../knowledge/rfc2326.pdf', 'r') as f:
#     rfc = f.read()
#     print(rfc[:50000])

from crewai_tools import PDFSearchTool
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

tool = PDFSearchTool(
    pdf="E:\\Seventh Semester\\Grad\\LLM-Evaluation\\rag_evaluation\\knowledge\\rfc2326.pdf",
    config=dict(
        llm=dict(
            provider="ollama",
            config=dict(
                model="qwen2.5:3b",
                temperature=0.0,
                # top_p=1,
                stream=True,
            ),
        ),
    )
)

print(tool.run("Server State Machine"))
