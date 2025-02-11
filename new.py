import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
import PyPDF2 

# Add an api from Groq website 
api_key = "" 

# Initialize the output parser
parser = StrOutputParser()

# Path to the resume PDF file
pdf_path = r""

# Extracting text from the PDF
with open(pdf_path, "rb") as file:
    reader = PyPDF2.PdfReader(file)  # Load the PDF
    text = ""
    for page in reader.pages:  # Iterate through all pages
        text += page.extract_text() + "\n"  # Extract text and append it



# Define the JSON template to structure extracted resume information
json_content = """{{
    "name": "",
    "email": "",
    "phone_1": "",
    "phone_2": "",
    "address": "",
    "city": "",
    "linkedin": "",
    "professional_experience_in_years": "",
    "highest_education": "",
    "is_fresher": "yes/no",
    "is_student": "yes/no",
    "skills": ["", ""],
    "applied_for_profile": "",
    "education": [
        {{
            "institute_name": "",
            "year_of_passing": "",
            "score": ""
        }},
        {{
            "institute_name": "",
            "year_of_passing": "",
            "score": ""
        }}
    ],
    "professional_experience": [
        {{
            "organisation_name": "",
            "duration": "",
            "profile": ""
        }},
        {{
            "organisation_name": "",
            "duration": "",
            "profile": ""
        }}
    ]
}}"""


messages = [
    SystemMessage(content=f"""Extract relevant information from the following resume text and fill the provided JSON template. 
    Ensure all keys in the template are present in the output, even if the value is empty or unknown. If a specific piece of information is not found, use 'Not provided' as the value.

    Resume text:
    provided by the user 

    JSON template:
    {json_content}

    Instructions:
    1. Carefully analyze the resume text.
    2. Extract relevant information for each field in the JSON template.
    3. If a piece of information is not explicitly stated, make a reasonable inference based on the context.
    4. Ensure all keys from the template are present in the output JSON.
    5. Format the output as a valid JSON string.

    Output the filled JSON template only, WITHOUT ANY ADDITIONAL TEXT OR EXPLANATIONS.
    """),
    HumanMessage(content=f"{text}") 
]

llm = ChatGroq(
    groq_api_key=api_key,                                                
    model="llama3-70b-8192"
)


result = llm.invoke(messages)
result = parser.invoke(result)

with open("output.txt", "w") as file:
    file.write(result)
