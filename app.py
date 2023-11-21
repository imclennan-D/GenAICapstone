import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import PyPDF2 

#Function to get the response back
def getLLMResponse(form_input,resume_content,statement_style):
    print("Initializing LLM\n")
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',     #https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
                    model_type='llama',
                    config={'max_new_tokens': 200,
                            'temperature': 0.01,
                            'context_length': 2000},)
    
    #Template for building the PROMPT
    #(form_input,resume_content, statement_style)
    template = """Please write a personal statement tailored to the position described below, using the candidate's resume (listed below) to identify relevant skills and experiences in 2-4 sentences. Maintain a tone consistent with a {statement_style} style. Return only the personal statement.

Job Description: {form_input}

Resume: {resume_content}

Personal Statement:"""

    #Creating the final PROMPT
    prompt = PromptTemplate(
    input_variables=["style","form_input","resume_content"],
    template=template,)

    #Generating the response using LLM
    print("Generating response\n(This may take a while...)\n")
    response=llm(prompt.format(form_input=form_input, resume_content=resume_content, statement_style=statement_style))
    print(response)

    return response

print("Starting Streamlit Construction\n---\n")
st.set_page_config(page_title="Generate Statement of Interest",
                    page_icon='🤝',
                    layout='centered',
                    initial_sidebar_state='collapsed')
st.header("Generate Statement of Interest 🤝")

form_input = st.text_area('Enter the project description', height=50)

uploaded_file = st.file_uploader("Choose a file", type="pdf")

statement_style = st.selectbox('Writing Style',
                                    ('Neutral', 'Excited', 'Braggadocious'),
                                       index=0)

submit = st.button("Generate")

# When 'Generate' button is clicked, execute the below code
if submit:
    if uploaded_file is not None:
                # Read the PDF file
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        # Extract the content
        resume_content = ""
        for page in range(len(pdf_reader.pages)):
            resume_content += pdf_reader.pages[page].extract_text()
        print("Document Read\n")
        # Display the content (for bug testing)
        #st.write(resume_content)
        #st.write(print("Starting LLM Response"))
        st.write(getLLMResponse(form_input,resume_content, statement_style))