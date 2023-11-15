import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

#Function to get the response back
def getLLMResponse(form_input,skills,experience,statement_style):

    # Wrapper for Llama-2-7B-Chat, Running Llama 2 on CPU

    #Quantization is reducing model precision by converting weights from 16-bit floats to 8-bit integers, 
    #enabling efficient deployment on resource-limited devices, reducing model size, and maintaining performance.

    #C Transformers offers support for various open-source models, 
    #among them popular ones like Llama, GPT4All-J, MPT, and Falcon.


    #C Transformers is the Python library that provides bindings for transformer models implemented in C/C++ using the GGML library

    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',     #https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
                    model_type='llama',
                    config={'max_new_tokens': 55,
                            'temperature': 0.01})
    
    
    #Template for building the PROMPT
    #{project_role} {skills} {experience} {style}
    template = """
    Summarize the following skills and experience to provide a statement of interest for a role as  a {project_role}.\n Skills: {skills}\n Experience: {experience}\n\n Constraints: Keep the statement of interest to under 250 words and use a {style} style.
    """

    #Creating the final PROMPT
    prompt = PromptTemplate(
    input_variables=["style","project_role","skills","experience"],
    template=template,)

  
    #Generating the response using LLM
    response=llm(prompt.format(project_role=form_input,skills=skills,experience=experience,style=statement_style))
    print(response)

    return response


st.set_page_config(page_title="Generate Statement of Interest",
                    page_icon='🤝',
                    layout='centered',
                    initial_sidebar_state='collapsed')
st.header("Generate Statement of Interest 🤝")

form_input = st.text_area('Enter the project role', height=50)

#Creating columns for the UI - To receive inputs from user
col1, col2, col3 = st.columns([10, 10, 5])
with col1:
    skills = st.text_input('Top 3 Skills')
with col2:
    experience = st.text_input('Relevant Experience')
with col3:
    statement_style = st.selectbox('Writing Style',
                                    ('Neutral', 'Excited', 'Braggadocious'),
                                       index=0)


submit = st.button("Generate")

#When 'Generate' button is clicked, execute the below code
if submit:
    st.write(getLLMResponse(form_input,skills,experience,statement_style))
