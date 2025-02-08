import os, time
import streamlit as st
import requests

VLLM_API_URL = "http://localhost:8000/v1/completions"

st.title("Medical Scribe")
st.write("Interact with our fine-tuned medical scribe")

default_prompt = """### Conversation:
Doctor: What brings you back into the clinic today, miss? 
Patient: I came in for a refill of my blood pressure medicine. 
Doctor: It looks like Doctor Kumar followed up with you last time regarding your hypertension, osteoarthritis, osteoporosis, hypothyroidism, allergic rhinitis and kidney stones.  Have you noticed any changes or do you have any concerns regarding these issues?  
Patient: No. 
Doctor: Have you had any fever or chills, cough, congestion, nausea, vomiting, chest pain, chest pressure?
Patient: No.  
Doctor: Great. Also, for our records, how old are you and what race do you identify yourself as?
Patient: I am seventy six years old and identify as a white female.

### Header:
General History

### Summary:
"""

user_input = st.text_area("Enter your prompt:", value=default_prompt, height=400)

with st.sidebar:
    llm_url = st.text_input("LLM URL:", value="http://localhost:8000")

    api_key = st.text_input("API Key:", value="scr1b3pt")

    model_name = st.text_input("LLM URL:", value="donaldrauscher/medical-scribe-vllm")

    max_tokens = st.number_input(label="Max tokens:", min_value=1, max_value=2048, value=1000)

    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

if st.button("Generate") and user_input:
    with st.spinner("Running..."):
        data = {
            "model": model_name,
            "prompt": user_input,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        start_time = time.time()
        response = requests.post(
            os.path.join(llm_url, 'v1/completions'), 
            json=data,
            headers={
                'Authorization': f'Bearer {api_key}'
            }
        )
        end_time = time.time()
        if response.status_code == 200:
            output = response.json()['choices'][0]['text']
            st.write(output)
            st.write(f"Response time: {end_time - start_time:.1f} seconds")
        else:
            st.write(response)
