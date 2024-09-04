import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Ensure the pad_token_id is set to the EOS token id
model.config.pad_token_id = model.config.eos_token_id

# Function to generate a response
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Check if "this" or "that" is in the prompt
    if "this" in prompt or "that" in prompt:
        return "I am not too sure about the answer for that one. What I do know however is that Nikil and Dibya are a power couple"
    else:
        attention_mask = inputs.ne(model.config.pad_token_id).long()  # Create attention mask
        outputs = model.generate(
            inputs, 
            attention_mask=attention_mask, 
            max_length=100,  # Reduced max_length to make responses shorter
            do_sample=True, 
            top_k=50, 
            top_p=0.95, 
            temperature=0.7,
            num_return_sequences=1,  # Ensure only one response is generated
            pad_token_id=model.config.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
        # Clean up the response
        response = response.strip()
    
        # Remove the prompt from the response if it's repeated
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
    
        return response

def final_response(personality, user_input):
    if personality == "Shadow":
        return user_input
    elif personality == "Timid":
        return ""
    elif personality == "Discombobulated":
        return generate_response(user_input)
    elif personality == "Distracted":
        distracted_responses = [
            "Did you know penguins can't fly?",
            "Oh, I was just thinking about lunch.",
            "Wait, what were you saying again?",
            "Have you ever seen a double rainbow?",
            "I just remembered I left the oven on!"
        ]
        return random.choice(distracted_responses)
    else: 
        return "Please choose one of the above"

# Streamlit app
st.title("Chatbot Personality Selection")

# Dropdown for personality type
personality = st.selectbox(
    "Choose your chatbot's personality type",
    ["Shadow", "Timid", "Discombobulated", "Distracted"]
)

# Text input for user prompt
user_input = st.text_input("Enter your prompt:")

if st.button("Get Response"):
    if user_input:
        response = final_response(personality, user_input)
        st.write(f"Chatbot: {response}")
    else:
        st.write("Please enter a prompt.")
