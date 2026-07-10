import streamlit as st
import os
from openai import OpenAI

# Page Config
st.set_page_config(
    page_title="Custom AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom header style
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
</style>
<div class="header-box">
    <h2>🤖 Project 9: Custom AI Assistant</h2>
    <p>Build a ChatGPT-like custom assistant powered by OpenAI GPT models. Configure custom personalities, adjust settings, and paste your API key to test.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.header("Configuration & API Key")

# Check for API key in environment or secrets
env_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

# API key input
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=env_key,
    type="password",
    placeholder="sk-..."
)

st.sidebar.write("---")
st.sidebar.header("Assistant Persona")

persona = st.sidebar.selectbox("Choose AI Personality", [
    "Data Scientist & AI Mentor",
    "Python Coding Expert",
    "Sarcastic Coding Buddy",
    "Creative Writer & Marketer",
    "Standard General Assistant"
])

model_choice = st.sidebar.selectbox("GPT Model", ["gpt-4o-mini", "gpt-4o"])
temperature = st.sidebar.slider("Creativity (Temperature)", 0.0, 2.0, 0.7, 0.1)
max_tokens = st.sidebar.slider("Max Response Length", 100, 2000, 800, 50)

# Clear chat history button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Map persona selection to system prompt
personas_map = {
    "Data Scientist & AI Mentor": "You are a senior data scientist and AI mentor. Explain concepts clearly, write concise code snippets, and teach best practices.",
    "Python Coding Expert": "You are an expert Python engineer. Provide optimal, bug-free, well-commented Python code solutions. Keep explanations straight to the point.",
    "Sarcastic Coding Buddy": "You are a witty, sarcastic coding partner. You make fun of syntax errors, give solid advice with humor, and use emojis frequently.",
    "Creative Writer & Marketer": "You are a professional marketer and creative writer. Focus on catchy headings, structured copies, and engaging story-telling.",
    "Standard General Assistant": "You are a helpful, direct, and polite AI chatbot."
}

system_prompt = personas_map[persona]

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check if API key is provided
if not openai_api_key:
    st.warning("⚠️ OpenAI API Key is missing. Please enter your API key in the sidebar to start chatting.")
    st.info("💡 Tip: To automate this when deploying on Streamlit Cloud, add your key to **Secrets** as `OPENAI_API_KEY = \"your-key\"`.")
else:
    # Accept user input
    if prompt := st.chat_input("Say something to your assistant..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Initialize OpenAI client using the user's provided API key variable
                client = OpenAI(api_key=openai_api_key)
                
                # Fetch chat completion
                response = client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *st.session_state.messages
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Render response
                ai_response = response.choices[0].message.content
                message_placeholder.markdown(ai_response)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                message_placeholder.error(f"Error calling OpenAI API: {e}")
