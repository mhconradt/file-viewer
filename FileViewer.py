import streamlit as st
import plotly.express as px
import pandas as pd
from streamlit_plotly_events import plotly_events
import subprocess
import os
import tiktoken
import openai
from tqdm import tqdm
import numpy as np
import umap
import pickle
from pathlib import Path

st.set_page_config(layout="wide")

st.title("Git Repository Viewer")
# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'current_directory' not in st.session_state:
    st.session_state.current_directory = None

# Load embedding cache if exists
cache_file = 'embedding_cache.pkl'
if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        embedding_cache = pickle.load(f)
else:
    embedding_cache = {}

# Define variables
max_tokens = 8192  # Max token limit for embeddings
embedding_model = "text-embedding-3-small"
encoding = tiktoken.encoding_for_model(embedding_model)

# Function to get embedding
def get_embedding(content):
    try:
        response = openai.embeddings.create(input=content, model=embedding_model)
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

language_map = {
    '.py': ('python', 'blue'),
    '.js': ('javascript', 'orange'),
    '.html': ('html', 'red'),
    '.css': ('css', 'green'),
    '.md': ('markdown', 'purple'),
    '.txt': ('text', 'gray'),
    '.json': ('json', 'brown'),
    '.xml': ('xml', 'pink'),
    '.yml': ('yaml', 'yellow'),
    '.yaml': ('yaml', 'yellow'),
    '.sql': ('sql', 'cyan'),
    '.sh': ('shell', 'black'),
    '.cpp': ('cpp', 'violet'),
    '.h': ('cpp', 'violet'),
    '.c': ('c', 'teal'),
    '.java': ('java', 'magenta'),
    '.rb': ('ruby', 'maroon'),
    '.go': ('go', 'lightblue'),
    '.rs': ('rust', 'darkorange'),
    '.ts': ('typescript', 'lightgreen'),
    '.php': ('php', 'lightpurple'),
    '.scala': ('scala', 'darkred'),
    '.kt': ('kotlin', 'darkblue'),
    '.swift': ('swift', 'lightred'),
    '.m': ('matlab', 'olive'),
    '.r': ('r', 'darkgreen'),
    '.jl': ('julia', 'lightpink')
}

# Function to process directory and generate dataset
def process_directory(directory):
    git_ls_files = subprocess.run(['git', 'ls-files'], stdout=subprocess.PIPE, text=True, cwd=directory)
    file_list = git_ls_files.stdout.strip().split('\n')

    st.write(f"Found {len(file_list)} tracked files.")

    # Display progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    embeddings_list = []
    filenames_list = []
    extensions_list = []

    for idx, filename in enumerate(tqdm(file_list)):
        # Update progress
        progress = (idx + 1) / len(file_list)
        progress_bar.progress(progress)
        status_text.text(f"Processing file {idx + 1}/{len(file_list)}: {filename}")
        filepath = directory / filename

        if filepath in embedding_cache:
            embedding = embedding_cache[filepath]
        else:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                tokens = encoding.encode(content)
                if len(tokens) > max_tokens:
                    st.write(f"Skipping {filename}, too long ({len(tokens)} tokens).")
                    continue
                embedding = get_embedding(content)
                if embedding:
                    embedding_cache[filepath] = embedding
            except Exception as e:
                st.write(f"Could not process {filename}: {e}")
                continue

        if embedding:
            embeddings_list.append(embedding)
            filenames_list.append(str(filepath.relative_to(directory)))
            extensions_list.append(filepath.suffix)

    # Save cache
    with open(cache_file, 'wb') as f:
        pickle.dump(embedding_cache, f)

    progress_bar.progress(1.0)
    status_text.text("Embedding generation completed.")

    # Convert embeddings to numpy array
    embedding_array = np.array(embeddings_list)

    print(locals())

    # Reduce embeddings to 3D using UMAP
    reducer = umap.UMAP(n_components=3, random_state=42)
    embedding_3d = reducer.fit_transform(embedding_array)

    # Create DataFrame
    df = pd.DataFrame({
        'x': embedding_3d[:, 0],
        'y': embedding_3d[:, 1],
        'z': embedding_3d[:, 2],
        'filename': filenames_list,
        'extension': extensions_list,
    })

    df['language'] = df['extension'].map(lambda ext: language_map.get(ext, ('text', 'lightgray'))[0])
    df['color'] = df['extension'].map(lambda ext: language_map.get(ext, ('text', 'lightgray'))[1])

    return df

# Get directory input
directory = Path(st.text_input(label="Git repository", value=str(Path.cwd())))

# Check if directory has changed
if st.session_state.current_directory != directory:
    st.session_state.current_directory = directory
    st.session_state.processed_data = process_directory(directory)

# Create Plotly 3D scatter plot
fig = px.scatter_3d(
    st.session_state.processed_data,
    x='x',
    y='y',
    z='z',
    hover_name='filename',
        hover_data={'x': False, 'y': False, 'z': False, 'extension': False},

    custom_data=['filename']
)
fig.update_traces(marker=dict(size=5))

# Display the plot and capture click events

col1, col2 = st.columns([3, 2])

with col1:
    selected_points = plotly_events(
    fig,
    click_event=True,
    hover_event=False,
    select_event=False,
    override_height=800
    )

# Comprehensive file extension to language mapping
language_map = {
    '.py': 'python', '.js': 'javascript', '.html': 'html', '.css': 'css',
    '.md': 'markdown', '.txt': 'text', '.json': 'json', '.xml': 'xml',
    '.yml': 'yaml', '.yaml': 'yaml', '.sql': 'sql', '.sh': 'shell',
    '.cpp': 'cpp', '.h': 'cpp', '.c': 'c', '.java': 'java',
    '.rb': 'ruby', '.go': 'go', '.rs': 'rust', '.ts': 'typescript',
    '.php': 'php', '.scala': 'scala', '.kt': 'kotlin', '.swift': 'swift',
    '.m': 'matlab', '.r': 'r', '.jl': 'julia'
}

# Handle the click event
with col2:
    if selected_points:
        clicked_point = selected_points[0]
        filename = st.session_state.processed_data.iloc[clicked_point['pointNumber']].filename
        st.write(f"You clicked on: {filename}")

        # Display file content within Streamlit
        try:
            with open(directory / filename, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            # Determine the code language based on file extension
            file_extension = os.path.splitext(filename)[1].lower()
            language = language_map.get(file_extension, 'text')
            st.code(content, language=language)
        except Exception as e:
            st.error(f"Could not open {filename}: {e}")
