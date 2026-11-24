# AI Video Assistant

AI Video Assistant converts a YouTube video or local media file into a transcript, summary, title, meeting insights, and a searchable knowledge base. It provides a Streamlit web interface and a command-line interface.

## Live : https://rag-video.streamlit.app

## Features

- Download and process YouTube audio
- Upload or process MP4, MOV, MKV, MP3, WAV, and M4A files
- Transcribe English audio locally with OpenAI Whisper
- Transcribe and translate Hinglish audio with Sarvam AI
- Generate a concise title and meeting summary with Mistral AI
- Extract action items, key decisions, and open questions
- Ask questions about the transcript through a ChromaDB RAG pipeline
- Download the full transcript or a Markdown analysis report

## Technology

- **UI:** Streamlit
- **Audio:** yt-dlp, FFmpeg, pydub
- **Speech-to-text:** OpenAI Whisper and Sarvam AI
- **LLM:** Mistral AI through LangChain
- **RAG:** LangChain, ChromaDB, and Hugging Face sentence embeddings
- **Export:** Plain text and Markdown

## Project Structure

```text
AI-Video-Assistant/
├── app.py                 # Streamlit web application
├── main.py                # Command-line application
├── test.py                # Example pipeline script
├── requirements.txt       # Python dependencies
├── runtime.txt            # Deployment Python runtime
├── core/
│   ├── transcriber.py     # Whisper and Sarvam transcription
│   ├── summarize.py       # Title and summary generation
│   ├── extractor.py       # Action items, decisions and questions
│   ├── rag_engine.py      # RAG prompt and retrieval chain
│   └── vector_store.py    # ChromaDB vector store
└── utils/
    └── audio_processor.py # Downloading, conversion, and chunking
```

## Requirements

- Python 3.12
- FFmpeg available on `PATH`
- A Mistral AI API key
- A Sarvam AI API key only when using Hinglish transcription

Internet access is required for YouTube downloads, API-based features, and the first download of Whisper or Hugging Face models.

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/SoyebAhmed5/AI-Video-Assistant.git
cd AI-Video-Assistant
```

### 2. Create and activate a virtual environment

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install FFmpeg

Verify that FFmpeg is installed:

```powershell
ffmpeg -version
```

On Windows, FFmpeg can be installed with [winget](https://learn.microsoft.com/windows/package-manager/winget/):

```powershell
winget install --id Gyan.FFmpeg -e
```

Restart the terminal after installation so the updated `PATH` is available.

## Configuration

Create a `.env` file in the project root:

```dotenv
MISTRAL_API_KEY=your_mistral_api_key
WHISPER_MODEL=small
SARVAM_API_KEY=your_sarvam_api_key
SARVAM_STT_MODEL=saaras:v2.5
```

| Variable | Required | Description |
| --- | --- | --- |
| `MISTRAL_API_KEY` | Yes | Used for summarization, insight extraction, title generation, and RAG answers |
| `WHISPER_MODEL` | No | Whisper model size; defaults to `small` |
| `SARVAM_API_KEY` | For Hinglish | Used with the Hinglish transcription option |
| `SARVAM_STT_MODEL` | No | Sarvam speech-to-text model; defaults to `saaras:v2.5` |

Keep `.env` out of source control. Never commit API keys or credentials.

## Running the Web Application

Start Streamlit from the project root:

```powershell
streamlit run app.py
```

Open the local URL shown by Streamlit, then:

1. Paste a YouTube URL or local file path, or upload media.
2. Select `english` or `hinglish`.
3. Click **Analyze video**.
4. Review the title, summary, action items, decisions, questions, and transcript.
5. Download an output or ask questions in the chat panel.

## Running from the Terminal

```powershell
python main.py
```

Enter a YouTube URL or local media path when prompted. Choose `english` or `hinglish`, then review the generated analysis and use the interactive RAG chat.

## Whisper Models

The default Whisper model is `small`. You can change it in `.env`:

```dotenv
WHISPER_MODEL=base
```

Common options include `tiny`, `base`, `small`, `medium`, and `large`. Larger models may produce better transcripts but require more memory and disk space.

## Processing Notes

- Audio is converted to mono 16 kHz WAV files.
- Audio is split into chunks of 10 minutes.
- English transcription uses local Whisper and does not require a Sarvam key.
- Hinglish transcription uses the Sarvam speech-to-text translation API.
- The RAG index is stored locally in `vector_db/`.
- Generated `downloads/`, temporary WAV chunks, and vector database data should not be committed.

## Troubleshooting

**`ffmpeg` is not recognized**

Install FFmpeg and ensure its `bin` directory is in the Windows `PATH`.

**Missing API key error**

Confirm that `.env` exists in the project root and contains `MISTRAL_API_KEY`. Add `SARVAM_API_KEY` when using Hinglish.

**YouTube download fails**

Update yt-dlp and verify that the URL is public and accessible:

```powershell
pip install --upgrade yt-dlp
```

**Whisper is slow or runs out of memory**

Use a smaller model such as `tiny` or `base`, close other applications, or use shorter media files.

**Streamlit port is already in use**

Start the app on another port:

```powershell
streamlit run app.py --server.port 8502
```