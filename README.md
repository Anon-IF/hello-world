# hello-world

## Gemini Language Model Client

A simple Python client for calling Google's Gemini language model via the [`google-genai`](https://pypi.org/project/google-genai/) SDK.

### Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Get an API key**

   Obtain a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

3. **Set the environment variable**

   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

### Usage

**Pass a prompt as a command-line argument:**

```bash
python gemini_client.py "Explain quantum computing in simple terms"
```

**Run in interactive mode (no arguments):**

```bash
python gemini_client.py
```

### Running Tests

```bash
python -m pytest test_gemini_client.py -v
```

### API

`call_gemini(prompt, model_name="gemini-2.0-flash") -> str`

| Parameter    | Type  | Description                                    |
|--------------|-------|------------------------------------------------|
| `prompt`     | `str` | The text prompt to send to the model.          |
| `model_name` | `str` | Gemini model to use (default: `gemini-2.0-flash`). |
