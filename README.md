# AI-FitTrainer

AI-FitTrainer is a real-time personal fitness assistant that analyzes squat technique and provides feedback to help users improve their form.

## Installation

1. Clone this repository: `git clone https://github.com/OumaymaRedissi/AIFitTrainer`
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`

## Adding OpenAI API Key

To use the AI-FitTrainer, you need to provide your OpenAI API key. Follow these steps:

1. Create a file named `.env` in the root directory of the project.
2. Inside the `.env` file, add the following line, replacing `your_actual_openai_api_key` with your real OpenAI API key:

OPENAI_API_KEY=your_actual_openai_api_key


3. Save the `.env` file.

## Usage

1. Run the Streamlit app: `streamlit run main.py`
2. Follow the on-screen instructions to upload a video for analysis.

## File Structure

- `main.py`: The main Streamlit application script.
- `fit_trainer/`: Package containing modularized modules for OpenAI, model, video processing, and UI.
- `env/`: Virtual environment directory.
- `assets/`: Folder containing assets like images.
- `img/`: Images used for the project.
- ...

## Contributing

Contributions are welcome! To contribute:

1. Fork this repository.
2. Create a new branch for your feature: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -m "Add new feature"`
4. Push your changes to your fork: `git push origin feature-name`
5. Create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
