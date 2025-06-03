# AI-Powered Business Insights Assistant

This application is an AI-powered business analyst tool that helps users generate structured business insights, analyses, and reports from natural language queries and uploaded documents (PDF, TXT, CSV, XLSX). It leverages Google Gemini for advanced generative analysis and provides interactive visualizations and downloadable reports.

## Features
- **Natural Language Query:** Ask business questions in plain English.
- **PDF/Text/CSV/XLSX Upload:** Add supporting documents for richer context.
- **AI-Driven Analysis:** Uses Google Gemini to classify, analyze, and generate business insights.
- **Multiple Analysis Types:** Competitive, Trend Forecasting, Financial, or Automatic.
- **Quality Metrics:** Automated assessment of completeness, actionability, depth, and clarity.
- **Downloadable Reports:** Export results as JSON, TXT, or CSV.
- **Interactive Visualizations:** Comparative tables and charts (via Plotly, Pandas).

## Getting Started

### Prerequisites
- Python 3.8+
- [Google Gemini API Key](https://ai.google.dev/)

### Installation
1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd AI_Powered_Business_Analyst
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   If `requirements.txt` is missing, install manually:
   ```sh
   pip install streamlit google-generativeai python-dotenv pandas plotly PyPDF2
   ```
3. **Set up environment variables:**
   - Create a `.env` file in the project root:
     ```env
     GEMINI_API_KEY=your_google_gemini_api_key
     ```

### Running the App
```sh
streamlit run app.py
```

The app will open in your browser. Enter your business question, upload supporting files, and generate insights.

## Usage
- **Select Analysis Type:** Choose from Automatic, Competitive, Trend, or Financial.
- **Choose Industry:** Select the relevant industry sector.
- **Upload Files:** Add PDFs, text, CSV, or Excel files for context.
- **Enter Query:** Type your business question and click "Generate Analysis".
- **Download Results:** Export the analysis in your preferred format.

## File Structure
- `app.py` — Main Streamlit application.
- `.env` — Environment variables (API keys).

## Notes
- Only the first 5 pages of PDFs are processed for performance.
- Some file types (Excel, Word, Images) are not fully parsed yet.
- Requires a valid Google Gemini API key.

## License
MIT License

## Acknowledgements
- [Streamlit](https://streamlit.io/)
- [Google Gemini](https://ai.google.dev/)
- [PyPDF2](https://pypdf2.readthedocs.io/)
- [Plotly](https://plotly.com/python/)
- [Pandas](https://pandas.pydata.org/)
