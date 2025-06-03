# app.py - Business Insights Assistant with PDF Support
import streamlit as st
import google.generativeai as genai
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import os
from datetime import datetime
import pandas as pd
import plotly.express as px
import base64
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Initialize environment
load_dotenv()

# Configure Gemini with error handling
try:
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    model = genai.GenerativeModel('models/gemini-2.0-flash')
except Exception as e:
    st.error(f"Failed to configure Gemini: {str(e)}")
    st.stop()

# Define data structures
@dataclass
class BusinessQuery:
    raw_query: str
    query_type: str
    domain: str
    entities: List[str]
    time_frame: str
    complexity: str
    follow_up_questions: List[str] = None
    uploaded_files: List[Dict] = None
    conversation_history: list = None

class BusinessInsightsAssistant:
    def __init__(self):
        self.prompt_templates = self._load_prompt_templates()
        
    def _load_prompt_templates(self) -> Dict:
        return {
            'competitive_analysis': {
                'system_prompt': """You are a senior business strategist. Analyze {entities} in {domain} sector.\nProvide structured response with:\n1. Executive Summary\n2. Key Findings\n3. Comparative Analysis\n4. Recommendations\n5. Implementation Plan""",
                'user_prompt': """Compare {entities} in {domain} considering {time_frame} trends.\nFocus on these specific aspects: {specific_requirements}"""
            },
            'trend_forecasting': {
                'system_prompt': """As a market analyst, forecast {domain} trends for {time_frame}.\nProvide structured response with:\n1. Executive Summary  \n2. Key Trends\n3. Market Implications\n4. Strategic Recommendations\n5. Risk Assessment""",
                'user_prompt': """Analyze trends in {domain} for {time_frame}.\nFocus on these specific areas: {specific_requirements}"""
            },
            'financial_analysis': {
                'system_prompt': """You are a financial analyst. Perform a financial analysis for {entities} in the {domain} sector over {time_frame}.\nProvide structured response with:\n1. Executive Summary\n2. Key Financial Metrics\n3. Financial Comparison Table\n4. Financial Insights\n5. Recommendations\n6. Risk Assessment""",
                'user_prompt': """Analyze the financial performance of {entities} in {domain} for {time_frame}.\nFocus on these aspects: {specific_requirements}"""
            },
            'automatic': {
                'system_prompt': """You are an AI business insights assistant. Given the following query, select the most appropriate analysis type (competitive, trend, or financial) and provide a structured response.\nInclude all relevant sections as appropriate.""",
                'user_prompt': """Analyze the following business question: {specific_requirements}\nInclude all relevant insights, comparisons, and recommendations."""
            }
        }
    
    def analyze_query(self, query: str, uploaded_files: List[Dict] = None) -> Optional[BusinessQuery]:
        prompt = f"""Classify this business query: '{query}'
        Respond in JSON format with:
        - query_type: [competitive_analysis|trend_forecasting|financial_analysis]  
        - domain: industry sector
        - entities: list of companies/products mentioned
        - time_frame: analysis period
        - complexity: [simple|moderate|complex]
        - follow_up_questions: 3 relevant clarifying questions"""
        
        try:
            response = model.generate_content(prompt)
            result = self._parse_json(response.text)
            
            # Process uploaded files
            file_context = ""
            if uploaded_files:
                file_context = "\nAdditional context from uploaded files:\n"
                for file in uploaded_files:
                    if file['type'] in ['text', 'pdf']:
                        file_context += f"\n{file['name']} content:\n{file['content'][:1000]}...\n"
                    else:
                        file_context += f"\nFile {file['name']} was uploaded (type: {file['type']})\n"
            
            return BusinessQuery(
                raw_query=query + file_context,
                query_type=result.get('query_type'),
                domain=result.get('domain'),
                entities=result.get('entities', []),
                time_frame=result.get('time_frame'),
                complexity=result.get('complexity'),
                follow_up_questions=result.get('follow_up_questions', []),
                uploaded_files=uploaded_files,
                conversation_history=[{"role": "user", "content": query}]
            )
        except Exception as e:
            st.error(f"Query analysis failed: {str(e)}")
            return None
    
    def generate_response(self, query: BusinessQuery) -> Dict:
        template = self.prompt_templates.get(query.query_type)
        if not template:
            return {"error": "Unsupported query type"}
        
        # Include file context in the prompt if available
        file_context = ""
        if query.uploaded_files:
            file_context = "\nAdditional context from uploaded files:\n"
            for file in query.uploaded_files:
                if file['type'] in ['text', 'pdf']:
                    file_context += f"\n{file['name']} content:\n{file['content'][:2000]}\n"
        
        # Prepare the user prompt with all required parameters
        user_prompt_params = {
            'entities': ", ".join(query.entities) if isinstance(query.entities, list) else query.entities,
            'domain': query.domain,
            'time_frame': query.time_frame,
            'specific_requirements': query.raw_query,  # Use raw_query for competitive/financial/automatic
            'specific_focus_areas': query.raw_query    # Use raw_query for trend forecasting
        }
        try:
            user_prompt = template['user_prompt'].format(**user_prompt_params)
        except KeyError as e:
            return {"error": f"Missing template parameter: {str(e)}"}
        
        # Build conversation from history
        conversation = query.conversation_history.copy() if query.conversation_history else []
        conversation.append({"role": "user", "content": user_prompt})
        # Only add system prompt if not already present
        if not conversation or conversation[0].get("role") != "system":
            conversation.insert(0, {"role": "system", "content": template['system_prompt'].format(
                entities=", ".join(query.entities) if isinstance(query.entities, list) else query.entities,
                domain=query.domain,
                time_frame=query.time_frame
            ) + ("\n" + file_context if file_context else "")})
        try:
            response = model.generate_content(conversation)
            # Append model response to history
            if query.conversation_history is not None:
                query.conversation_history.append({"role": "model", "content": response.text})
            return self._structure_response(response.text, query.query_type)
        except Exception as e:
            st.error(f"Response generation failed: {str(e)}")
            return {"error": f"Response generation failed: {str(e)}"}
    
    def _structure_response(self, raw_response: str, query_type: str) -> Dict:
        structure_prompt = f"""Convert this analysis to structured JSON format:
        {raw_response}
        
        Include these sections:
        - executive_summary: Brief overview
        - key_findings: Bullet points
        - comparative_analysis: Table if applicable  
        - recommendations: Prioritized list
        - implementation_plan: Phased approach
        - risk_assessment: Key risks"""
        
        try:
            structured = model.generate_content(structure_prompt)
            return self._parse_json(structured.text)
        except:
            return {"analysis": raw_response}
    
    def _parse_json(self, text: str) -> Dict:
        try:
            return json.loads(text[text.find('{'):text.rfind('}')+1])
        except:
            return {"analysis": text}
    
    def _calculate_metrics(self, response: Dict) -> Dict:
        """Calculate quality metrics for the analysis"""
        metrics = {
            "completeness": {"score": 0, "description": "Coverage of key aspects"},
            "actionability": {"score": 0, "description": "Practical recommendations"},
            "depth": {"score": 0, "description": "Analysis thoroughness"},
            "clarity": {"score": 0, "description": "Explanation quality"}
        }
        
        if isinstance(response, dict):
            # Score based on presence of key sections
            sections = ['executive_summary', 'key_findings', 'recommendations']
            present_sections = sum(1 for s in sections if s in response)
            metrics['completeness']['score'] = min(100, (present_sections/len(sections))*100)
            
            # Score based on recommendation quality
            if 'recommendations' in response:
                metrics['actionability']['score'] = 80 if isinstance(response['recommendations'], list) else 60
            
            # Depth score
            if 'key_findings' in response and isinstance(response['key_findings'], list):
                metrics['depth']['score'] = min(100, len(response['key_findings'])*10)
            
            # Clarity score (simple heuristic)
            metrics['clarity']['score'] = 75  # Base score
        
        return metrics

def extract_pdf_text(uploaded_file):
    """Extract text from PDF with error handling"""
    try:
        pdf_reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() for page in pdf_reader.pages[:5])  # Limit pages
    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
        return ""

def process_uploaded_file(uploaded_file):
    """Process uploaded files of different types"""
    file_type = uploaded_file.type.split('/')[0]
    content = ""
    
    try:
        if file_type == 'text':
            content = uploaded_file.getvalue().decode('utf-8')
        elif 'pdf' in uploaded_file.type:
            content = extract_pdf_text(uploaded_file)
        elif 'excel' in uploaded_file.type:
            content = "Excel data (processing not implemented)"
        elif 'word' in uploaded_file.type:
            content = "Word document (processing not implemented)"
        elif file_type == 'image':
            content = "Image content (processing not implemented)"
    except Exception as e:
        content = f"Error processing file: {str(e)}"
    
    return {
        'name': uploaded_file.name,
        'type': file_type,
        'content': content,
        'size': uploaded_file.size
    }

def display_analysis(response: Dict, metrics: Dict, query: BusinessQuery):
    """Display analysis results with metrics and download options"""
    st.success("## Analysis Results")
    
    # Metrics dashboard
    st.subheader("Quality Assessment")
    cols = st.columns(4)
    cols[0].metric("Completeness", f"{metrics['completeness']['score']}/100", 
                  help=metrics['completeness']['description'])
    cols[1].metric("Actionability", f"{metrics['actionability']['score']}/100",
                  help=metrics['actionability']['description'])
    cols[2].metric("Depth", f"{metrics['depth']['score']}/100",
                  help=metrics['depth']['description'])
    cols[3].metric("Clarity", f"{metrics['clarity']['score']}/100",
                  help=metrics['clarity']['description'])
    
    # Analysis sections
    if isinstance(response, dict):
        with st.expander("📌 Executive Summary", expanded=True):
            st.write(response.get('executive_summary', 'Not available'))
        
        with st.expander("🔍 Key Findings"):
            findings = response.get('key_findings', [])
            if isinstance(findings, list):
                for item in findings:
                    st.markdown(f"- {item}")
            else:
                st.write(findings)
        
        if 'comparative_analysis' in response:
            with st.expander("📊 Comparative Analysis"):
                try:
                    comp_data = response['comparative_analysis']
                    if isinstance(comp_data, dict):
                        # Convert {metric: {company: value}} to {company: {metric: value}}
                        company_dict = {}
                        for metric, companies in comp_data.items():
                            if isinstance(companies, dict):
                                for company, value in companies.items():
                                    if company not in company_dict:
                                        company_dict[company] = {}
                                    company_dict[company][metric] = str(value)
                        df = pd.DataFrame.from_dict(company_dict, orient='index')
                        st.dataframe(df)
                    elif isinstance(comp_data, list):
                        df = pd.DataFrame(comp_data)
                        st.dataframe(df)
                    else:
                        st.write(comp_data)
                except Exception as e:
                    st.warning(f"Could not display comparative analysis: {str(e)}")
                    st.write(response['comparative_analysis'])
        
        with st.expander("💡 Recommendations"):
            recs = response.get('recommendations', [])
            if isinstance(recs, list):
                for i, rec in enumerate(recs, 1):
                    st.markdown(f"{i}. **{rec}**")
            else:
                st.write(recs)
        
        if 'implementation_plan' in response:
            with st.expander("🛠️ Implementation Plan"):
                st.write(response['implementation_plan'])
    else:
        st.write(response)
    
    # Download options
    st.subheader("📥 Download Report")
    
    # Prepare data for downloads
    report_data = {
        "query": query.raw_query,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "analysis": response
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON download
        st.download_button(
            label="Download JSON",
            data=json.dumps(report_data, indent=2),
            file_name=f"business_insights_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    with col2:
        # Text summary
        txt_content = f"Business Insights Report\n{'='*40}\n\n"
        txt_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        if isinstance(response, dict):
            for section in ['executive_summary', 'key_findings', 'recommendations']:
                if section in response:
                    txt_content += f"{section.replace('_', ' ').title()}:\n"
                    if isinstance(response[section], list):
                        txt_content += "\n".join(f"- {item}" for item in response[section]) + "\n\n"
                    else:
                        txt_content += f"{response[section]}\n\n"
        
        st.download_button(
            label="Download Text",
            data=txt_content,
            file_name=f"insights_summary_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    
    with col3:
        # CSV if comparative data exists
        if isinstance(response, dict) and 'comparative_analysis' in response:
            try:
                comp_data = response['comparative_analysis']
                if isinstance(comp_data, dict):
                    df = pd.DataFrame.from_dict(comp_data, orient='index').transpose()
                elif isinstance(comp_data, list):
                    df = pd.DataFrame(comp_data)
                else:
                    raise ValueError("Unsupported format")
                
                st.download_button(
                    label="Download CSV",
                    data=df.to_csv(index=False),
                    file_name=f"comparative_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.warning(f"Could not prepare CSV: {str(e)}")
        else:
            st.warning("No comparative data")
def main():
    st.set_page_config(page_title="Business Insights Assistant", layout="wide")
    
    # Initialize session state
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    if 'follow_up_needed' not in st.session_state:
        st.session_state.follow_up_needed = False
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    st.title("📈 AI-Powered Business Insights")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Automatic", "Competitive", "Trend", "Financial"],
            key="analysis_type"
        )
        industry = st.selectbox(
            "Industry",
            ["Technology", "Finance", "Healthcare", "Retail", "Manufacturing"],
            key="industry"
        )
        
        st.header("📂 Upload Supporting Files")
        uploaded_files = st.file_uploader(
            "Add relevant documents",
            type=['txt', 'pdf', 'csv', 'xlsx'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = [process_uploaded_file(f) for f in uploaded_files]
            for file in st.session_state.uploaded_files:
                st.info(f"{file['name']} ({file['type']}, {file['size']/1024:.1f} KB)")
    
    # Main query input
    query = st.text_area(
        "Enter your business question:",
        placeholder="E.g., Analyze our competitive position in the European market...",
        height=150,
        key="query_input"
    )
    
    # Action buttons
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("Generate Analysis", type="primary", use_container_width=True):
            if not query.strip():
                st.warning("Please enter a query")
            else:
                with st.spinner("Processing your request..."):
                    try:
                        assistant = BusinessInsightsAssistant()
                        business_query = assistant.analyze_query(
                            query,
                            st.session_state.uploaded_files
                        )
                        
                        if business_query is None:
                            return
                        
                        if business_query.follow_up_questions:
                            st.session_state.follow_up_needed = True
                            st.session_state.current_query = business_query
                            st.rerun()
                        else:
                            response = assistant.generate_response(business_query)
                            metrics = assistant._calculate_metrics(response)
                            
                            st.session_state.current_analysis = {
                                "response": response,
                                "metrics": metrics,
                                "query": business_query
                            }
                            st.session_state.follow_up_needed = False
                            st.rerun()
                    
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
    
    with col2:
        if st.button("Clear Results", type="secondary", use_container_width=True):
            st.session_state.current_analysis = None
            st.session_state.follow_up_needed = False
            st.rerun()
    
    # Follow-up questions
    if st.session_state.follow_up_needed and hasattr(st.session_state, 'current_query'):
        st.subheader("🔍 Additional Information Needed")
        st.write("Please answer these questions to improve the analysis:")
        
        answers = []
        for i, question in enumerate(st.session_state.current_query.follow_up_questions[:3]):
            answers.append(st.text_input(
                f"Q{i+1}: {question}",
                key=f"q_{i}",
                placeholder="Your answer..."
            ))
        
        if st.button("Submit Answers", type="primary"):
            if any(a.strip() for a in answers):
                enhanced_query = st.session_state.current_query.raw_query + "\n\nFollow-up Answers:\n"
                for q, a in zip(st.session_state.current_query.follow_up_questions[:3], answers):
                    if a.strip():
                        enhanced_query += f"- {q}: {a}\n"
                
                with st.spinner("Generating enhanced analysis..."):
                    try:
                        assistant = BusinessInsightsAssistant()
                        updated_query = BusinessQuery(
                            raw_query=enhanced_query,
                            query_type=st.session_state.current_query.query_type,
                            domain=st.session_state.current_query.domain,
                            entities=st.session_state.current_query.entities,
                            time_frame=st.session_state.current_query.time_frame,
                            complexity=st.session_state.current_query.complexity,
                            uploaded_files=st.session_state.current_query.uploaded_files
                        )
                        
                        response = assistant.generate_response(updated_query)
                        metrics = assistant._calculate_metrics(response)
                        
                        st.session_state.current_analysis = {
                            "response": response,
                            "metrics": metrics,
                            "query": updated_query
                        }
                        st.session_state.follow_up_needed = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Enhanced analysis failed: {str(e)}")
            else:
                st.warning("Please provide at least one answer")
    
    # Display current analysis if available
    if st.session_state.current_analysis:
        display_analysis(
            st.session_state.current_analysis['response'],
            st.session_state.current_analysis['metrics'],
            st.session_state.current_analysis['query']
        )

if __name__ == "__main__":
    # Install required package if not available
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        st.warning("Installing required PyPDF2 package...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
        from PyPDF2 import PdfReader
    
    main()