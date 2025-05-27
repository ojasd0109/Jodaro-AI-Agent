import json
import tempfile
import csv
import streamlit as st
import pandas as pd
import logging
from collections import Counter
import openai
import re
import time
import xlsxwriter
from datetime import datetime
import io
import base64

# ✅ Enable Logging for Debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize session state for recent searches if it doesn't exist
if 'recent_searches' not in st.session_state:
    st.session_state.recent_searches = []

# ✅ Function to analyze Amazon product data
def analyze_amazon_product_data(df, column_name):
    """Analyze Amazon product data for keyword frequency, attribute categorization, and market insights."""
    if column_name not in df.columns:
        st.error(f"❌ Column '{column_name}' not found.")
        return None

    openai.api_key = st.session_state.openai_key
    column_data = df[column_name].dropna().astype(str)

    # Combine the first 500 values into a single sample for LLM
    column_text = " ".join(column_data.tolist()[:500])

    prompt = f"""
    You are an expert Amazon product researcher and AI analyst.

    A dataset was uploaded from Amazon product listings. The user selected the column '{column_name}'.

    Below are sample entries from this column:
    {column_text}

    Based on this, return your analysis in the following structured format:
    
    1. Most Common Keywords:
       - Top 20 words/phrases used most frequently in this column.
       - Include a frequency count.

    2. Keyword Categorization:
       - Group and categorize the keywords into meaningful product attributes or umbrella categories.
       - Example: "silk panel", "window treatment" → Category: Product Type.

    3. Umbrella Parameters:
       - Identify standard attributes commonly used in e-commerce like Size, Color, Material, Pattern, etc.
       - Assign matching keywords under each umbrella parameter.
       - Add a "Miscellaneous" group for any that don't fit.

    4. Market Insights:
       - Highlight trends or product types dominating this column.
       - Mention any value propositions or patterns in descriptions.

    Format your result in Markdown table and bullet point sections.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=1800
        )
        return response.choices[0].message.content

    except Exception as e:
        logging.error(f"❌ OpenAI API request failed: {e}")
        return f"❌ OpenAI API request failed: {e}"

# ✅ Function to preprocess uploaded file
def preprocess_and_save(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None
        
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name 
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
        return temp_path, df.columns.tolist(), df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        logging.error(f"Error processing file: {e}")
        return None, None, None

# ✅ Function to analyze common phrases in a column
def analyze_common_phrases(df, column_name):
    if column_name not in df.columns:
        st.error(f"❌ Column '{column_name}' not found in the dataset.")
        return None 
    
    # ✅ Check if the column is numeric
    if df[column_name].dtype == 'object':
        st.error(f"❌ Column '{column_name}' is not numeric.")
        return None

    word_counter = Counter()
    text_data = ' '.join(df[column_name].dropna().astype(str))
    words = re.findall(r'\b\w+\b', text_data.lower())  
    word_counter.update(words)

    common_words = sorted(word_counter.items(), key=lambda x: x[1])  # Sorted in ascending order
    
    common_df = pd.DataFrame(common_words, columns=["Word", "Frequency"])
    
    return common_df

# ✅ Function to send queries to OpenAI API
def ask_openai(query, df, column_name):
    """Send a query to OpenAI API for processing."""
    
    openai_client = openai.OpenAI(api_key=st.session_state.openai_key)

    column_data = df[column_name].dropna().astype(str).tolist()
    column_text = " ".join(column_data[:500])

    prompt = f"""
    You are an AI data analyst. The user uploaded a dataset and selected the '{column_name}' column.
    Here are sample rows from that column:
    {column_text}

    Perform the following analysis:
    {query}

    Format your response as structured insights.
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    
    except Exception as e:
        logging.error(f"OpenAI API request failed: {e}")
        return f"❌ OpenAI API request failed: {e}"

def validate_parameter_against_category(parameter, product_category, openai_client):
    """Validate if a parameter is relevant to the product category using LLM."""
    prompt = f"""
    You are an expert in product categorization and attribute analysis.
    
    Product Category: {product_category}
    Parameter to validate: {parameter}
    
    Determine if this parameter is relevant and commonly used for products in this category.
    Consider:
    1. Is this parameter typically used to describe products in this category?
    2. Would customers expect to see this parameter when shopping for these products?
    3. Is this parameter meaningful for product comparison in this category?
    
    Return ONLY 'YES' if the parameter is relevant, or 'NO' if it's not relevant.
    """
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=50
        )
        result = response.choices[0].message.content.strip().upper()
        return result == 'YES'
    except Exception as e:
        logging.error(f"Error validating parameter: {e}")
        return False

def standardize_parameter_value(value, parameter, openai_client):
    """Standardize a parameter value using LLM."""
    prompt = f"""
    You are an expert in product attribute standardization.
    
    Parameter: {parameter}
    Current value: {value}
    
    Standardize this value to a common format used in e-commerce.
    Consider:
    1. Common abbreviations and their full forms
    2. Standard units of measurement
    3. Common product attribute formats
    
    Return ONLY the standardized value, nothing else.
    If the value cannot be standardized meaningfully, return 'N/A'.
    """
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error standardizing value: {e}")
        return 'N/A'

def validate_parameter_value(value, parameter, openai_client):
    """Validate if a parameter value is correct for the given parameter."""
    prompt = f"""
    You are an expert in product attribute validation.
    
    Parameter: {parameter}
    Value to validate: {value}
    
    Determine if this value is correct and appropriate for this parameter.
    Consider:
    1. Is the value in the correct format?
    2. Is the value within expected ranges?
    3. Is the value meaningful for this parameter?
    
    Return ONLY 'YES' if the value is correct, or 'NO' if it's incorrect.
    """
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=50
        )
        result = response.choices[0].message.content.strip().upper()
        return result == 'YES'
    except Exception as e:
        logging.error(f"Error validating value: {e}")
        return False

def extract_parameter_value(text, parameter, openai_client, max_attempts=3):
    """Extract and validate parameter value with standardization process."""
    prompt = f"""
    Extract the value of the parameter '{parameter}' from the following text.
    Return ONLY the value, nothing else.
    If no value is found, return "N/A".
    
    Text: {text}
    """
    
    try:
        # First attempt to extract the value
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        value = response.choices[0].message.content.strip()
        
        if value == "N/A":
            return value
            
        # Validate and standardize process
        attempts = 0
        while attempts < max_attempts:
            # Validate the value
            validation_prompt = f"""
            You are an expert in product attribute validation.
            
            Parameter: {parameter}
            Value to validate: {value}
            
            Determine if this value is correct and appropriate for this parameter.
            Consider:
            1. Is the value in the correct format?
            2. Is the value within expected ranges?
            3. Is the value meaningful for this parameter?
            
            Return ONLY 'YES' if the value is correct, or 'NO' if it's incorrect.
            """
            
            validation_response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": validation_prompt}],
                temperature=0.3,
                max_tokens=50
            )
            
            if validation_response.choices[0].message.content.strip().upper() == 'YES':
                return value
                
            # If invalid, try to standardize
            standardization_prompt = f"""
            You are an expert in product attribute standardization.
            
            Parameter: {parameter}
            Current value: {value}
            
            Standardize this value to a common format used in e-commerce.
            Consider:
            1. Common abbreviations and their full forms
            2. Standard units of measurement
            3. Common product attribute formats
            
            Return ONLY the standardized value, nothing else.
            If the value cannot be standardized meaningfully, return 'N/A'.
            """
            
            standardization_response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": standardization_prompt}],
                temperature=0.3,
                max_tokens=100
            )
            
            standardized_value = standardization_response.choices[0].message.content.strip()
            
            # Validate the standardized value
            validation_response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": validation_prompt}],
                temperature=0.3,
                max_tokens=50
            )
            
            if validation_response.choices[0].message.content.strip().upper() == 'YES':
                return standardized_value
                
            # If still invalid, try to extract again
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            value = response.choices[0].message.content.strip()
            
            attempts += 1
            
        # If we've exhausted all attempts, return N/A
        return "N/A"
        
    except Exception as e:
        logging.error(f"Error extracting parameter value: {e}")
        return "N/A"

# ✅ Function to analyze umbrella parameters across all columns
def analyze_all_umbrella_parameters(df):
    """Analyze umbrella parameters across all columns in the dataset."""
    openai.api_key = st.session_state.openai_key
    
    # Get product category from the first row
    product_category = None
    if 'Category' in df.columns:
        product_category = df['Category'].iloc[0]
    elif 'Product Category' in df.columns:
        product_category = df['Product Category'].iloc[0]
    
    if not product_category:
        st.warning("⚠️ Product category not found. Using default category validation.")
        product_category = "General Products"
    
    # Initialize OpenAI client
    openai_client = openai.OpenAI(api_key=st.session_state.openai_key)
    
    # Filter columns with sufficient data
    valid_columns = []
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        if non_null_count > 0 and non_null_count / len(df) > 0.1:  # At least 10% non-null values
            valid_columns.append(col)
    
    if not valid_columns:
        return "❌ No columns found with sufficient data for analysis."
    
    # Process each column individually
    all_parameters = []
    parameter_words = {}  # Dictionary to store words under each parameter
    parameter_asins = {}  # Dictionary to store ASINs for each parameter
    
    for col in valid_columns:
        try:
            # Get all non-null entries from the column
            column_data = df[col].dropna().astype(str).tolist()
            if not column_data:
                continue
                
            column_text = " ".join(column_data)
            
            prompt = f"""
            You are an expert Amazon product researcher and AI analyst.
            
            Analyze this data from the column '{col}':
            {column_text}
            
            Identify standard e-commerce parameters (like Size, Color, Material, etc.) present in this data.
            For each parameter, also identify the common words/phrases used under that parameter.
            
            Return in this exact format:
            Parameter: [parameter name]
            Words: [word1, word2, word3, ...]
            
            Only return parameters that are clearly present in the data.
            Do not include any other text or explanations.
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse the response to extract parameters and their words
            current_param = None
            current_words = []
            
            for line in response.choices[0].message.content.split("\n"):
                if line.startswith("Parameter:"):
                    if current_param and current_words:
                        # Validate parameter against product category
                        validation_prompt = f"""
                        You are an expert in product categorization and attribute analysis.
                        
                        Product Category: {product_category}
                        Parameter to validate: {current_param}
                        
                        Determine if this parameter is relevant and commonly used for products in this category.
                        Consider:
                        1. Is this parameter typically used to describe products in this category?
                        2. Would customers expect to see this parameter when shopping for these products?
                        3. Is this parameter meaningful for product comparison in this category?
                        
                        Return ONLY 'YES' if the parameter is relevant, or 'NO' if it's not relevant.
                        """
                        
                        validation_response = openai_client.chat.completions.create(
                            model="gpt-3.5-turbo-0125",
                            messages=[{"role": "user", "content": validation_prompt}],
                            temperature=0.3,
                            max_tokens=50
                        )
                        
                        if validation_response.choices[0].message.content.strip().upper() == 'YES':
                            if current_param not in parameter_words:
                                parameter_words[current_param] = set()
                            parameter_words[current_param].update(current_words)
                            all_parameters.append(current_param)
                    current_param = line.split(": ")[1]
                    current_words = []
                elif line.startswith("Words:"):
                    current_words = [word.strip() for word in line.split(": ")[1].split(",")]
            
            # Add the last parameter if exists
            if current_param and current_words:
                # Validate parameter against product category
                validation_prompt = f"""
                You are an expert in product categorization and attribute analysis.
                
                Product Category: {product_category}
                Parameter to validate: {current_param}
                
                Determine if this parameter is relevant and commonly used for products in this category.
                Consider:
                1. Is this parameter typically used to describe products in this category?
                2. Would customers expect to see this parameter when shopping for these products?
                3. Is this parameter meaningful for product comparison in this category?
                
                Return ONLY 'YES' if the parameter is relevant, or 'NO' if it's not relevant.
                """
                
                validation_response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=[{"role": "user", "content": validation_prompt}],
                    temperature=0.3,
                    max_tokens=50
                )
                
                if validation_response.choices[0].message.content.strip().upper() == 'YES':
                    if current_param not in parameter_words:
                        parameter_words[current_param] = set()
                    parameter_words[current_param].update(current_words)
                    all_parameters.append(current_param)
            
        except Exception as e:
            logging.error(f"Error analyzing column {col}: {e}")
            continue
    
    # Get unique parameters and ensure we have 20
    unique_parameters = list(set(all_parameters))
    if len(unique_parameters) < 20:
        # If we have less than 20 parameters, try to get more by analyzing the data again
        additional_prompt = f"""
        Analyze the following data and identify additional standard e-commerce parameters:
        {column_text}
        
        Return ONLY parameter names, one per line, in this format:
        Parameter: [parameter name]
        
        Focus on finding standard e-commerce parameters that might have been missed.
        """
        
        try:
            additional_response = openai.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": additional_prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            for line in additional_response.choices[0].message.content.split("\n"):
                if line.startswith("Parameter:"):
                    param = line.split(": ")[1]
                    if param not in unique_parameters:
                        unique_parameters.append(param)
                        if len(unique_parameters) >= 20:
                            break
        except Exception as e:
            logging.error(f"Error getting additional parameters: {e}")
    
    # Take top 20 parameters
    unique_parameters = unique_parameters[:20]
    
    # Analyze ASIN relationships for each parameter
    for param in unique_parameters:
        param_words = parameter_words.get(param, set())
        if param_words:
            # Find ASINs that contain these words
            matching_asins = []
            for idx, row in df.iterrows():
                row_text = " ".join(str(val) for val in row if pd.notna(val))
                if any(word.lower() in row_text.lower() for word in param_words):
                    # Use index as product ID if ASIN column doesn't exist
                    product_id = row.get('ASIN', f'Product_{idx}')
                    matching_asins.append(product_id)
            parameter_asins[param] = matching_asins
    
    # Format the results
    result = "# Umbrella Parameters Analysis\n\n"
    result += "## Top 20 Umbrella Parameters\n"
    result += "| S.No. | Umbrella Parameter |\n"
    result += "|-------|-------------------|\n"
    
    for idx, param in enumerate(unique_parameters, 1):
        result += f"| {idx} | {param} |\n"
    
    result += "\n## Parameter-ASIN Relationships\n"
    result += "| Umbrella Parameter | Number of Related ASINs |\n"
    result += "|-------------------|------------------------|\n"
    
    for param in unique_parameters:
        asin_count = len(parameter_asins.get(param, []))
        result += f"| {param} | {asin_count} |\n"
    
    result += "\n## Detailed Parameter Analysis\n"
    for param in unique_parameters:
        result += f"\n### {param}\n"
        
        # Common Words Table
        words = parameter_words.get(param, set())
        if words:
            result += "#### Common Words\n"
            result += "| Word |\n"
            result += "|------|\n"
            for word in sorted(words):
                result += f"| {word} |\n"
        
        # Related ASINs Table
        asins = parameter_asins.get(param, [])
        if asins:
            result += "\n#### Related ASINs\n"
            result += "| ASIN |\n"
            result += "|------|\n"
            for asin in sorted(asins):
                result += f"| {asin} |\n"
        else:
            result += "\n#### Related ASINs\n"
            result += "No related ASINs found.\n"
        
        result += "\n---\n"  # Add a horizontal line between parameters
    
    result += "\n## Recommendations\n"
    result += "- Focus on standardizing the most common parameters across products\n"
    result += "- Consider adding missing standard parameters for better product categorization\n"
    result += "- Review and consolidate similar parameters to improve consistency\n"
    
    # After creating the markdown result, create the Excel file
    excel_output, excel_filename = create_color_coded_excel(df, parameter_asins, parameter_words)
    
    if excel_output:
        result += f"\n\n## 📊 Excel Analysis Output\n"
        result += f"A detailed color-coded Excel analysis has been generated: `{excel_filename}`\n"
        result += "The Excel file includes:\n"
        result += "- Umbrella Parameters with their common words\n"
        result += "- Related ASINs for each parameter\n"
        result += "- Color-coded sections for better visualization\n"
        result += "- A summary sheet with key statistics\n"
        
        # Add Excel file download button and display
        if excel_output:
            # Reset the BytesIO object position
            excel_output.seek(0)
            
            # Add download button at the top
            try:
                # Generate timestamp for unique key
                download_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="📥 Download Excel Analysis",
                    data=excel_output.getvalue(),
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_umbrella_analysis_{download_timestamp}"
                )
            except Exception as e:
                st.error(f"Error creating download button: {e}")
                logging.error(f"Error creating download button: {e}")
            
            # Add custom CSS for better table display
            st.markdown("""
                <style>
                .stDataFrame {
                    background-color: white;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                    margin: 10px 0;
                }
                .stDataFrame td {
                    padding: 5px;
                    border: 1px solid #ddd;
                }
                .stDataFrame th {
                    background-color: #D9E1F2;
                    font-weight: bold;
                    padding: 5px;
                    border: 2px solid black;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Display Parameter Values sheet
            st.subheader("📊 Excel Preview - Parameter Values")
            try:
                excel_output.seek(0)
                excel_df = pd.read_excel(excel_output, sheet_name="Parameter Values")
                
                # Log the DataFrame info for debugging
                logging.info(f"DataFrame shape: {excel_df.shape}")
                logging.info(f"DataFrame columns: {excel_df.columns.tolist()}")
                
                # Style the DataFrame
                def highlight_rows(x):
                    return ['background-color: #F2F2F2' if i % 2 == 0 else 'background-color: #FFFFFF' for i in range(len(x))]
                
                styled_df = excel_df.style.apply(highlight_rows, axis=1)
                
                # Display the DataFrame with error handling
                try:
                    st.dataframe(styled_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying DataFrame: {e}")
                    # Try displaying without styling as fallback
                    st.dataframe(excel_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error reading Parameter Values sheet: {e}")
                logging.error(f"Error reading Parameter Values sheet: {e}")
            
            # Display Summary sheet
            st.subheader("📊 Excel Preview - Summary")
            try:
                excel_output.seek(0)
                summary_df = pd.read_excel(excel_output, sheet_name="Summary")
                
                # Log the Summary DataFrame info for debugging
                logging.info(f"Summary DataFrame shape: {summary_df.shape}")
                logging.info(f"Summary DataFrame columns: {summary_df.columns.tolist()}")
                
                # Style the summary DataFrame
                def highlight_summary_rows(x):
                    return ['background-color: #F2F2F2' if i % 2 == 0 else 'background-color: #FFFFFF' for i in range(len(x))]
                
                styled_summary = summary_df.style.apply(highlight_summary_rows, axis=1)
                
                # Display the Summary DataFrame with error handling
                try:
                    st.dataframe(styled_summary, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying Summary DataFrame: {e}")
                    # Try displaying without styling as fallback
                    st.dataframe(summary_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error reading Summary sheet: {e}")
                logging.error(f"Error reading Summary sheet: {e}")
            
            # Add a note about the Excel file
            st.info("💡 The Excel file has been generated with color-coded sections for better visualization. You can download it using the button above.")
    
    return result, excel_output, excel_filename

def create_color_coded_excel(df, parameter_asins, parameter_words):
    """Create a color-coded Excel file for umbrella parameters analysis."""
    try:
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"umbrella_parameters_analysis_{timestamp}.xlsx"
        
        # Create an in-memory Excel file
        output = io.BytesIO()
        workbook = xlsxwriter.Workbook(output)
        
        # Use all parameters without filtering
        all_parameters = parameter_asins
        
        if not all_parameters:
            return None, None
            
        # Create the main worksheet
        worksheet = workbook.add_worksheet("Parameter Values")
        
        # Define colors for different sections
        colors = {
            'header': '#D9E1F2',  # Light blue
            'data': '#FFFFFF',    # White
            'alternate': '#F2F2F2'  # Light gray
        }
        
        # Create formats
        header_format = workbook.add_format({
            'bg_color': colors['header'],
            'bold': True,
            'border': 1,
            'align': 'center',
            'valign': 'vcenter'
        })
        
        data_format = workbook.add_format({
            'bg_color': colors['data'],
            'border': 1,
            'align': 'center',
            'valign': 'vcenter'
        })
        
        alternate_format = workbook.add_format({
            'bg_color': colors['alternate'],
            'border': 1,
            'align': 'center',
            'valign': 'vcenter'
        })
        
        # Write headers
        worksheet.write('A1', 'ASIN', header_format)
        for idx, param in enumerate(all_parameters.keys(), 1):
            worksheet.write(0, idx, param, header_format)
        
        # Set column widths
        worksheet.set_column('A:A', 15)  # ASIN column
        for idx in range(1, len(all_parameters) + 1):
            worksheet.set_column(idx, idx, 20)  # Parameter columns
        
        # Get all unique ASINs
        all_asins = set()
        for asins in all_parameters.values():
            all_asins.update(asins)
        all_asins = sorted(list(all_asins))
        
        # Initialize OpenAI client
        openai_client = openai.OpenAI(api_key=st.session_state.openai_key)
        
        # Write data
        for row_idx, asin in enumerate(all_asins, 1):
            # Write ASIN
            worksheet.write(row_idx, 0, asin, data_format if row_idx % 2 == 0 else alternate_format)
            
            # Get row text for parameter extraction
            try:
                # Find the row index in the original DataFrame
                if 'ASIN' in df.columns:
                    row_data = df[df['ASIN'] == asin].iloc[0]
                else:
                    # If ASIN column doesn't exist, use the index from the ASIN string
                    idx = int(asin.split('_')[1])
                    row_data = df.iloc[idx]
                
                row_text = " ".join(str(val) for val in row_data if pd.notna(val))
                
                # Extract and write parameter values
                for col_idx, param in enumerate(all_parameters.keys(), 1):
                    value = extract_parameter_value(row_text, param, openai_client)
                    worksheet.write(row_idx, col_idx, value, data_format if row_idx % 2 == 0 else alternate_format)
            except Exception as e:
                logging.error(f"Error processing row {row_idx}: {e}")
                continue
        
        # Add a summary sheet
        summary_sheet = workbook.add_worksheet("Summary")
        
        # Create summary formats
        summary_header_format = workbook.add_format({
            'bg_color': colors['header'],
            'bold': True,
            'border': 1,
            'align': 'center',
            'valign': 'vcenter'
        })
        
        summary_data_format = workbook.add_format({
            'bg_color': colors['data'],
            'border': 1,
            'align': 'center',
            'valign': 'vcenter'
        })
        
        # Write summary headers
        summary_sheet.write('A1', 'Analysis Summary', summary_header_format)
        summary_sheet.merge_range('A1:B1', 'Analysis Summary', summary_header_format)
        
        # Write summary data
        summary_sheet.write('A2', 'Total Parameters', summary_data_format)
        summary_sheet.write('B2', len(all_parameters), summary_data_format)
        
        summary_sheet.write('A3', 'Total ASINs', summary_data_format)
        summary_sheet.write('B3', len(all_asins), summary_data_format)
        
        summary_sheet.write('A4', 'Generated on', summary_data_format)
        summary_sheet.write('B4', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), summary_data_format)
        
        # Add parameter statistics
        summary_sheet.write('A6', 'Parameter Statistics', summary_header_format)
        summary_sheet.merge_range('A6:B6', 'Parameter Statistics', summary_header_format)
        
        summary_sheet.write('A7', 'Parameter', summary_header_format)
        summary_sheet.write('B7', 'Number of Related ASINs', summary_header_format)
        
        for idx, (param, asins) in enumerate(all_parameters.items(), 8):
            summary_sheet.write(idx, 0, param, summary_data_format)
            summary_sheet.write(idx, 1, len(asins), summary_data_format)
        
        # Set column widths for summary sheet
        summary_sheet.set_column('A:A', 30)
        summary_sheet.set_column('B:B', 20)
        
        # Save the workbook
        workbook.close()
        output.seek(0)
        
        return output, excel_filename
        
    except Exception as e:
        logging.error(f"Error creating Excel file: {e}")
        return None, None

# ✅ Streamlit App UI
st.title("📊 AI-Powered Amazon Product Research Tool")

# ✅ Sidebar for API Keys
with st.sidebar:
    st.header("API Keys")
    openai_key = st.text_input("Enter your OpenAI API key:", type="password")
    if openai_key:
        st.session_state.openai_key = openai_key
        st.success("✅ API key saved!")
    else:
        st.warning("⚠️ Please enter your OpenAI API key to proceed.")
        st.info("💡 You can get an API key from: https://platform.openai.com/api-keys")

# ✅ File Upload Widget
uploaded_file = st.file_uploader("Upload Amazon Product Data (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    if "openai_key" not in st.session_state:
        st.error("❌ Please enter your OpenAI API key in the sidebar to proceed.")
    else:
        temp_path, columns, df = preprocess_and_save(uploaded_file)
        
        if temp_path and columns and df is not None:
            st.write("✅ Uploaded Data:")
            st.dataframe(df)
            st.write("📊 Available Columns:", columns)

            st.header("🔍 Amazon Product Data Analysis")
            
            # Create two columns for better layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Select Data to Analyze")
                analysis_column = st.selectbox(
                    "Choose a Column for Analysis",
                    columns,
                    help="Select the column you want to analyze for keywords and insights"
                )
                
                # Add text input for custom queries
                custom_query = st.text_area(
                    "📝 Enter Custom Query (Optional)",
                    help="Enter any specific questions about the data you want to analyze"
                )
                
                # Create a row for buttons
                button_col1, button_col2 = st.columns(2)
                
                with button_col1:
                    if st.button("🔍 Analyze All Umbrella Parameters", type="secondary"):
                        with st.spinner('⏳ Analyzing umbrella parameters across all columns...'):
                            try:
                                umbrella_analysis, excel_output, excel_filename = analyze_all_umbrella_parameters(df)
                                
                                # Show Excel preview first
                                if excel_output:
                                    st.subheader("📊 Excel Analysis Preview")
                                    
                                    # Add download button at the top
                                    try:
                                        # Generate timestamp for unique key
                                        download_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        st.download_button(
                                            label="📥 Download Excel Analysis",
                                            data=excel_output.getvalue(),
                                            file_name=excel_filename,
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                            key=f"download_umbrella_analysis_{download_timestamp}"
                                        )
                                    except Exception as e:
                                        st.error(f"Error creating download button: {e}")
                                        logging.error(f"Error creating download button: {e}")
                                    
                                    # Add custom CSS for better table display
                                    st.markdown("""
                                        <style>
                                        .stDataFrame {
                                            background-color: white;
                                            border: 1px solid #ddd;
                                            border-radius: 5px;
                                            padding: 10px;
                                            margin: 10px 0;
                                        }
                                        .stDataFrame td {
                                            padding: 5px;
                                            border: 1px solid #ddd;
                                        }
                                        .stDataFrame th {
                                            background-color: #D9E1F2;
                                            font-weight: bold;
                                            padding: 5px;
                                            border: 2px solid black;
                                        }
                                        </style>
                                    """, unsafe_allow_html=True)
                                    
                                    # Display Parameter Values sheet
                                    st.subheader("📊 Parameter Values")
                                    try:
                                        excel_output.seek(0)
                                        excel_df = pd.read_excel(excel_output, sheet_name="Parameter Values")
                                        
                                        # Style the DataFrame
                                        def highlight_rows(x):
                                            return ['background-color: #F2F2F2' if i % 2 == 0 else 'background-color: #FFFFFF' for i in range(len(x))]
                                        
                                        styled_df = excel_df.style.apply(highlight_rows, axis=1)
                                        st.dataframe(styled_df, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error displaying Parameter Values: {e}")
                                        logging.error(f"Error displaying Parameter Values: {e}")
                                    
                                    # Display Summary sheet
                                    st.subheader("📊 Summary")
                                    try:
                                        excel_output.seek(0)
                                        summary_df = pd.read_excel(excel_output, sheet_name="Summary")
                                        
                                        # Style the summary DataFrame
                                        def highlight_summary_rows(x):
                                            return ['background-color: #F2F2F2' if i % 2 == 0 else 'background-color: #FFFFFF' for i in range(len(x))]
                                        
                                        styled_summary = summary_df.style.apply(highlight_summary_rows, axis=1)
                                        st.dataframe(styled_summary, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error displaying Summary: {e}")
                                        logging.error(f"Error displaying Summary: {e}")
                                    
                                    st.info("💡 The Excel file has been generated with color-coded sections for better visualization.")
                                    st.markdown("---")  # Add a separator
                                
                                # Then show the markdown analysis
                                if umbrella_analysis:
                                    st.markdown("### 📊 Detailed Analysis")
                                    st.markdown(umbrella_analysis)
                                    
                            except Exception as e:
                                st.error(f"❌ Error analyzing umbrella parameters: {e}")
                                logging.error(f"❌ Error analyzing umbrella parameters: {e}")
                    
                with button_col2:
                    if st.button("🚀 Analyze Product Data", type="primary"):
                        with st.spinner('⏳ Analyzing product data...'):
                            try:
                                # First, perform the standard Amazon product analysis
                                analysis_result = analyze_amazon_product_data(df, analysis_column)
                                if analysis_result:
                                    st.markdown("### 📊 Analysis Results")
                                    st.markdown(analysis_result)
                                
                                # If there's a custom query, process it
                                if custom_query.strip():
                                    st.markdown("### 🔍 Custom Query Analysis")
                                    response_content = ask_openai(custom_query, df, analysis_column)
                                    st.markdown(response_content)
                                
                                # Add the query to recent searches if it's not empty
                                if custom_query.strip() and custom_query not in st.session_state.recent_searches:
                                    st.session_state.recent_searches.insert(0, custom_query)
                                    st.session_state.recent_searches = st.session_state.recent_searches[:5]
                                    
                            except Exception as e:
                                st.error(f"❌ Error analyzing product data: {e}")
                                logging.error(f"❌ Error analyzing product data: {e}")
            
            with col2:
                st.subheader("Analysis Guide")
                st.info("""
                This analysis will provide:
                1. Top 20 most common keywords
                2. Keyword categorization
                3. Product attributes
                4. Market insights
                5. Custom query analysis
                """)
                
                st.subheader("💡 Tips")
                st.markdown("""
                - For best results, analyze columns like:
                  - Product descriptions
                  - Bullet points
                  - Features
                  - Categories
                """)
                
                st.subheader("🔍 Umbrella Parameters")
                st.markdown("""
                Click "Analyze All Umbrella Parameters" to:
                - View standard parameters across all columns
                - See parameter distribution
                - Get recommendations for improvements
                """)
                
                st.subheader("📝 Custom Queries")
                st.markdown("""
                You can ask specific questions about:
                - Keyword frequency
                - Product attributes
                - Market trends
                - Any other data insights
                """)

            # Display Recent Searches
            if st.session_state.recent_searches:
                st.subheader("🔍 Recent Searches")
                for search in st.session_state.recent_searches:
                    if st.button(f"📝 {search}", key=f"recent_{search}"):
                        with st.spinner('⏳ Processing your request...'):
                            try:
                                response_content = ask_openai(search, df, analysis_column)
                                st.markdown(response_content)
                            except Exception as e:
                                st.error(f"❌ Error processing request: {e}")
                                logging.error(f"❌ Error processing request: {e}")
