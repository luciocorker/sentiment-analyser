# -------------- SECTION 1: IMPORTS --------------
import streamlit as st
from textblob import TextBlob
import pandas as pd
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import os
import plotly.express as px
import plotly.graph_objects as go
from nrclex import NRCLex
import nltk
import ssl

# Fix SSL certificate issues for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data with error handling
def download_nltk_data():
    """Download required NLTK data with comprehensive error handling."""
    required_data = [
        'punkt',
        'averaged_perceptron_tagger',
        'brown',
        'conll2000',
        'punkt_tab'  # New tokenizer for newer NLTK versions
    ]
    
    for data_name in required_data:
        try:
            nltk.data.find(f'tokenizers/{data_name}')
        except LookupError:
            try:
                nltk.download(data_name, quiet=True)
            except Exception as e:
                st.warning(f"Could not download {data_name}: {e}")
        except Exception:
            try:
                nltk.data.find(f'taggers/{data_name}')
            except LookupError:
                try:
                    nltk.download(data_name, quiet=True)
                except Exception as e:
                    st.warning(f"Could not download {data_name}: {e}")
            except Exception:
                try:
                    nltk.data.find(f'corpora/{data_name}')
                except LookupError:
                    try:
                        nltk.download(data_name, quiet=True)
                    except Exception as e:
                        st.warning(f"Could not download {data_name}: {e}")

# Download NLTK data at startup
download_nltk_data()

import json
from io import BytesIO
try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

# -------------- SECTION 2: PAGE CONFIGURATION --------------
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def setup_page():
    """Configure the Streamlit page layout and appearance"""
    # This function can now be empty or removed, but keep for compatibility
    pass


# -------------- SECTION 3: MAIN APP LAYOUT --------------
def create_sidebar():
    """Create the sidebar with information about the app"""
    st.sidebar.header("About This App")
    st.sidebar.markdown(
        '<span style="color:white;">This app uses the <b>TextBlob</b> library to perform basic sentiment analysis.</span>',
        unsafe_allow_html=True
    )

    st.sidebar.header("How It Works")
    st.sidebar.markdown(
        """
        1. You enter text in the text area.
        2. Click the 'Analyze Sentiment' button.
        3. The app uses `TextBlob` to calculate:
            * **Polarity**: Negative (-1) to Positive (+1)
            * **Subjectivity**: Objective (0) to Subjective (1)
        4. It classifies the sentiment based on the polarity score.
        """
    )


def create_main_section():
    """Create the main app title and description"""
    st.title("💬 Simple Sentiment Analysis App")
    st.write(
        "Enter some text below, and we'll analyze its sentiment (Positive, Negative, or Neutral) "
        "using the TextBlob library."
    )
    st.markdown("---")


# -------------- SECTION 4: TEXT INPUT AREA --------------
def create_text_input():
    """Create and return the text input area"""
    # Only set value if not in session_state to avoid Streamlit warning
    # Remove the else branch to avoid duplicate key usage
    default_value = ""
    return st.text_area(
        "Type or paste your text here:",
        value=st.session_state.get("user_input_text", default_value),
        height=150,
        placeholder="E.g., 'Streamlit makes building web apps so easy and fun!'",
        key="user_input_text"
    )

# -------------- SECTION 5: SENTIMENT ANALYSIS LOGIC --------------
def analyze_sentiment(text):
    """
    Analyze text sentiment using TextBlob.

    Parameters:
        text (str): The text to analyze

    Returns:
        tuple: (polarity, subjectivity, sentiment_label, emoji)
    """
    # Handle empty input
    if not text:
        return 0.0, 0.0, "Neutral", "😐"

    # Process text with TextBlob
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
    subjectivity = blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)

    # Classify sentiment based on polarity
    if polarity > 0.1:
        sentiment_label = "Positive"
        emoji = "😊"
    elif polarity < -0.1:
        sentiment_label = "Negative"
        emoji = "😠"
    else:
        sentiment_label = "Neutral"
        emoji = "😐"

    return polarity, subjectivity, sentiment_label, emoji

def get_sentiment_confidence(polarity):
    """Return confidence scores for negative, neutral, positive."""
    # Softmax-like mapping for demonstration
    import numpy as np
    neg = max(0, -polarity)
    pos = max(0, polarity)
    neu = 1 - abs(polarity)
    scores = np.array([neg, neu, pos])
    scores = np.clip(scores, 0, None)
    if scores.sum() == 0:
        scores = np.array([0.33, 0.34, 0.33])
    else:
        scores = scores / scores.sum()
    return {"Negative": scores[0], "Neutral": scores[1], "Positive": scores[2]}

def extract_keywords(text):
    """Extract noun phrases as keywords using TextBlob with error handling."""
    try:
        blob = TextBlob(text)
        return list(blob.noun_phrases)
    except Exception as e:
        # Fallback: extract basic keywords using simple word filtering
        st.warning("Advanced keyword extraction unavailable. Using basic extraction.")
        words = text.lower().split()
        # Simple keyword extraction: words longer than 3 characters that aren't common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        keywords = [word.strip('.,!?";()[]{}') for word in words if len(word) > 3 and word not in stop_words]
        return list(set(keywords))[:10]  # Return unique keywords, limit to 10

def analyze_emotions(text):
    """Return emotion scores using NRCLex with error handling."""
    if not text.strip():
        return {}
    try:
        emotion_obj = NRCLex(text)
        # Get raw emotion scores (counts)
        raw_scores = emotion_obj.raw_emotion_scores
        total = sum(raw_scores.values())
        # Normalize to proportions
        if total > 0:
            norm_scores = {k: v / total for k, v in raw_scores.items()}
        else:
            norm_scores = {}
        return norm_scores
    except Exception as e:
        st.warning("Emotion analysis unavailable.")
        return {}

# -------------- SECTION 6: ANALYSIS & RESULTS --------------
def perform_analysis(text):
    """Perform sentiment analysis and display results"""
    # Only analyze if there's text
    if not text:
        st.warning("⚠️ Please enter some text above before analyzing.")
        return

    # --- Loading Line with Percentage ---
    import time
    progress_placeholder = st.empty()
    # Reduce sleep time or skip for large texts
    if len(text) > 2000:
        # For very large text, skip artificial delay
        progress_placeholder.progress(1.0, text="Analyzing the text... 100%")
    else:
        for percent in range(0, 101, 5):
            progress_placeholder.progress(percent / 100.0, text=f"Analyzing the text... {percent}%")
            time.sleep(0.005)  # Reduced from 0.02 to 0.005 for faster UI
    progress_placeholder.empty()
    # Show analysis in progress
    polarity, subjectivity, sentiment, emoji = analyze_sentiment(text)

    st.subheader("📊 Analysis Results")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Overall Sentiment",
            value=f"{sentiment} {emoji}"
        )
    with col2:
        st.metric(
            label="Polarity Score",
            value=f"{polarity:.2f}",
            help="Ranges from -1 (very negative) to +1 (very positive). Closer to 0 is more neutral."
        )

    st.metric(
        label="Subjectivity Score",
        value=f"{subjectivity:.2f}",
        help="Ranges from 0 (very objective) to 1 (very subjective)."
    )

    # --- Confidence Pie Chart ---
    st.subheader("🔵 Sentiment Confidence")
    conf = get_sentiment_confidence(polarity)
    fig = px.pie(
        names=list(conf.keys()),
        values=list(conf.values()),
        color=list(conf.keys()),
        color_discrete_map={"Negative": "#e57373", "Neutral": "#ffd54f", "Positive": "#64b5f6"},
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    * **Sentiment:** The overall feeling expressed (Positive, Negative, or Neutral).
    * **Polarity:** How positive or negative the text is.
    * **Subjectivity:** How much the text expresses personal opinions vs. factual information.
    """)

    # --- Explanation Feature ---
    keywords = extract_keywords(text)
    emotion_scores = analyze_emotions(text)
    explanation = explain_sentiment(text, polarity, subjectivity, keywords, emotion_scores)
    st.markdown(f"**Why this score?** {explanation}")

    # --- Keyword Extraction ---
    st.subheader("🔑 Sentiment Drivers (Keywords)")
    if keywords:
        st.markdown(
            " ".join([f"<span style='background-color:#bbdefb; color:#0d47a1; padding:2px 6px; border-radius:6px; margin:2px'>{kw}</span>" for kw in keywords]),
            unsafe_allow_html=True
        )
    else:
        st.write("No significant keywords found.")

    # --- Emotion Analysis ---
    st.subheader("🎭 Emotion Analysis")
    if emotion_scores:
        emotion_df = pd.DataFrame({
            "Emotion": list(emotion_scores.keys()),
            "Score": list(emotion_scores.values())
        }).sort_values("Score", ascending=False)
        st.bar_chart(emotion_df.set_index("Emotion"))
        top_emotions = emotion_df[emotion_df["Score"] > 0].Emotion.tolist()
        if top_emotions:
            st.write("**Top emotions:**", ", ".join(top_emotions))
    else:
        st.write("No strong emotions detected in the text.")

    # --- Visualization Section ---
    st.subheader("📈 Sentiment Visualization")
    df = pd.DataFrame({
        'Metric': ['Polarity', 'Subjectivity'],
        'Score': [polarity, subjectivity]
    })
    st.bar_chart(df.set_index('Metric'))

def explain_sentiment(text, polarity, subjectivity, keywords, emotions):
    """Generate an explanation for the sentiment score."""
    explanation = []
    if polarity > 0.1:
        explanation.append("The text expresses positive sentiment, likely due to positive words or phrases.")
    elif polarity < -0.1:
        explanation.append("The text expresses negative sentiment, likely due to negative words or phrases.")
    else:
        explanation.append("The text is neutral, with a balance of positive and negative expressions.")
    if subjectivity > 0.5:
        explanation.append("The text is subjective, indicating personal opinions or feelings.")
    else:
        explanation.append("The text is objective, focusing on facts rather than opinions.")
    if keywords:
        explanation.append(f"Key phrases influencing sentiment: {', '.join(keywords[:5])}")
    if emotions:
        top_emotions = sorted(emotions.items(), key=lambda x: -x[1])
        top_emotions = [f"{k} ({v:.2f})" for k, v in top_emotions if v > 0][:3]
        if top_emotions:
            explanation.append(f"Top detected emotions: {', '.join(top_emotions)}")
    return " ".join(explanation)

# -------------- SECTION 7: AUDIO TEXT EXTRACTION --------------
def extract_text_from_audio(audio_file):
    recognizer = sr.Recognizer()
    # Save uploaded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio.flush()
        with sr.AudioFile(temp_audio.name) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
        except Exception:
            text = ""
    os.unlink(temp_audio.name)
    return text

# -------------- SECTION 8: ENHANCED BATCH EMOTION VISUALIZATIONS --------------
def create_batch_emotion_visualizations(df):
    """Create comprehensive emotion visualizations for batch processing results."""
    st.subheader("📊 Batch Emotion Analysis Visualizations")
    
    # Collect all emotion data for aggregate analysis
    all_emotions_data = []
    
    for idx, row in df.iterrows():
        emotion_scores = analyze_emotions(row["Text"])
        if emotion_scores:
            emotion_scores['Text_ID'] = f"Text {idx+1}"
            all_emotions_data.append(emotion_scores)
    
    if not all_emotions_data:
        st.warning("No emotions detected in the batch texts.")
        return
    
    # Create tabs for different visualization types
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Individual Analysis", "🔥 Heatmap", "📊 Aggregate Analysis", "🎯 Comparison"])
    
    with tab1:
        st.subheader("Individual Text Emotion Analysis")
        for idx, row in df.iterrows():
            with st.expander(f"📝 Text {idx+1}: {row['Text'][:50]}..." + ("..." if len(row['Text']) > 50 else "")):
                emotion_scores = analyze_emotions(row["Text"])
                if emotion_scores:
                    # Create emotion DataFrame
                    emotion_df = pd.DataFrame({
                        "Emotion": list(emotion_scores.keys()),
                        "Score": list(emotion_scores.values())
                    }).sort_values("Score", ascending=False)
                    
                    # Filter out zero scores for cleaner visualization
                    emotion_df_filtered = emotion_df[emotion_df["Score"] > 0]
                    
                    if len(emotion_df_filtered) > 0:
                        # Create horizontal bar chart using Plotly
                        fig = px.bar(
                            emotion_df_filtered, 
                            x="Score", 
                            y="Emotion", 
                            orientation='h',
                            title=f"Emotions in Text {idx+1}",
                            color="Score",
                            color_continuous_scale="Viridis",
                            text="Score"
                        )
                        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                        fig.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show top emotions as metrics
                        top_emotions = emotion_df_filtered.head(3)
                        cols = st.columns(min(3, len(top_emotions)))
                        for i, (_, emotion_row) in enumerate(top_emotions.iterrows()):
                            if i < len(cols):
                                cols[i].metric(
                                    label=f"🎭 {emotion_row['Emotion'].title()}",
                                    value=f"{emotion_row['Score']:.3f}"
                                )
                    else:
                        st.info("No significant emotions detected in this text.")
                else:
                    st.write("No emotions detected in this text.")
    
    with tab2:
        st.subheader("Emotion Heatmap Across All Texts")
        if len(all_emotions_data) > 1:
            # Create emotion matrix for heatmap
            emotion_names = set()
            for emotions in all_emotions_data:
                emotion_names.update(emotions.keys())
            emotion_names.discard('Text_ID')  # Remove Text_ID if present
            emotion_names = sorted(list(emotion_names))
            
            # Build matrix
            matrix_data = []
            text_labels = []
            for idx, emotions in enumerate(all_emotions_data):
                row = []
                for emotion in emotion_names:
                    row.append(emotions.get(emotion, 0))
                matrix_data.append(row)
                text_labels.append(f"Text {idx+1}")
            
            # Create heatmap using Plotly
            fig = px.imshow(
                matrix_data,
                x=emotion_names,
                y=text_labels,
                color_continuous_scale="RdYlBu_r",
                title="Emotion Intensity Heatmap",
                aspect="auto"
            )
            fig.update_layout(
                height=max(400, len(text_labels) * 30),
                xaxis_title="Emotions",
                yaxis_title="Texts"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("📈 Heatmap Insights")
            matrix_df = pd.DataFrame(matrix_data, columns=emotion_names, index=text_labels)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Most Emotional Text:**")
                most_emotional = matrix_df.sum(axis=1).idxmax()
                st.metric("Text", most_emotional, f"{matrix_df.sum(axis=1).max():.3f}")
                
            with col2:
                st.write("**Dominant Emotion Overall:**")
                dominant_emotion = matrix_df.sum(axis=0).idxmax()
                st.metric("Emotion", dominant_emotion.title(), f"{matrix_df.sum(axis=0).max():.3f}")
        else:
            st.info("Heatmap requires at least 2 texts for comparison.")
    
    with tab3:
        st.subheader("Aggregate Emotion Analysis")
        # Combine all emotions across texts
        combined_emotions = {}
        for emotions in all_emotions_data:
            for emotion, score in emotions.items():
                if emotion != 'Text_ID':
                    combined_emotions[emotion] = combined_emotions.get(emotion, 0) + score
        
        # Calculate averages
        avg_emotions = {k: v/len(all_emotions_data) for k, v in combined_emotions.items()}
        
        # Create aggregate visualization
        agg_df = pd.DataFrame({
            "Emotion": list(avg_emotions.keys()),
            "Average_Score": list(avg_emotions.values())
        }).sort_values("Average_Score", ascending=False)
        
        # Filter significant emotions
        significant_emotions = agg_df[agg_df["Average_Score"] > 0.01]
        
        if len(significant_emotions) > 0:
            # Pie chart for emotion distribution
            fig_pie = px.pie(
                significant_emotions,
                values="Average_Score",
                names="Emotion",
                title="Overall Emotion Distribution Across All Texts",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Bar chart for detailed scores
            fig_bar = px.bar(
                significant_emotions,
                x="Emotion",
                y="Average_Score",
                title="Average Emotion Scores Across All Texts",
                color="Average_Score",
                color_continuous_scale="Plasma",
                text="Average_Score"
            )
            fig_bar.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Top emotions summary
            st.subheader("🏆 Top Emotions Summary")
            top_5 = significant_emotions.head(5)
            for _, row in top_5.iterrows():
                st.metric(
                    label=f"🎭 {row['Emotion'].title()}",
                    value=f"{row['Average_Score']:.4f}",
                    delta=f"Rank #{_+1}"
                )
        else:
            st.info("No significant emotions detected across the texts.")
    
    with tab4:
        st.subheader("Emotion Comparison Between Texts")
        if len(df) >= 2:
            # Allow user to select texts for comparison
            text_options = [f"Text {i+1}: {row['Text'][:30]}..." for i, row in df.iterrows()]
            selected_texts = st.multiselect(
                "Select texts to compare (max 5):",
                options=text_options,
                default=text_options[:min(3, len(text_options))],
                max_selections=5
            )
            
            if len(selected_texts) >= 2:
                # Get selected indices
                selected_indices = [text_options.index(text) for text in selected_texts]
                
                # Create comparison data
                comparison_data = []
                for idx in selected_indices:
                    emotions = analyze_emotions(df.iloc[idx]["Text"])
                    for emotion, score in emotions.items():
                        if score > 0:  # Only include emotions with scores > 0
                            comparison_data.append({
                                "Text": f"Text {idx+1}",
                                "Emotion": emotion.title(),
                                "Score": score
                            })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Create grouped bar chart
                    fig = px.bar(
                        comparison_df,
                        x="Emotion",
                        y="Score",
                        color="Text",
                        barmode="group",
                        title="Emotion Comparison Between Selected Texts",
                        text="Score"
                    )
                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Radar chart comparison
                    st.subheader("🎯 Emotion Profile Radar Chart")
                    
                    # Create radar chart data
                    fig_radar = go.Figure()
                    
                    all_emotions = sorted(list(set(comparison_df["Emotion"])))
                    colors = px.colors.qualitative.Set1
                    
                    for i, text_id in enumerate(comparison_df["Text"].unique()):
                        text_data = comparison_df[comparison_df["Text"] == text_id]
                        emotion_scores = []
                        for emotion in all_emotions:
                            score = text_data[text_data["Emotion"] == emotion]["Score"].values
                            emotion_scores.append(score[0] if len(score) > 0 else 0)
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=emotion_scores + [emotion_scores[0]],  # Close the polygon
                            theta=all_emotions + [all_emotions[0]],
                            fill='toself',
                            name=text_id,
                            line_color=colors[i % len(colors)]
                        ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, max(comparison_df["Score"]) * 1.1] if len(comparison_df) > 0 else [0, 1]
                            )),
                        title="Emotion Profile Comparison (Radar Chart)",
                        showlegend=True,
                        height=500
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                else:
                    st.info("No emotions detected in the selected texts for comparison.")
            else:
                st.info("Please select at least 2 texts for comparison.")
        else:
            st.info("Comparison requires at least 2 texts.")

# -------------- SECTION 9: CUSTOM STYLES --------------
def set_custom_style():
    st.markdown("""
        <style>
        /* Main background */
        .stApp {
            background: linear-gradient(135deg, #e3f0ff 0%, #b3cfff 100%);
        }
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(135deg, #0d47a1 0%, #1976d2 100%);
            color: #fff;
        }
        /* Sidebar header and info */
        .stSidebar .css-1d391kg, .stSidebar .css-1v0mbdj {
            color: #fff !important;
        }
        /* Title */
        h1 {
            color: #1565c0;
        }
        /* Subheaders */
        h2, h3 {
            color: #1976d2;
        }
        /* Buttons */
        .stButton>button {
            background-color: #1976d2;
            color: #fff;
            border-radius: 8px;
            border: none;
            padding: 0.5em 1.5em;
            font-weight: bold;
            transition: background 0.2s;
        }
        .stButton>button:hover {
            background-color: #0d47a1;
            color: #fff;
        }
        /* Metrics */
        div[data-testid="stMetric"] {
            background: #e3f0ff;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        /* Bar chart background */
        .element-container .stPlotlyChart, .element-container .stAltairChart, .element-container .stVegaLiteChart {
            background: #f5faff;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

# -------------- SECTION 10: MAIN FUNCTION --------------
def main():
    """Main function to run the Streamlit app"""
    # Set up the page configuration
    setup_page()
    set_custom_style()

    create_main_section()
    create_sidebar()

    # Handle clear button BEFORE text area is rendered
    if "clear_text" not in st.session_state:
        st.session_state.clear_text = False

    # Set user_input_text to empty if clear was pressed
    if st.session_state.clear_text:
        st.session_state.user_input_text = ""
        st.session_state.clear_text = False

    # Display the text input area
    st.header("Enter Text for Analysis")
    user_text = create_text_input()

    # --- File Upload Section for Audio ---
    st.header("Or Upload Audio for Analysis")
    uploaded_file = st.file_uploader(
        "Upload audio (.wav, .mp3, .m4a) file",
        type=["wav", "mp3", "m4a"]
    )

    extracted_text = ""
    if uploaded_file:
        filetype = uploaded_file.type
        # --- Audio Player ---
        st.audio(uploaded_file, format=filetype)
        if "audio" in filetype:
            st.info("Extracting text from audio...")
            extracted_text = extract_text_from_audio(uploaded_file)
            st.success("Extracted text from audio:")
            st.write(extracted_text if extracted_text else "No speech detected.")
        else:
            st.warning("Unsupported file type.")

    # Use extracted text if available, else use user_text
    text_to_analyze = extracted_text if extracted_text else user_text

    # --- Place Analyze button ---
    col1, = st.columns([3])
    analyze_clicked = False
    with col1:
        if st.button("Analyze Sentiment ✨", key="analyze_button"):
            analyze_clicked = True

    if analyze_clicked:
        perform_analysis(text_to_analyze)

    # --- Batch Processing Section ---
    st.header("Batch Processing (Multiple Texts)")

    # Option 1: Paste texts
    batch_input = st.text_area(
        "Paste multiple texts here (one per line):",
        height=100,
        placeholder="Enter one text per line for batch analysis...",
        key="batch_input"
    )

    # Option 2: Upload CSV or Excel
    uploaded_batch_file = st.file_uploader(
        "Or upload a CSV or Excel file with a column of texts",
        type=["csv", "xlsx"],
        key="batch_file"
    )

    texts = []
    file_error = False
    if uploaded_batch_file is not None:
        try:
            if uploaded_batch_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_batch_file)
            else:
                df_upload = pd.read_excel(uploaded_batch_file)
            # Try to find a column with 'text' in the name, else use the first column
            text_col = None
            for col in df_upload.columns:
                if "text" in col.lower():
                    text_col = col
                    break
            if text_col is None:
                text_col = df_upload.columns[0]
            texts = df_upload[text_col].astype(str).tolist()
            st.success(f"Loaded {len(texts)} texts from file (column: '{text_col}')")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            file_

        if __name__ == "__main__":
    main()
