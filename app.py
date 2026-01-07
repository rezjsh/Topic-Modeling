import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Import your project modules
from topic_modeling.config.configuration import ConfigurationManager
from topic_modeling.pipeline.stage_10_inference import InferencePipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="TopicLens | AI Topic Explorer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #4CAF50; color: white; }
    .topic-card { padding: 20px; border-radius: 10px; background-color: white; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- Singleton Loaders ---
@st.cache_resource
def get_config():
    return ConfigurationManager()

@st.cache_resource
def load_inference_pipeline():
    # Now strictly follows the Configuration Manager settings
    cm = get_config()
    return InferencePipeline(cm)

# --- App Logic ---

def main():
    st.sidebar.title("🔮 TopicLens")
    st.sidebar.markdown("---")
    
    menu = st.sidebar.selectbox(
        "Navigation", 
        ["Dashboard & Stats", "Live Inference", "Model Benchmark"]
    )

    if menu == "Dashboard & Stats":
        render_dashboard()
    elif menu == "Live Inference":
        render_inference()
    elif menu == "Model Benchmark":
        render_benchmark()

def render_dashboard():
    st.title("📊 Project Dashboard")
    st.markdown("Overview of the latest trained models and discovered topics.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Models Trained", "4", "+1")
    col2.metric("Best Coherence", "0.58", "LDA")
    col3.metric("Dataset Size", "11.3k", "20Newsgroups")

    st.markdown("### Top Keywords per Topic")
    topics_data = {
        "Space": ["nasa", "orbit", "moon", "launch", "satellite"],
        "Medicine": ["doctor", "health", "cancer", "study", "disease"],
        "Graphics": ["image", "software", "color", "display", "pixel"],
        "Religion": ["god", "church", "faith", "bible", "christ"]
    }

    cols = st.columns(2)
    for i, (name, words) in enumerate(topics_data.items()):
        with cols[i % 2]:
            st.markdown(f"**Topic: {name}**")
            st.code(" | ".join(words))

def render_inference():
    st.title("🔍 Live Topic Inference")
    st.write("Enter unseen text below to identify topics using the active production model.")

    user_input = st.text_area(
        "User Text Input", 
        height=200, 
        placeholder="e.g., The rocket was launched into the lower earth orbit..."
    )

    if st.button("Predict Topic"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            with st.spinner("Running Inference Pipeline..."):
                try:
                    # Load pipeline without needing manual overrides
                    pipeline = load_inference_pipeline()
                    results = pipeline.run_pipeline([user_input])
                    res = results[0]

                    st.success(f"Prediction Complete!")
                    
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.markdown("### Topic Identity")
                        st.metric("Topic ID", res['topic_id'])
                        st.metric("Confidence", f"{res['confidence']:.2%}")
                    
                    with c2:
                        st.markdown("### Topic Visualization")
                        # Generate WordCloud using the synced key 'top_words'
                        text_for_cloud = " ".join(res['top_words'] * 5)
                        wc = WordCloud(background_color="white", colormap="viridis").generate(text_for_cloud)
                        
                        fig, ax = plt.subplots()
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                        
                    st.markdown("---")
                    st.markdown(f"**Top keywords for this prediction:**")
                    st.info(", ".join(res['top_words']))

                except Exception as e:
                    st.error(f"Inference Error: {e}")

def render_benchmark():
    st.title("📈 Model Benchmarking")
    st.write("Comparison of different architectures based on Coherence and Diversity.")

    try:
        data = pd.DataFrame({
            'Model': ['LDA', 'NMF', 'ProdLDA', 'BERTopic'],
            'Coherence': [0.45, 0.42, 0.52, 0.58],
            'Diversity': [0.85, 0.88, 0.75, 0.92]
        })

        st.table(data)
        st.markdown("### Coherence Comparison")
        fig = px.bar(data, x='Model', y='Coherence', color='Model', 
                     text_auto='.2f', title="Higher is Better")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load benchmark data. Error: {e}")

if __name__ == "__main__":
    main()