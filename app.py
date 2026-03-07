import streamlit as st
import traceback

# Import our processing modules
from pdf_loader import extract_text_from_pdf
from chunking import chunk_text
from embedding import get_embeddings
from vector_store import create_vector_store, search_chunks
from summarizer import generate_summary, answer_question
from image_extractor import extract_images_from_pdf

# ---------------------------------------------------------------------------
# Setup & Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Paper Companion", page_icon="📚", layout="wide")

st.title("📚 Research Paper Companion")
st.caption("Seamlessly summarize papers, extract figures, and chat with your documents.")

# ---------------------------------------------------------------------------
# Sidebar Settings & Upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 1. Upload Document")
    uploaded_pdf = st.file_uploader("Drop your PDF here", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_pdf:
        st.divider()
        st.header("⚙️ 2. Summary Preferences")
        length_option = st.selectbox("Length", ["Short", "Medium", "Long"], index=1)
        style_option = st.selectbox("Style", ["Paragraph", "Bullet Points", "Key Insights", "Research Notes"], index=0)
        tone_option = st.selectbox("Tone", ["Simple Explanation", "Academic Style", "Technical Explanation"], index=0)
        
        process_btn = st.button("Process & Summarize", type="primary", use_container_width=True)
    else:
        # Fallback to avoid NameError if someone tries to access these when no file is uploaded
        length_option, style_option, tone_option = "Medium", "Paragraph", "Simple Explanation"

    # --- Export Section ---
    if st.session_state.get("processed", False):
        st.divider()
        st.header("💾 3. Export Data")
        
        # Build the markdown content
        export_content = f"# Executive Summary\n\n{st.session_state.get('summary', '')}\n\n---\n\n# Chat History\n\n"
        for msg in st.session_state.get('messages', []):
            role_header = "🧑‍💻 User" if msg["role"] == "user" else "🤖 AI Assistant"
            export_content += f"### {role_header}\n{msg['content']}\n\n"
            
        st.download_button(
            label="📥 Download Notes (.md)",
            data=export_content,
            file_name="research_notes.md",
            mime="text/markdown",
            use_container_width=True
        )

# ---------------------------------------------------------------------------
# Main Application Flow
# ---------------------------------------------------------------------------

# Initialize session state for interactive elements
if "processed" not in st.session_state:
    st.session_state.processed = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "used_prefs" not in st.session_state:
    st.session_state.used_prefs = {"tone": "Simple Explanation", "style": "Paragraph"}
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# Reset state if a new file is uploaded
if uploaded_pdf:
    file_id = f"{uploaded_pdf.name}_{uploaded_pdf.size}"
    if st.session_state.last_uploaded_file != file_id:
        st.session_state.processed = False
        st.session_state.messages = []
        st.session_state.last_uploaded_file = file_id
        if "summary" in st.session_state: del st.session_state.summary
        if "chunks" in st.session_state: del st.session_state.chunks
        if "index" in st.session_state: del st.session_state.index
        if "extracted_images" in st.session_state: del st.session_state.extracted_images

# If the user clicks Process in the sidebar
if uploaded_pdf and 'process_btn' in locals() and process_btn:
    try:
        # 1. Heavy processing (Cached)
        # We use a nested spinner approach: only show the big one if we don't have chunks/index yet
        if "chunks" not in st.session_state or "index" not in st.session_state:
            with st.spinner("Building knowledge base from PDF..."):
                full_text = extract_text_from_pdf(uploaded_pdf)
                chunks = chunk_text(full_text, chunk_size=200, overlap=40)
                embeddings = get_embeddings(chunks)
                index = create_vector_store(embeddings)
                
                st.session_state.chunks = chunks
                st.session_state.index = index
        
        # Always re-extract images if not present (cached anyway)
        if "extracted_images" not in st.session_state:
            with st.spinner("Extracting figures..."):
                st.session_state.extracted_images = extract_images_from_pdf(uploaded_pdf)
            
        # 2. Retrieve & Summarize (Always re-run on button click to reflect new prefs)
        with st.spinner("Fetching relevant sections..."):
            query = "Abstract, methodology, key findings, important insights, and conclusion summary of the research paper."
            retrieved_chunks = search_chunks(query, st.session_state.index, st.session_state.chunks, k=5)
            
        st.subheader("✨ Executive Summary")
        st.session_state.used_prefs = {"tone": tone_option, "style": style_option}
        
        summary_gen = generate_summary(retrieved_chunks, style_option, length_option, tone_option)
        st.session_state.summary = st.write_stream(summary_gen)
            
        st.session_state.processed = True
        st.rerun() # Refresh to show tabs
        
    except Exception as e:
        st.error(f"Failed to process document: {e}")
        st.session_state.processed = False
        traceback.print_exc()

# ---------------------------------------------------------------------------
# Interactive Canvas
# ---------------------------------------------------------------------------
if st.session_state.processed:
    # Use tabs for a clean, interactive UI
    tab1, tab2, tab3 = st.tabs(["📝 Summary", "📊 Figures", "💬 Chat & Equations"])
    
    with tab1:
        st.subheader("✨ Executive Summary")
        used_tone = st.session_state.used_prefs.get("tone", "Simple Explanation")
        used_style = st.session_state.used_prefs.get("style", "Paragraph")
        st.caption(f"Tone: **{used_tone}** | Format: **{used_style}**")
        with st.container(border=True):
            st.markdown(st.session_state.summary)

    with tab2:
        st.subheader("🖼️ Extracted Figures")
        if st.session_state.extracted_images:
            cols = st.columns(2)
            for idx, img_data in enumerate(st.session_state.extracted_images):
                with cols[idx % 2]:
                    with st.container(border=True):
                        st.image(img_data["path"])
                        st.caption(f"**Figure {idx+1}:** {img_data['caption']}")
        else:
            st.info("No diagrams or charts were cleanly extracted from this PDF.")

    with tab3:
        st.subheader("💬 Chat with Document")
        st.caption("Ask questions, request section breakdowns, or ask to explain mathematical equations.")
        
        # Display chat messages from history
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
        # Suggested questions (only show if no chat history yet to keep UI clean)
        if not st.session_state.messages:
            st.markdown("**Suggested Questions:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Explain the methodology", use_container_width=True):
                    st.session_state.suggested_q = "Can you explain the detailed methodology used in this paper?"
            with col2:
                if st.button("What are the limitations?", use_container_width=True):
                    st.session_state.suggested_q = "What are the limitations and future work mentioned in the research?"
            with col3:
                if st.button("Summarize the results", use_container_width=True):
                    st.session_state.suggested_q = "What are the key results, metrics, and conclusions?"
            
        # Chat Input
        user_question = st.chat_input("E.g., Can you explain the main formula used for the loss function?")
        
        # Override with suggested question if clicked
        if st.session_state.get("suggested_q"):
            user_question = st.session_state.suggested_q
            st.session_state.suggested_q = None

        if user_question:
            # Display user message instantly
            st.chat_message("user").write(user_question)
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            # Display assistant response
            with st.chat_message("assistant"):
                # Actively search the FAISS vector database
                context_chunks = search_chunks(user_question, st.session_state.index, st.session_state.chunks, k=6)
                # Pass the chunks to the LLM
                answer_gen = answer_question(user_question, context_chunks)
                answer = st.write_stream(answer_gen)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                    
else:
    if not uploaded_pdf:
        st.info("👋 **Welcome!** Please drag and drop a research paper into the sidebar to begin.")
