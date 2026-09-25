import streamlit as st

from src.orchestrator import Orchestrator


st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔎",
    layout="wide"
)


st.title("🔎 Multi-Agent Research System")
st.write(
    "Planner → Researcher → Critic → Writer"
)

st.divider()

query = st.text_area(
    "Enter your research question",
    placeholder="Example: What are the latest developments in Generative AI?",
    height=120
)


if st.button("🚀 Start Research", type="primary"):

    if not query.strip():
        st.warning("Please enter a research question.")
        st.stop()

    with st.spinner("Running multi-agent research pipeline..."):

        try:
            orchestrator = Orchestrator()
            result = orchestrator.run(
                query,
                verbose=False
            )

            st.success("Research completed successfully!")

            st.subheader("📋 Research Subtasks")

            for i, task in enumerate(result.subtasks, 1):
                st.write(f"**{i}.** {task}")

            st.divider()

            st.subheader("📄 Final Research Report")

            st.markdown(result.final_report_markdown)

            st.divider()

            st.subheader("📊 Pipeline Information")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Input Tokens",
                    result.total_input_tokens
                )

            with col2:
                st.metric(
                    "Output Tokens",
                    result.total_output_tokens
                )

            with col3:
                st.metric(
                    "Estimated Cost",
                    f"${result.estimated_cost_usd:.4f}"
                )

            if result.critic_reports:
                st.subheader("🧐 Critic Confidence")

                confidence = result.critic_reports[-1].confidence

                st.progress(
                    min(max(confidence, 0.0), 1.0)
                )

                st.write(
                    f"Confidence: {confidence:.2%}"
                )

        except Exception as e:

            st.error(
                f"An error occurred: {str(e)}"
            )
