import streamlit as st
import uuid

@st.cache_resource
def get_app():
    from pharma_guard import app
    return app

app = get_app()

st.set_page_config(page_title="PharmaGuard AI", page_icon=":test_tube:", layout="centered")

st.title(":test_tube: PharmaGuard AI")
st.caption("Multi-agent RAG compliance co-pilot for pharma IT governance")
st.divider()

task_input = st.text_area(
     "Describe your task in plain English",
     placeholder="e.g Releasing a new LIMS module with electronic batch records and audit trail capabilities",
     height=100
 )

analyze_clicked = st.button("Analyze Task", type="primary")


if analyze_clicked:
    if not task_input.strip():
        st.warning("Please describe a task first.")
    else:
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        with st.spinner("Running 5-agent compliance pipeline..."):
            result = app.invoke({
                "task_description": task_input,
                "task_type": "",
                "applicable_frameworks": [],
                "retrieved_requirements": [],
                "deviations": [],
                "risk_level": "",
                "compliance_brief": {},
                "messages": []
            }, config=config)
        
        st.success("Analysis complete")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Task Type", result["task_type"].title())
        with col2:
            risk_colors = {"low": ":green_circle:", "medium": ":yellow_circle:", "high": ":red_circle:"}
            risk = result["risk_level"].lower()
            st.metric("Risk Level", f"{risk_colors.get(risk, '')} {risk.upper()}")
        
        st.subheader("Applcable Frameworks")
        st.write(",".join(result["applicable_frameworks"]))

        st.subheader("Deviations Found")
        for d in result["deviations"]:
            st.write(f":warning: {d}")

        brief = result["compliance_brief"]

        st.subheader("Summary")
        st.write(brief["summary"])

        col3, col4 = st.columns(2)
        with col3:
            st.subheader(":white_check_mark: Do's")
            for item in brief.get("dos", []):
                st.write(f"- {item}")
        with col4:
            st.subheader(":no_entry_sign: Don'ts")
            for item in brief.get("donts", []):
                st.write(f"- {item}")
    
    st.divider()
    rec = brief["recommendation"].upper()
    rec_colors = {"APPROVE": "success", "DELAY": "warning", "REJECT": "error"}
    getattr(st, rec_colors.get(rec, "info"))(f"**Final Recommendation: {rec}**")