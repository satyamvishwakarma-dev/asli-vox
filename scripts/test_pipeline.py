import streamlit as st

from scripts.components import run_tests

st.title("Asli-Vox")
st.subheader("Pipeline test dashboard")


test_input = st.text_area(
    "Test input",
    value="Hello from Asli-Vox",
    help="Enter text to use for the smoke test.",
)

if st.button("Run tests", type="primary"):
    results = run_tests(test_input)
    passed = sum(result[1] for result in results)

    st.write(f"**Result:** {passed}/{len(results)} tests passed")
    for name, success, message in results:
        if success:
            st.success(f"PASS — {name}")
        else:
            st.error(f"FAIL — {name}: {message}")
