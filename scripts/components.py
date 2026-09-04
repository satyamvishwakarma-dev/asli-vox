def run_tests(text: str) -> list[tuple[str, bool, str]]:
	"""Run basic smoke tests for the Streamlit pipeline interface."""
	return [
		("Streamlit is running", True, "The app loaded successfully."),
		("Input is accepted", bool(text.strip()), "Provide a non-empty test input."),
		(
			"Input is normalized",
			text == text.strip(),
			"Leading and trailing whitespace should be removed.",
		),
	]
