# Human Review Sheet HTML Tool

This tool provides an interactive interface for reviewing commit diffs and entering human labels/rationales.

## How to Run

1. **Start a Local HTTP Server**
   - In the directory containing `artifacts/results/human_review_sheet_review.html`, run:
     
     ```sh
     python3 -m http.server 8000
     ```
   - Or, if you prefer a different port:
     ```sh
     python3 -m http.server 8080
     ```

2. **Open the Review Tool in Your Browser**
   - Go to: [http://localhost:8000/artifacts/results/human_review_sheet_review.html](http://localhost:8000/artifacts/results/human_review_sheet_review.html)


3. **Usage**
   - The tool will load rows from `human_review_sheet.csv`.
   - Click on diff links to view code changes.
   - **Choose your review mode:**
     - **Blind review:** Hide the LLM label/rationale columns by clicking the "Hide LLM Columns" button. Review each diff and make your own decision without seeing the LLM’s answer.
     - **LLM-assisted review:** Show the LLM label/rationale columns by clicking the "Show LLM Columns" button. You can check and validate the LLM’s decision and rationale, or provide your own.
   - Enter your label and rationale for each row.
   - Use the export/download button to save your reviewed data.

---

**Tips:**
- You can switch between modes at any time using the toggle button.
- Your exported CSV will include your GitHub handle and timestamp for attribution.


## Notes
- Opening the HTML file directly (file://) will not work due to browser security restrictions. Always use a local HTTP server.
- Make sure `human_review_sheet.csv` is present in the same directory as the HTML file or adjust the fetch path in the HTML if needed.
