import time
from flask import Flask, request, jsonify, render_template, url_for
from concurrent.futures import ThreadPoolExecutor
from graph_rag.dav_g_r import process_main_model
from lazy_graph_rag.dav_l_g_r import process_secondary_model

app = Flask(__name__)

@app.route("/")
def index():
    """
    Renders the index.html template as the main page.
    """
    return render_template("index.html")

@app.route("/process_pdf", methods=["POST"])
def process_pdf():
    """
    Handles PDF uploads and processes them using both the graph_rag and lazy_graph_rag models concurrently.
    """
    start_time = time.time()
    uploaded_files = request.files.getlist("pdf")

    if not uploaded_files:
        return jsonify({"response": "No files uploaded.", "processing_time": time.time() - start_time}), 400

    # Function to process a single file with both RAG models
    def process_file(file):
        try:
            main_start = time.time()
            main_result = process_main_model(file)
            main_time = time.time() - main_start

            lazy_start = time.time()
            lazy_result = process_secondary_model(file)
            lazy_time = time.time() - lazy_start

            return {
                "main_rag": {
                    "summary": main_result["summary"],
                    "graph_url": url_for("static", filename="graph_rag.html"),
                    "processing_time": main_time,
                },
                "lazy_rag": {
                    "summary": lazy_result["summary"],
                    "graph_url": url_for("static", filename="lazy_graph_rag.html"),
                    "processing_time": lazy_time,
                },
            }
        except Exception as e:
            return {"error": f"Error processing file: {str(e)}"}

    # Process all files concurrently
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file) for file in uploaded_files]
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                results.append({"error": f"Processing failed: {str(e)}"})

    # Overall processing time
    overall_processing_time = time.time() - start_time

    return jsonify({
        "results": results,
        "processing_time": overall_processing_time,
    })

if __name__ == '__main__':
   from waitress import serve
   serve(app, host='0.0.0.0', port=8080)
    
