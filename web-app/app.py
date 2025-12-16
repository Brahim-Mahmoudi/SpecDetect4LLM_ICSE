# app.py
from flask import Flask, render_template, request, send_file, redirect
from pathlib import Path
import json
import shutil
import tempfile
import os
from specdetect4llm import discover_available_rules, run_analysis, RULES_ROOT

app = Flask(__name__)
# Maximum uploaded file size (e.g., 500MB)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

RULE_METADATA = {
    "R25": "LLM Temperature Not Explicitly Set",
    "R26": "LLM Version Pinning Not Explicitly Set Read",
    "R27": "LLM With No System Message",
    "R28": "LLM Calls Without Bounded Metrics",
    "R29": "No Structured Output in Pipeline",
    "R30": "Reasoning Effort Not Explicitly Set",
    "R31": "Raw Vision Payload",
    "R32": "Overspecified Sampling Parameters",
    "R33": "Anonymous Inference Cal",
    # Ajoutez ici d'autres règles si nécessaire, ex: "R24": "Nom de la règle 24"
    "PARSE_ERROR": "Erreur d'Analyse (Fichier Invalide)"
}

# app.py (Focus on error handling and responses)

@app.route('/', methods=['GET', 'POST'])
def index():
    available_rules = discover_available_rules(RULES_ROOT)
    error = None
    
    if request.method == 'POST':
        # ... (All initial checks and initialization of zip_file and selected_rules) ...
        zip_file = request.files.get('project_zip')
        selected_rules = request.form.getlist('rules')

        # If basic checks fail, return to the index with an error.
        if not zip_file or zip_file.filename == '' or not selected_rules:
            error = "Please select a file and at least one rule."
            return render_template('index.html', rules=available_rules, error=error)

        temp_dir_obj = None
        
        try:
            # --- ATTEMPT ANALYSIS ---
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_path = Path(temp_dir_obj.name)
            zip_path = temp_path / "project.zip"
            zip_file.save(zip_path)

            project_dir = temp_path / "project_extracted"
            shutil.unpack_archive(zip_path, project_dir)

            # Starting analysis. This is where the 'summary' exception could be raised.
            results, total_files, summary = run_analysis(project_dir, selected_rules)
            
            # 1. PREPARE ON SUCCESS
            results_json_str = json.dumps(results, indent=2, ensure_ascii=False)
            
            # SUCCESS: Return results page immediately
            return render_template(
                'results.html',
                results=results,
                total_files=total_files,
                summary=summary, 
                project_extracted_path=str(project_dir),
                project_name=zip_file.filename,
                results_json=results_json_str,
                rule_metadata=RULE_METADATA
            )
            
        except shutil.ReadError:
            error = "Error: The file is not a valid ZIP/TAR or is corrupted."
            
        except Exception as e:
            # FATAL ERROR (includes run_analysis failure)
            print(f"FATAL ERROR DURING ANALYSIS: {e}") 
            error = f"Internal analysis error: {e}"
            
        finally:
            # 2. Cleanup
            if temp_dir_obj:
                try:
                    temp_dir_obj.cleanup() 
                except Exception:
                    pass

        # 3. FAILURE: If execution reaches here (after an except), return to the index.
        # This ensures we do not render results.html without the required variables.
        if error:
            return render_template('index.html', rules=available_rules, error=error, rule_metadata=RULE_METADATA)
                
    # 4. GET requests
    return render_template('index.html', rules=available_rules, error=error, rule_metadata=RULE_METADATA)

# Route for JSON export (unchanged)
@app.route('/download_json', methods=['POST'])
def download_json():
    # ... (download_json code remains the same as previously) ...
    results_json = request.form['results_json']
    
    temp_json_path = Path(tempfile.gettempdir()) / "specdetect_results.json"
    with open(temp_json_path, 'w', encoding='utf-8') as f:
        f.write(results_json)

    return send_file(
        temp_json_path,
        as_attachment=True,
        download_name='specdetect_results.json',
        mimetype='application/json'
    )

if __name__ == '__main__':
    app.run(debug=True)