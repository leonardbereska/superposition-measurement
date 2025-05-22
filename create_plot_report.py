import os
import glob
import json
import re
import base64
from pathlib import Path
import argparse
from datetime import datetime

def parse_experiment_name(experiment_name):
    """Parse experiment name to extract metadata."""
    # Format: YYYY-MM-DD_HH-MM_model_classes-class_attack
    pattern = r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2})_(\w+)_(\d+)-class_(\w+)"
    match = re.match(pattern, experiment_name)
    
    if match:
        date_str, time_str, model_type, n_classes, attack_type = match.groups()
        date_time = f"{date_str} {time_str.replace('-', ':')}"
        return {
            "date": date_str,
            "time": time_str,
            "datetime": date_time,
            "model_type": model_type,
            "n_classes": int(n_classes),
            "attack_type": attack_type
        }
    return {
        "experiment_name": experiment_name
    }

def read_config_file(experiment_path):
    """Read the config.json file if it exists."""
    config_path = os.path.join(experiment_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def find_plot_pdfs(base_dir, model_type=None, n_classes=None, attack_type=None, date_from=None, date_to=None):
    """Find all PDF files in 'plots' directories under the specified base directory with optional filters."""
    plot_paths = []
    
    # Convert date strings to datetime objects for comparison if provided
    date_from_obj = datetime.strptime(date_from, "%Y-%m-%d") if date_from else None
    date_to_obj = datetime.strptime(date_to, "%Y-%m-%d") if date_to else None
    
    # Find all experiment directories that directly match the pattern
    experiment_dirs = [d for d in glob.glob(f"{base_dir}/*") 
                     if os.path.isdir(d) and not any(subdir in d for subdir in ['capacity', 'older', 'to evaluate'])]
    
    for experiment_dir in experiment_dirs:
        experiment_name = os.path.basename(experiment_dir)
        plots_dir = os.path.join(experiment_dir, "plots")
        
        # Skip if plots directory doesn't exist
        if not os.path.exists(plots_dir):
            continue
        
        # Parse metadata from experiment name
        metadata = parse_experiment_name(experiment_name)
        
        # Read config file for additional metadata
        config = read_config_file(experiment_dir)
        
        # Apply filters
        if model_type and metadata.get('model_type') != model_type:
            continue
            
        if n_classes and metadata.get('n_classes') != n_classes:
            continue
            
        if attack_type and metadata.get('attack_type') != attack_type:
            continue
            
        if date_from_obj and 'datetime' in metadata:
            exp_date = datetime.strptime(metadata['date'], "%Y-%m-%d")
            if exp_date < date_from_obj:
                continue
                
        if date_to_obj and 'datetime' in metadata:
            exp_date = datetime.strptime(metadata['date'], "%Y-%m-%d")
            if exp_date > date_to_obj:
                continue
        
        # Get all PDF files in this plots directory
        pdfs = glob.glob(f"{plots_dir}/*.pdf")
        for pdf in pdfs:
            # Store the relative path from the base_dir
            rel_path = os.path.relpath(pdf, start=os.path.dirname(base_dir))
            plot_paths.append({
                'path': pdf,
                'rel_path': rel_path,
                'experiment': experiment_name,
                'filename': os.path.basename(pdf),
                'metadata': metadata,
                'config': config
            })
    
    return plot_paths

def embed_pdf_as_base64(pdf_path):
    """Read PDF file and encode it as base64 for direct embedding in HTML."""
    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()
    
    base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
    return f'<embed src="data:application/pdf;base64,{base64_pdf}" type="application/pdf" width="100%" height="600px" />'

def generate_html_report(plot_paths, output_file, group_by='experiment'):
    """Generate an HTML report with all the plots directly embedded."""
    # Group plots based on the specified attribute
    groups = {}
    
    for plot in plot_paths:
        if group_by == 'experiment':
            key = plot['experiment']
        elif group_by == 'model_type':
            key = plot['metadata'].get('model_type', 'Unknown')
        elif group_by == 'n_classes':
            key = f"{plot['metadata'].get('n_classes', 'Unknown')}-class"
        elif group_by == 'attack_type':
            key = plot['metadata'].get('attack_type', 'Unknown')
        elif group_by == 'date':
            key = plot['metadata'].get('date', 'Unknown')
        else:
            key = plot['experiment']
            
        if key not in groups:
            groups[key] = []
        groups[key].append(plot)
    
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Adversarial Robustness Plots Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            h2 {
                color: #444;
                margin-top: 30px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
            }
            h3 {
                color: #555;
            }
            .plot-container {
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin: 20px 0;
                padding: 20px;
            }
            .plot-title {
                font-weight: bold;
                margin-bottom: 10px;
            }
            .toc {
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin: 20px 0;
                padding: 20px;
            }
            .toc ul {
                list-style-type: none;
                padding-left: 20px;
            }
            .toc a {
                text-decoration: none;
                color: #0066cc;
            }
            .toc a:hover {
                text-decoration: underline;
            }
            .metadata {
                font-size: 0.9em;
                color: #666;
                margin-top: 5px;
            }
            .filters {
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin: 20px 0;
                padding: 20px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
        </style>
    </head>
    <body>
        <h1>Adversarial Robustness Plots Report</h1>
        
        <div class="filters">
            <h2>Experiment Summary</h2>
            <p>Total experiments: <strong>""" + str(len(set(plot['experiment'] for plot in plot_paths))) + """</strong></p>
            <p>Total plots: <strong>""" + str(len(plot_paths)) + """</strong></p>
            
            <h3>Model Types</h3>
            <ul>
    """
    
    # Add model type summary
    model_types = {}
    for plot in plot_paths:
        model_type = plot['metadata'].get('model_type', 'Unknown')
        if model_type not in model_types:
            model_types[model_type] = 0
        model_types[model_type] += 1
    
    for model_type, count in sorted(model_types.items()):
        html_content += f'<li>{model_type}: {count} plots</li>\n'
    
    html_content += """
            </ul>
            
            <h3>Number of Classes</h3>
            <ul>
    """
    
    # Add class count summary
    class_counts = {}
    for plot in plot_paths:
        n_classes = plot['metadata'].get('n_classes', 'Unknown')
        if n_classes not in class_counts:
            class_counts[n_classes] = 0
        class_counts[n_classes] += 1
    
    for n_classes, count in sorted(class_counts.items()):
        html_content += f'<li>{n_classes}-class: {count} plots</li>\n'
    
    html_content += """
            </ul>
            
            <h3>Attack Types</h3>
            <ul>
    """
    
    # Add attack type summary
    attack_types = {}
    for plot in plot_paths:
        attack_type = plot['metadata'].get('attack_type', 'Unknown')
        if attack_type not in attack_types:
            attack_types[attack_type] = 0
        attack_types[attack_type] += 1
    
    for attack_type, count in sorted(attack_types.items()):
        html_content += f'<li>{attack_type}: {count} plots</li>\n'
    
    html_content += """
            </ul>
        </div>
        
        <div class="toc">
            <h2>Table of Contents</h2>
            <ul>
    """
    
    # Add table of contents
    for group in sorted(groups.keys()):
        group_id = group.replace(" ", "_").replace(".", "_").replace("-", "_")
        html_content += f'<li><a href="#{group_id}">{group}</a> ({len(groups[group])} plots)</li>\n'
    
    html_content += """
            </ul>
        </div>
    """
    
    # Add plots grouped by the specified attribute
    for group in sorted(groups.keys()):
        group_id = group.replace(" ", "_").replace(".", "_").replace("-", "_")
        html_content += f'<h2 id="{group_id}">{group}</h2>\n'
        
        # Add a table with experiment metadata for this group
        html_content += """
        <table>
            <tr>
                <th>Experiment</th>
                <th>Model Type</th>
                <th>Classes</th>
                <th>Attack Type</th>
                <th>Date</th>
            </tr>
        """
        
        # Get unique experiments for this group
        experiments = {}
        for plot in groups[group]:
            if plot['experiment'] not in experiments:
                experiments[plot['experiment']] = plot
        
        for exp_name, plot in sorted(experiments.items()):
            metadata = plot['metadata']
            html_content += f"""
            <tr>
                <td>{exp_name}</td>
                <td>{metadata.get('model_type', 'Unknown')}</td>
                <td>{metadata.get('n_classes', 'Unknown')}</td>
                <td>{metadata.get('attack_type', 'Unknown')}</td>
                <td>{metadata.get('date', 'Unknown')}</td>
            </tr>
            """
        
        html_content += """
        </table>
        """
        
        for plot in sorted(groups[group], key=lambda x: x['filename']):
            plot_name = plot['filename'].replace('.pdf', '')
            metadata = plot['metadata']
            config = plot['config']
            
            # Extract relevant config information
            model_config = config.get('model', {})
            adversarial_config = config.get('adversarial', {})
            
            print(f"Processing {plot['path']}...")
            
            html_content += f"""
            <div class="plot-container">
                <div class="plot-title">{plot_name}</div>
                {embed_pdf_as_base64(plot['path'])}
                <div class="metadata">
                    <p><strong>Path:</strong> {plot['rel_path']}</p>
                    <p><strong>Experiment:</strong> {plot['experiment']}</p>
            """
            
            if metadata:
                html_content += f"""
                    <p><strong>Model:</strong> {metadata.get('model_type', 'Unknown')}</p>
                    <p><strong>Classes:</strong> {metadata.get('n_classes', 'Unknown')}</p>
                    <p><strong>Attack:</strong> {metadata.get('attack_type', 'Unknown')}</p>
                    <p><strong>Date:</strong> {metadata.get('date', 'Unknown')}</p>
                """
            
            if model_config:
                html_content += f"""
                    <p><strong>Model Config:</strong> Hidden Dim: {model_config.get('hidden_dim', 'Unknown')}, 
                    Output Dim: {model_config.get('output_dim', 'Unknown')}</p>
                """
            
            if adversarial_config:
                html_content += f"""
                    <p><strong>Adversarial Config:</strong> Attack: {adversarial_config.get('attack_type', 'Unknown')}, 
                    Epsilons: {', '.join(map(str, adversarial_config.get('test_epsilons', [])))}</p>
                """
            
            html_content += """
                </div>
            </div>
            """
    
    html_content += """
    </body>
    </html>
    """
    
    # Write the HTML to a file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Report generated: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate a report of PDF plots from experiments with PDFs directly embedded')
    parser.add_argument('--dataset', default='mnist', help='Dataset type (default: mnist)')
    parser.add_argument('--output', default='plot_report.html', help='Output HTML file (default: plot_report.html)')
    parser.add_argument('--model-type', help='Filter by model type (e.g., cnn, mlp)')
    parser.add_argument('--n-classes', type=int, help='Filter by number of classes')
    parser.add_argument('--attack-type', help='Filter by attack type (e.g., fgsm, pgd)')
    parser.add_argument('--date-from', help='Filter by date from (YYYY-MM-DD)')
    parser.add_argument('--date-to', help='Filter by date to (YYYY-MM-DD)')
    parser.add_argument('--group-by', default='experiment', choices=['experiment', 'model_type', 'n_classes', 'attack_type', 'date'],
                        help='Group plots by this attribute in the report')
    args = parser.parse_args()
    
    base_dir = f"results/adversarial_robustness/{args.dataset}"
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    # Find all plots with filters
    plot_paths = find_plot_pdfs(
        base_dir,
        model_type=args.model_type,
        n_classes=args.n_classes,
        attack_type=args.attack_type,
        date_from=args.date_from,
        date_to=args.date_to
    )
    
    print(f"Found {len(plot_paths)} PDF plots:")
    for plot in plot_paths:
        print(f"- {plot['rel_path']}")
    
    # Generate the HTML report with the specified grouping
    generate_html_report(plot_paths, args.output, group_by=args.group_by)

if __name__ == "__main__":
    main() 