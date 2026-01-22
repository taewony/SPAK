import os
import markdown
import re

HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>__TITLE__</title>
    <!-- MathJax Configuration -->
    <script>
      window.MathJax = {
        tex: {
          inlineMath: [['$', '$'], ['\\(', '\\)']]),
          displayMath: [['$$', '$$'], ['\\[', '\\]']]
        }
      };
    </script>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {
            font-family: 'Times New Roman', Times, serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px;
            background-color: #f9f9f9;
        }
        .paper-content {
            background: white;
            padding: 50px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 4px;
        }
        h1 {
            text-align: center;
            font-size: 2.2em;
            margin-bottom: 40px;
            color: #111;
        }
        h2 {
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
            margin-top: 40px;
            color: #222;
        }
        h3 {
            margin-top: 30px;
            color: #333;
        }
        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-left: 5px solid #ccc;
            overflow-x: auto;
            font-family: 'Courier New', Courier, monospace;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
        }
        table, th, td {
            border: 1px solid #ddd;
        }
        th, td {
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        .abstract {
            font-style: italic;
            margin-bottom: 30px;
            padding: 20px;
            background: #fdfdfd;
            border: 1px solid #eee;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            font-size: 0.9em;
            color: #777;
        }
    </style>
</head>"""

BODY = """<body>
    <div class="paper-content">
        __CONTENT__
    </div>
    <div class="footer">
        &copy; 2026 SPAK Research Team. Generated from Markdown by SPAK Agent.
    </div>
</body>
</html>
"""

def convert_to_html(md_path, html_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    html_body = markdown.markdown(md_text, extensions=['extra', 'codehilite', 'toc'])

    html_body = re.sub(r'<h2>Abstract</h2>\s*<p>(.*?)</p>', 
                       r'<h2>Abstract</h2><div class="abstract"><p>\1</p></div>', 
                       html_body, flags=re.DOTALL)

    title_match = re.search(r'<h1>(.*?)</h1>', html_body)
    title = title_match.group(1) if title_match else "SPAK Research Paper"

    # Simple string replacement
    final_html = (HEADER + BODY).replace("__TITLE__", title).replace("__CONTENT__", html_body)

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    print(f"✅ Successfully converted {md_path} to {html_path}")

if __name__ == "__main__":
    convert_to_html("papers/paper_gemini_reviewed.md", "whitepaper/SPAK_Final_Paper.html")
