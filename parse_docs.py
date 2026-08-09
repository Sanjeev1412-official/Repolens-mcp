import re
import sys

def extract():
    with open(r'C:\Users\sande\.gemini\antigravity-ide\brain\1867e5f7-b612-4608-bb59-937a5485e8bf\.system_generated\steps\760\content.md', 'r', encoding='utf-8') as f:
        html = f.read()
    
    # The actual content is often in a JSON-like structure in Next.js pages (self.__next_f.push)
    # Let's just find everything between <p> or <h1> etc if it's HTML, but it's a React page.
    # Let's extract all text content from React's static props or children strings
    matches = re.findall(r'children:?`([^`]*)`', html)
    for m in matches:
        print(m[:200])

if __name__ == "__main__":
    extract()
