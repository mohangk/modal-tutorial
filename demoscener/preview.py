"""Write a self-contained, click-to-run Dwitter preview for saved completions."""

import html
import json
from pathlib import Path


def write_preview(output_dir: Path, results: list[dict]) -> None:
    cards = []
    for result in results:
        # Programs run only when clicked, inside a frame without same-origin access.
        program = json.dumps(result["code"]).replace("<", "\\u003c")
        frame = """<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval'; style-src 'unsafe-inline'">
<style>body{margin:0;background:white}canvas{width:100%;display:block}pre{color:#b00;white-space:pre-wrap}</style>
</head><body><canvas id="c" width="1920" height="1080"></canvas><pre id="error"></pre><script>
var c=document.getElementById('c'),x=c.getContext('2d');
var S=Math.sin,C=Math.cos,T=Math.tan;
var R=(r,g,b,a=1)=>`rgba(${r|0},${g|0},${b|0},${a})`;
var u,start;
try{u=new Function('t',PROGRAM)}catch(e){document.getElementById('error').textContent=String(e)}
function frame(now){
  if(start===undefined)start=now;
  try{u((now-start)/1000)}catch(e){document.getElementById('error').textContent=String(e);return}
  requestAnimationFrame(frame);
}
if(u)requestAnimationFrame(frame);
</script></body></html>""".replace("PROGRAM", program)
        cards.append(
            f'<article><h2>{html.escape(result["stage"])}</h2>'
            f'<p>{result["tokens"]} new tokens · {result["seconds"]} seconds</p>'
            '<button class="run">Run animation</button> '
            '<button class="stop">Stop</button>'
            f'<iframe sandbox="allow-scripts" data-source="{html.escape(frame, quote=True)}" '
            'title="Animation preview"></iframe>'
            f'<pre>{html.escape(result["code"])}</pre></article>'
        )
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Demoscener stage comparison</title><style>
body{font:16px system-ui;background:#f4f5f7;color:#18222f;margin:32px auto;padding:0 20px;max-width:1200px}
h1{margin-bottom:8px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}
article{background:white;border:1px solid #d6dbe1;border-radius:10px;padding:20px}h2{margin-top:0}
button{padding:8px 12px;cursor:pointer}iframe{display:block;width:100%;aspect-ratio:16/9;border:1px solid #ddd;margin-top:16px;background:white}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}p{line-height:1.6}
</style></head><body><h1>Demoscener: compare training stages</h1>
<p>Same prefix, different frozen weights. Judge runtime success and visual preference separately.
These completions have not been repaired. Truncated or invalid programs may fail.</p>
<p>Click Run to render a program in a sandboxed frame. Stop removes it.
A program with an infinite loop can still stall its browser tab.</p>
<main>CARDS</main><script>
document.querySelectorAll('article').forEach(card=>{
  const frame=card.querySelector('iframe');
  card.querySelector('.run').onclick=()=>frame.srcdoc=frame.dataset.source;
  card.querySelector('.stop').onclick=()=>frame.removeAttribute('srcdoc');
});
</script></body></html>""".replace("CARDS", "\n".join(cards))
    (output_dir / "viewer.html").write_text(page)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    write_preview(args.results.parent, json.loads(args.results.read_text())["results"])
