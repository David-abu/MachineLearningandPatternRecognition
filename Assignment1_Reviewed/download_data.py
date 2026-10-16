from pathlib import Path
import urllib.request,zipfile,io,hashlib,json
root=Path(__file__).resolve().parent
for d in ['data','out','figs','logs']: (root/d).mkdir(exist_ok=True)
manifest=[]
for name,url in [('wine','https://archive.ics.uci.edu/static/public/186/wine+quality.zip'),('har','https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip')]:
 target=root/'data'/f'{name}_source.zip'
 if not target.exists():
  print('Downloading',url,flush=True)
  with urllib.request.urlopen(url,timeout=120) as response: target.write_bytes(response.read())
 raw=target.read_bytes(); z=zipfile.ZipFile(io.BytesIO(raw))
 if name=='wine':
  entry=next(n for n in z.namelist() if n.endswith('winequality-white.csv'))
  (root/'data'/'wine_white.csv').write_bytes(z.read(entry))
 else:
  nested=[n for n in z.namelist() if n.endswith('.zip')]
  if nested: z=zipfile.ZipFile(io.BytesIO(z.read(nested[0])))
  dest=root/'data'/'har'; dest.mkdir(exist_ok=True)
  for base in ['X_train.txt','X_test.txt','y_train.txt','y_test.txt','activity_labels.txt']:
   entry=next(n for n in z.namelist() if n.endswith('/'+base) or n==base)
   (dest/base).write_bytes(z.read(entry))
 manifest.append(dict(dataset=name,url=url,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
 print(name,len(raw),'archive bytes; extracted',flush=True)
(root/'data'/'sources.json').write_text(json.dumps(manifest,indent=2))
